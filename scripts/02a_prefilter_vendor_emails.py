#!/usr/bin/env python3
"""
Step 2A — Heuristic prefilter: score emails and write decision log + pass/fail outputs.
"""
from __future__ import annotations
import time
import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

try:
    import yaml
except Exception:
    yaml = None  # we will error with helpful message if config is YAML and pyyaml not installed

from registry import append_registry_entry

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("prefilter2a")

# ---------- logging config ----------
PROGRESS_EVERY_N = 500  # INFO heartbeat frequency


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        path = "config/config.yaml"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.endswith(".yaml") or path.endswith(".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config. Install with `pip install pyyaml`.")
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def load_email_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def score_email(email: Dict[str, Any], config: Dict[str, Any]) -> (int, List[str]):
    pre = config.get("prefilter", {})
    threshold = int(pre.get("threshold", 999))

    reasons: List[str] = []
    score = 0

    subject_weight = float(pre.get("subject_weight", 1.0))
    body_weight = float(pre.get("body_weight", 0.5))

    subject = (email.get("headers", {}).get("Subject") or "").lower()
    body = (email.get("body", {}).get("raw_text") or "").lower()[:5000]

    for group_name, group in pre.get("positive_keywords", {}).items():
        weight = int(group.get("weight", 0))
        for term in group.get("terms", []):
            t = term.lower()
            if t in subject:
                score += int(weight * subject_weight)
                reasons.append(f"{group_name}:subject:{term}")
            elif t in body:
                score += int(weight * body_weight)
                reasons.append(f"{group_name}:body:{term}")
            if score >= threshold:
                return score, reasons

    att_cfg = pre.get("attachments", {})
    mime = email.get("mime_meta", {})
    if mime.get("has_attachments"):
        score += int(att_cfg.get("has_attachment_weight", 0))
        reasons.append("attachment:present")
        if score >= threshold:
            return score, reasons
        for att in mime.get("attachments", []):
            fname = (att.get("filename") or "").lower()
            for kw in att_cfg.get("filename_keywords", []):
                if kw.lower() in fname:
                    score += int(att_cfg.get("filename_keyword_weight", 0))
                    reasons.append(f"attachment:filename:{kw}")
                    if score >= threshold:
                        return score, reasons

    for group_name, group in pre.get("negative_keywords", {}).items():
        weight = int(group.get("weight", 0))
        for term in group.get("terms", []):
            t = term.lower()
            if t in subject or t in body:
                score += weight
                reasons.append(f"negative:{group_name}:{term}")

    return score, reasons


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def process(
    input_dir: Path,
    output_dir: Path,
    registry_path: str,
    decision_log_path: Path,
    config: Dict[str, Any],
    limit: int = 0,
    only_prefix: Optional[str] = None,
    dry_run: bool = False,
):
    step_start = time.monotonic()

    # ---------- counters ----------
    total_input = 0
    processed = 0
    failed = 0
    skipped_total = 0
    skipped_reasons = Counter()
    output_written = 0

    # ---------- registry ----------
    registry_cache: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if "email_id" in r:
                        registry_cache[r["email_id"]] = r
                except Exception:
                    continue

    # ---------- gather files ----------
    files: List[Path] = []
    for mailbox_dir in input_dir.iterdir():
        if not mailbox_dir.is_dir():
            continue
        for shard_dir in mailbox_dir.iterdir():
            if not shard_dir.is_dir():
                continue
            for f in shard_dir.iterdir():
                if f.is_file() and f.suffix == ".json":
                    if only_prefix and not f.name.startswith(only_prefix):
                        continue
                    files.append(f)

    files.sort()
    if limit and limit > 0:
        files = files[:limit]

    total_expected = len(files)
    threshold = int(config.get("prefilter", {}).get("threshold", 5))

    logger.info(
        "step_start step=step2a_prefilter input_dir=%s total_input_expected=%d threshold=%d dry_run=%s limit=%s",
        input_dir,
        total_expected,
        threshold,
        dry_run,
        limit or "none",
    )

    for idx, f in enumerate(files, start=1):
        total_input += 1
        try:
            email = load_email_json(f)
            email_id = email.get("email_id")
            content_hash = email.get("content_hash")

            if not email_id:
                skipped_total += 1
                skipped_reasons["missing_email_id"] += 1
                logger.debug("record_skipped reason=missing_email_id path=%s", f)
                continue

            registry_entry = registry_cache.get(email_id)
            if (
                registry_entry
                and registry_entry.get("content_hash") == content_hash
                and registry_entry.get("last_completed_step") == "step2a_prefilter"
            ):
                skipped_total += 1
                skipped_reasons["unchanged"] += 1
                logger.debug("record_skipped reason=unchanged email_id=%s", email_id)
                continue

            score, reasons = score_email(email, config)
            passed = score >= threshold

            decision = {
                "email_id": email_id,
                "score": score,
                "passed": passed,
                "reasons": reasons,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            if dry_run:
                logger.debug(
                    "[dry-run] decision email_id=%s score=%d passed=%s",
                    email_id,
                    score,
                    passed,
                )
            else:
                ensure_dir(str(decision_log_path.parent))
                with open(decision_log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(decision, ensure_ascii=False) + "\n")

                if passed:
                    sub = email_id[:2]
                    dst = output_dir / sub / f"{email_id}.json"
                    atomic_copy(f, dst)
                    output_written += 1

                append_registry_entry(registry_path, {
                    "email_id": email_id,
                    "content_hash": content_hash,
                    "last_completed_step": "step2a_prefilter",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })

            processed += 1

        except Exception as e:
            failed += 1
            logger.warning(
                "record_failed step=step2a_prefilter path=%s error=%s",
                f,
                e,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )

        if total_input % PROGRESS_EVERY_N == 0:
            logger.info(
                "step_progress step=step2a_prefilter seen=%d/%d processed=%d skipped=%d failed=%d skipped_breakdown=%s",
                total_input,
                total_expected,
                processed,
                skipped_total,
                failed,
                dict(skipped_reasons),
            )

    elapsed = time.monotonic() - step_start
    invariant_ok = total_input == processed + skipped_total + failed

    logger.info(
        "step_summary step=step2a_prefilter total_input=%d processed=%d skipped_total=%d failed=%d output_written=%d skipped_breakdown=%s invariant_ok=%s elapsed_s=%.2f",
        total_input,
        processed,
        skipped_total,
        failed,
        output_written,
        dict(skipped_reasons),
        invariant_ok,
        elapsed,
    )

    if not invariant_ok:
        logger.error(
            "step_invariant_violation step=step2a_prefilter lhs_total_input=%d rhs_sum=%d",
            total_input,
            processed + skipped_total + failed,
        )

    logger.info(
        "logging_hints step=step2a_prefilter "
        "Disable DEBUG to remove per-record diagnostics; "
        "increase PROGRESS_EVERY_N to reduce INFO volume; "
        "decision scoring is unaffected by logging configuration."
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    state_dir = Path(args.state_dir)
    ensure_dir(str(state_dir))
    decision_log_path = Path(state_dir) / "step2a_prefilter_decisions.jsonl"
    registry_path = str(state_dir / "processing_registry.jsonl")

    process(
        input_dir,
        output_dir,
        registry_path,
        decision_log_path,
        cfg,
        limit=args.limit,
        only_prefix=args.only_prefix,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    while True:
        main()
        logger.warning("Prefilter idle: sleeping for 5 minutes")
        time.sleep(300)
