#!/usr/bin/env python3
"""
Step 2A — Heuristic prefilter: score emails and write decision log + pass/fail outputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import yaml
except Exception:
    yaml = None  # we will error with helpful message if config is YAML and pyyaml not installed

from registry import append_registry_entry, get_registry_entry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prefilter")


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
    reasons: List[str] = []
    score = 0

    pre = config.get("prefilter", {})
    subject_weight = float(pre.get("subject_weight", 1.0))
    body_weight = float(pre.get("body_weight", 0.5))

    subject = (email.get("headers", {}).get("Subject") or "").lower()
    body = (email.get("body", {}).get("raw_text") or "").lower()

    # ---- POSITIVE KEYWORDS ----
    for group_name, group in pre.get("positive_keywords", {}).items():
        weight = int(group.get("weight", 0))
        terms = group.get("terms", [])

        for term in terms:
            t = term.lower()
            if t in subject:
                delta = int(weight * subject_weight)
                score += delta
                reasons.append(f"{group_name}:subject:{term}")
            elif t in body:
                delta = int(weight * body_weight)
                score += delta
                reasons.append(f"{group_name}:body:{term}")

    # ---- ATTACHMENTS ----
    att_cfg = pre.get("attachments", {})
    mime = email.get("mime_meta", {})
    if mime.get("has_attachments"):
        w = int(att_cfg.get("has_attachment_weight", 0))
        score += w
        reasons.append("attachment:present")

        for att in mime.get("attachments", []):
            fname = (att.get("filename") or "").lower()
            for kw in att_cfg.get("filename_keywords", []):
                if kw.lower() in fname:
                    score += int(att_cfg.get("filename_keyword_weight", 0))
                    reasons.append(f"attachment:filename:{kw}")

    # ---- NEGATIVE KEYWORDS ----
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


def process(input_dir: Path, output_dir: Path, registry_path: str, decision_log_path: Path, config: Dict[str, Any], limit: int = 0, only_prefix: Optional[str] = None, dry_run: bool = False):
    # gather files
# gather files (mailbox -> shard -> email.json)
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
    logger.info("Prefilter: found %d candidate files", len(files))
    if limit and limit > 0:
        files = files[:limit]

    threshold = int(config.get("prefilter", {}).get("threshold", 5))
    processed = 0
    for f in files:
        try:
            email = load_email_json(f)
            email_id = email.get("email_id")
            content_hash = email.get("content_hash")
            if not email_id:
                logger.warning("Skipping file without email_id: %s", f)
                continue

            registry_entry = get_registry_entry(registry_path, email_id)
            if registry_entry and registry_entry.get("content_hash") == content_hash and registry_entry.get("last_completed_step") == "step2a_prefilter":
                logger.debug("Skipping unchanged %s", email_id)
                continue

            score, reasons = score_email(email, config)
            passed = score >= threshold
            decision = {
                "email_id": email_id,
                "score": score,
                "passed": passed,
                "reasons": reasons,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            if dry_run:
                logger.info("[dry-run] %s -> score=%d passed=%s reasons=%s", email_id, score, passed, reasons)
            else:
                # write decision log
                ensure_dir(str(decision_log_path.parent))
                with open(decision_log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(decision, ensure_ascii=False) + "\n")

                # if passed, copy to output_dir preserving subdir
                if passed:
                    sub = email_id[:2]
                    dst = output_dir / sub / f"{email_id}.json"
                    atomic_copy(f, dst)

                # update registry
                append_registry_entry(registry_path, {
                    "email_id": email_id,
                    "content_hash": content_hash,
                    "last_completed_step": "step2a_prefilter",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
                logger.info("Processed %s -> score=%d passed=%s", email_id, score, passed)

            processed += 1
        except Exception as e:
            logger.exception("Failed to prefilter %s: %s", f, e)
    logger.info("Prefilter processed %d files", processed)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Input directory (data/emails_raw_json)")
    parser.add_argument("--output-dir", required=True, help="Output directory for passed emails (data/emails_prefiltered)")
    parser.add_argument("--state-dir", required=True, help="State directory (data/state)")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=0, help="Process only N files (0 = all)")
    parser.add_argument("--only-prefix", default=None, help="Process only files whose filename starts with this prefix")
    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs; show planned actions")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    state_dir = Path(args.state_dir)
    ensure_dir(str(state_dir))
    decision_log_path = Path(state_dir) / "step2a_prefilter_decisions.jsonl"
    registry_path = str(state_dir / "processing_registry.jsonl")

    process(input_dir, output_dir, registry_path, decision_log_path, cfg, limit=args.limit, only_prefix=args.only_prefix, dry_run=args.dry_run)


if __name__ == "__main__":
    main()


