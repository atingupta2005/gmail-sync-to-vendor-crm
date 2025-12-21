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
    """
    Compute a simple integer score and return reasons.
    Config expected to contain prefilter.keyword_weights mapping and prefilter.threshold.
    """
    reasons: List[str] = []
    score = 0
    weights = config.get("prefilter", {}).get("keyword_weights", {})
    subject = (email.get("headers", {}).get("Subject") or "") or ""
    body = email.get("body", {}).get("raw_text") or ""
    text = f"{subject}\n{body}".lower()
    for kw, w in weights.items():
        if kw.lower() in text:
            score += int(w)
            reasons.append(f"keyword:{kw}")

    # attachment boost
    if email.get("mime_meta", {}).get("has_attachments"):
        score += int(config.get("prefilter", {}).get("attachment_boost", 1))
        reasons.append("has_attachment")

    # automated sender negative signal (noreply)
    sender = (email.get("headers", {}).get("From") or "") or ""
    if "noreply" in sender.lower() or "no-reply" in sender.lower() or "do-not-reply" in sender.lower():
        score -= 5
        reasons.append("automated_sender")

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
    files: List[Path] = []
    for sub in input_dir.iterdir():
        if sub.is_dir():
            for f in sub.iterdir():
                if f.is_file():
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


