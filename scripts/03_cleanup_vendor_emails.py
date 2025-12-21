#!/usr/bin/env python3
"""
STEP 3 — Cleanup & Reduction

Cleans vendor-candidate emails to remove noise (quoted replies, signatures,
disclaimers, HTML junk) and produce high-signal text for downstream AI steps.

NO AI usage. Deterministic only.
"""

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

LOG = logging.getLogger("step3_cleanup")


# -------------------------
# Utility helpers
# -------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def shard_path(base: Path, email_id: str) -> Path:
    return base / email_id[:2] / f"{email_id}.json"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# -------------------------
# Cleanup logic
# -------------------------

QUOTED_PATTERNS = [
    re.compile(r"^>+", re.MULTILINE),
    re.compile(r"^On .* wrote:$", re.MULTILINE),
    re.compile(r"^From: .*", re.MULTILINE),
    re.compile(r"^Sent: .*", re.MULTILINE),
    re.compile(r"^To: .*", re.MULTILINE),
    re.compile(r"^Subject: .*", re.MULTILINE),
]

SIGNATURE_SPLIT_PATTERNS = [
    re.compile(r"^--\s*$", re.MULTILINE),
    re.compile(r"^Regards,?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Thanks,?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Best regards,?$", re.MULTILINE | re.IGNORECASE),
]

DISCLAIMER_KEYWORDS = [
    "confidential",
    "intended recipient",
    "privileged",
    "legal disclaimer",
]


def remove_quoted_blocks(text: str) -> (str, bool):
    original = text
    for pat in QUOTED_PATTERNS:
        text = pat.split(text)[0]
    return text.strip(), text != original


def remove_signature(text: str) -> (str, bool):
    original = text
    for pat in SIGNATURE_SPLIT_PATTERNS:
        parts = pat.split(text, maxsplit=1)
        if len(parts) > 1:
            text = parts[0]
            break
    return text.strip(), text != original


def remove_disclaimer(text: str) -> (str, bool):
    lower = text.lower()
    for kw in DISCLAIMER_KEYWORDS:
        idx = lower.find(kw)
        if idx != -1:
            return text[:idx].strip(), True
    return text, False


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def cleanup_text(raw_text: str) -> Dict[str, Any]:
    meta = {
        "removed_quoted_blocks": False,
        "removed_signature": False,
        "removed_disclaimer": False,
    }

    text = raw_text or ""

    text, changed = remove_quoted_blocks(text)
    meta["removed_quoted_blocks"] = changed

    text, changed = remove_signature(text)
    meta["removed_signature"] = changed

    text, changed = remove_disclaimer(text)
    meta["removed_disclaimer"] = changed

    text = normalize_whitespace(text)

    return {
        "cleaned_text": text,
        "meta": meta,
    }


def truncate_text(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(".")
    if last_period > 200:
        return cut[: last_period + 1]
    return cut


# -------------------------
# Registry handling
# -------------------------

def load_registry(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    registry = {}
    if not registry_path.exists():
        return registry
    with registry_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                registry[rec["email_id"]] = rec
            except Exception:
                continue
    return registry


def append_registry(registry_path: Path, record: Dict[str, Any]) -> None:
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------------------
# Main
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    candidates_dir = Path(args.candidates_dir)
    output_dir = Path(args.output_dir)
    state_dir = Path(args.state_dir)
    registry_path = state_dir / "processing_registry.jsonl"

    registry = load_registry(registry_path)

    processed = skipped = failed = 0

    for json_path in candidates_dir.rglob("*.json"):
        if args.limit and processed >= args.limit:
            break

        try:
            with json_path.open("r", encoding="utf-8") as f:
                email = json.load(f)

            email_id = email["email_id"]
            content_hash = email["content_hash"]

            reg = registry.get(email_id)
            if reg and reg.get("last_completed_step") == "step3_cleanup" and reg.get("content_hash") == content_hash:
                skipped += 1
                continue

            body = email.get("body", {})
            raw_text = body.get("raw_text") or ""

            cleanup = cleanup_text(raw_text)
            cleaned_text = truncate_text(cleanup["cleaned_text"])

            output = {
                "email_id": email_id,
                "content_hash": content_hash,
                "source_email_path": str(json_path),
                "cleaned_text": cleaned_text,
                "cleanup_meta": {
                    "original_length": len(raw_text),
                    "cleaned_length": len(cleaned_text),
                    **cleanup["meta"],
                },
                "processing": {
                    "cleaned_at": utc_now(),
                    "schema_version": 1,
                },
            }

            out_path = shard_path(output_dir, email_id)
            ensure_parent(out_path)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            append_registry(
                registry_path,
                {
                    "email_id": email_id,
                    "content_hash": content_hash,
                    "last_completed_step": "step3_cleanup",
                    "timestamp": utc_now(),
                },
            )

            processed += 1

        except Exception as e:
            failed += 1
            append_registry(
                registry_path,
                {
                    "email_id": email_id if "email_id" in locals() else "unknown",
                    "error": str(e),
                    "last_completed_step": "step3_cleanup_error",
                    "timestamp": utc_now(),
                },
            )
            LOG.error("cleanup failed for %s: %s", json_path, e)

    LOG.info(
        "Done: processed=%d skipped=%d failed=%d",
        processed,
        skipped,
        failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
