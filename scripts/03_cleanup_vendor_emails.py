#!/usr/bin/env python3
"""
STEP 3 — Cleanup & Reduction (Contact-Preserving)

Removes conversation noise while preserving ALL vendor identity
and contact information.

NO AI usage. Deterministic only.
"""
import time
import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

# -------------------------
# Logging setup
# -------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("cleanup_vendor_emails")

# Progress heartbeat frequency (safe to increase to reduce log volume)
PROGRESS_EVERY_N = 500

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
# Contact detection (CRITICAL)
# -------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"""
    (?:
        \+?\d{1,3}[\s.-]?
    )?
    (?:\(?\d{2,4}\)?[\s.-]?)?
    \d{3,4}[\s.-]?\d{3,4}
""", re.VERBOSE)

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

def contains_contact_info(text: str) -> bool:
    return bool(
        EMAIL_RE.search(text)
        or PHONE_RE.search(text)
        or URL_RE.search(text)
    )

# -------------------------
# Cleanup logic
# -------------------------

QUOTED_REPLY_PATTERNS = [
    re.compile(r"^>+", re.MULTILINE),
    re.compile(r"^On .* wrote:$", re.MULTILINE),
]

FORWARDED_MARKERS = [
    re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}", re.IGNORECASE),
    re.compile(r"^Begin forwarded message:", re.IGNORECASE),
]

SIGNATURE_DELIMITERS = [
    re.compile(r"^--\s*$", re.MULTILINE),
    re.compile(r"^Regards,?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Thanks,?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Best regards,?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Sincerely,?$", re.MULTILINE | re.IGNORECASE),
]

DISCLAIMER_KEYWORDS = [
    "confidential",
    "intended recipient",
    "privileged",
    "legal disclaimer",
]

# -------------------------
# Cleanup operations
# -------------------------

def remove_quoted_and_forwarded(text: str) -> Tuple[str, bool]:
    original = text

    for pat in FORWARDED_MARKERS + QUOTED_REPLY_PATTERNS:
        split = pat.split(text, maxsplit=1)
        if len(split) > 1:
            logger.debug("Removing quoted/forwarded block")
            text = split[0]

    return text.strip(), text != original


def split_signature(text: str) -> Tuple[str, str, bool]:
    """
    Extract signature WITHOUT deleting it.
    """
    for pat in SIGNATURE_DELIMITERS:
        parts = pat.split(text, maxsplit=1)
        if len(parts) > 1:
            logger.debug("Signature delimiter detected")
            body = parts[0].strip()
            signature = parts[1].strip()
            return body, signature, True

    return text.strip(), "", False


def remove_disclaimer_safely(text: str) -> Tuple[str, bool]:
    """
    Remove disclaimer ONLY if it contains no contact info.
    """
    lower = text.lower()
    for kw in DISCLAIMER_KEYWORDS:
        idx = lower.find(kw)
        if idx != -1:
            tail = text[idx:]
            if not contains_contact_info(tail):
                logger.debug("Removing disclaimer block without contact info")
                return text[:idx].strip(), True
            logger.debug("Disclaimer retained due to contact info")
    return text, False


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def truncate_body_only(body: str, max_chars: int = 2000) -> str:
    if len(body) <= max_chars:
        return body
    logger.debug("Truncating body text to max_chars=%d", max_chars)
    cut = body[:max_chars]
    last_period = cut.rfind(".")
    if last_period > 200:
        return cut[: last_period + 1]
    return cut


def cleanup_text(raw_text: str) -> Dict[str, Any]:
    meta = {
        "removed_quoted_blocks": False,
        "removed_disclaimer": False,
        "signature_extracted": False,
        "contact_preserved": True,
    }

    text = raw_text or ""

    text, changed = remove_quoted_and_forwarded(text)
    meta["removed_quoted_blocks"] = changed

    body, signature, changed = split_signature(text)
    meta["signature_extracted"] = changed

    body, changed = remove_disclaimer_safely(body)
    meta["removed_disclaimer"] = changed

    body = normalize_whitespace(body)
    signature = normalize_whitespace(signature)

    body = truncate_body_only(body)

    cleaned = body
    if signature:
        cleaned = f"{body}\n\n--- SIGNATURE ---\n{signature}"

    return {
        "cleaned_text": cleaned,
        "signature_text": signature,
        "meta": meta,
    }

# -------------------------
# Registry handling (unchanged)
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
    start_ts = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    candidates_dir = Path(args.candidates_dir)
    output_dir = Path(args.output_dir)
    state_dir = Path(args.state_dir)
    registry_path = state_dir / "processing_registry_step3_cleanup.jsonl"

    registry = load_registry(registry_path)

    processed = 0
    failed = 0
    skipped_total = 0
    skipped_already_processed = 0
    output_written = 0
    total_input = 0

    logger.info(
        "STEP 3 START: candidates_dir=%s output_dir=%s limit=%s",
        candidates_dir,
        output_dir,
        args.limit or "none",
    )

    for json_path in candidates_dir.rglob("*.json"):
        total_input += 1

        if args.limit and processed >= args.limit:
            logger.info("Processing limit reached (%d); stopping early", args.limit)
            break

        try:
            with json_path.open("r", encoding="utf-8") as f:
                email = json.load(f)

            email_id = email["email_id"]
            content_hash = email["content_hash"]

            reg = registry.get(email_id)
            if reg and reg.get("last_completed_step") == "step3_cleanup" and reg.get("content_hash") == content_hash:
                skipped_total += 1
                skipped_already_processed += 1
                logger.debug("Skipping %s: already processed with same content_hash", email_id)
                continue

            raw_text = email.get("body", {}).get("raw_text") or ""

            cleanup = cleanup_text(raw_text)

            output = {
                "email_id": email_id,
                "content_hash": content_hash,
                "source_email_path": str(json_path),
                "cleaned_text": cleanup["cleaned_text"],
                "signature_text": cleanup["signature_text"],
                "cleanup_meta": {
                    "original_length": len(raw_text),
                    "cleaned_length": len(cleanup["cleaned_text"]),
                    **cleanup["meta"],
                },
                "processing": {
                    "cleaned_at": utc_now(),
                    "schema_version": 2,
                },
            }

            out_path = shard_path(output_dir, email_id)
            ensure_parent(out_path)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            output_written += 1

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
            logger.warning("Recoverable failure for %s: %s", json_path, e)
            append_registry(
                registry_path,
                {
                    "email_id": email_id if "email_id" in locals() else "unknown",
                    "error": str(e),
                    "last_completed_step": "step3_cleanup_error",
                    "timestamp": utc_now(),
                },
            )

        if total_input % PROGRESS_EVERY_N == 0:
            logger.info(
                "Progress: seen=%d processed=%d skipped=%d (already_processed=%d) failed=%d",
                total_input,
                processed,
                skipped_total,
                skipped_already_processed,
                failed,
            )

    elapsed = time.time() - start_ts
    invariant_ok = total_input == processed + skipped_total + failed

    logger.info(
        "STEP 3 COMPLETE: total_input=%d processed=%d skipped=%d failed=%d output_written=%d invariant_ok=%s elapsed_sec=%.2f",
        total_input,
        processed,
        skipped_total,
        failed,
        output_written,
        invariant_ok,
        elapsed,
    )

    logger.info(
        "Logging performance notes: DEBUG logs inside the hot loop are safe to disable. "
        "Increase PROGRESS_EVERY_N to reduce log volume during large runs."
    )

    return 0


if __name__ == "__main__":
    while True:
        main()
        logger.warning("Cleanup job sleeping for 5 minutes (heartbeat)")
        time.sleep(300)
