#!/usr/bin/env python3
"""
Step 1 — Ingest Maildir (cur/, new/) and write one JSON per email.
Incremental, idempotent, and fast.
"""
from __future__ import annotations
import time
import argparse
import hashlib
import json
import logging
import os
import re
from datetime import datetime
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter

from registry import append_registry_entry

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("ingest")

# ---------- logging config ----------
PROGRESS_EVERY_N = 500  # INFO heartbeat frequency (safe to increase)

# ---------- helpers ----------

def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def normalize_headers_for_hash(headers: Dict[str, str]) -> str:
    return "\n".join(f"{k.lower()}:{v.strip()}" for k, v in sorted(headers.items()))


def extract_body(msg) -> Dict[str, Optional[str]]:
    text = None
    html = None

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            try:
                if ctype == "text/plain" and text is None:
                    text = part.get_content().strip()
                    break
                elif ctype == "text/html" and html is None:
                    html = part.get_content()
            except Exception:
                pass
    else:
        try:
            if msg.get_content_type() == "text/plain":
                text = msg.get_content()
            elif msg.get_content_type() == "text/html":
                html = msg.get_content()
        except Exception:
            pass

    if text:
        return {"raw_text": text, "raw_html": html}
    if html:
        plain = re.sub(r"<[^>]+>", "", html)
        return {"raw_text": plain, "raw_html": html}
    return {"raw_text": "", "raw_html": None}


def mime_meta_from_msg(msg) -> Dict[str, Any]:
    attachments = []

    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        if filename:
            payload = part.get_payload(decode=True)
            attachments.append({
                "filename": filename,
                "content_type": part.get_content_type(),
                "size": len(payload) if payload else None
            })

    return {
        "has_attachments": bool(attachments),
        "attachments": attachments
    }


def compute_email_id_and_hash(raw_bytes: bytes, msg) -> (str, str):
    message_id = msg.get("Message-ID") or msg.get("Message-Id")
    if message_id:
        email_id = sha1_hex(message_id.encode("utf-8"))
    else:
        headers = {k: v for k, v in msg.items()}
        norm = normalize_headers_for_hash(headers).encode("utf-8") + b"\n" + raw_bytes
        email_id = sha1_hex(norm)

    return email_id, sha1_hex(raw_bytes)


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    tmp.replace(path)


# ---------- main ingestion ----------

def process_maildir(
    maildir_root: Path,
    output_dir: Path,
    registry_path: Path,
    folder_label: str,
    registry_cache: Dict[str, Dict[str, Any]],
    limit: int = 0,
    only_prefix: Optional[str] = None,
    dry_run: bool = False,
):
    step_start = time.monotonic()

    # ----- counters -----
    total_input = 0
    processed = 0
    failed = 0
    skipped_total = 0
    skipped_reasons = Counter()
    output_written = 0

    files: List[Path] = []

    for sub in ("cur", "new"):
        p = maildir_root / sub
        if not p.exists():
            continue
        for f in p.iterdir():
            if f.is_file():
                if only_prefix and not f.name.startswith(only_prefix):
                    continue
                files.append(f)

    files.sort()

    logger.info(
        "step_start step=step1_ingest maildir=%s discovered_files=%d dry_run=%s limit=%s",
        maildir_root,
        len(files),
        dry_run,
        limit or "none",
    )

    if limit:
        files = files[:limit]

    total_expected = len(files)
    logger.info(
        "step_input step=step1_ingest total_input_expected=%d",
        total_expected,
    )

    for idx, fpath in enumerate(files, start=1):
        total_input += 1

        try:
            stat = fpath.stat()
            fast_key = f"{fpath.name}:{stat.st_mtime}:{stat.st_size}"

            prev = registry_cache.get(fast_key)
            if prev:
                skipped_total += 1
                skipped_reasons["fast_key_hit"] += 1
                # logger.debug(
                #     "record_skipped reason=fast_key_hit path=%s",
                #     fpath,
                # )
                continue

            raw_bytes = fpath.read_bytes()
            msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
            email_id, content_hash = compute_email_id_and_hash(raw_bytes, msg)

            prev = registry_cache.get(email_id)
            if prev and prev.get("content_hash") == content_hash:
                skipped_total += 1
                skipped_reasons["content_hash_match"] += 1
                # logger.debug(
                #     "record_skipped reason=content_hash_match email_id=%s path=%s",
                #     email_id,
                #     fpath,
                # )
                continue

            out = {
                "email_id": email_id,
                "content_hash": content_hash,
                "origin_maildir": str(maildir_root),
                "source_filename": fpath.name,
                "maildir_path": str(fpath),
                "file_mtime": fpath.stat().st_mtime,
                "file_size": fpath.stat().st_size,
                "headers": dict(msg.items()),
                "mime_meta": mime_meta_from_msg(msg),
                "body": extract_body(msg),
                "processing": {
                    "ingested_at": datetime.utcnow().isoformat() + "Z",
                    "schema_version": "1",
                    "parsed_ok": True,
                },
            }

            shard = email_id[:2]
            out_path = output_dir / folder_label / shard / f"{email_id}.json"

            if not dry_run:
                atomic_write_json(out_path, out)
                append_registry_entry(registry_path, {
                    "email_id": email_id,
                    "content_hash": content_hash,
                    "last_completed_step": "step1_ingest",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
                registry_cache[email_id] = {"content_hash": content_hash}
                output_written += 1

            processed += 1

        except Exception as e:
            failed += 1
            logger.warning(
                "record_failed step=step1_ingest path=%s error=%s",
                fpath,
                e,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )

            if not dry_run:
                failed_path = output_dir / folder_label / "failed" / f"{fpath.name}.json"
                atomic_write_json(failed_path, {
                    "email_id": None,
                    "content_hash": None,
                    "maildir_path": str(fpath),
                    "processing": {
                        "ingested_at": datetime.utcnow().isoformat() + "Z",
                        "parsed_ok": False,
                        "parse_error": str(e),
                    },
                })

        if total_input % PROGRESS_EVERY_N == 0:
            logger.info(
                "step_progress step=step1_ingest seen=%d/%d processed=%d skipped=%d failed=%d skipped_breakdown=%s",
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
        "step_summary step=step1_ingest total_input=%d processed=%d skipped_total=%d failed=%d output_written=%d skipped_breakdown=%s invariant_ok=%s elapsed_s=%.2f",
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
            "step_invariant_violation step=step1_ingest lhs_total_input=%d rhs_sum=%d",
            total_input,
            processed + skipped_total + failed,
        )

    logger.info(
        "logging_hints step=step1_ingest "
        "Disable DEBUG to remove per-record diagnostics; "
        "increase PROGRESS_EVERY_N to reduce INFO volume; "
        "stack traces only appear when DEBUG is enabled."
    )


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maildir-roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-prefix")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    registry_path = Path(args.state_dir) / "processing_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    registry_cache: Dict[str, Dict[str, Any]] = {}
    if registry_path.exists():
        with registry_path.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if "email_id" in r:
                        registry_cache[r["email_id"]] = r
                except Exception:
                    continue

    for root in map(Path, args.maildir_roots):
        process_maildir(
            root,
            output_dir,
            registry_path,
            folder_label=root.name,
            registry_cache=registry_cache,
            limit=args.limit,
            only_prefix=args.only_prefix,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    while True:
        main()
        logger.warning("Ingest idle: sleeping for 5 minutes")
        time.sleep(300)
