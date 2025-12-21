#!/usr/bin/env python3
"""
Ingest Maildir (cur/, new/) and write one JSON per email.
Supports --dry-run, --limit, --only-prefix for fast iteration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Dict, Any, Optional

from registry import append_registry_entry, get_registry_entry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def normalize_headers_for_hash(headers: Dict[str, str]) -> str:
    parts = []
    for k in sorted(headers.keys()):
        parts.append(f"{k.lower()}:{headers[k].strip()}")
    return "\n".join(parts)


def extract_body(msg) -> Dict[str, Optional[str]]:
    # Prefer text/plain, else fallback to html stripped
    text = None
    html = None
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain" and text is None:
                try:
                    text = part.get_content().strip()
                except Exception:
                    pass
            elif ctype == "text/html" and html is None:
                try:
                    html = part.get_content()
                except Exception:
                    pass
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            text = msg.get_content()
        elif ctype == "text/html":
            html = msg.get_content()

    if text:
        return {"raw_text": text, "raw_html": html}
    if html:
        # naive html -> text fallback
        plain = re.sub(r"<[^>]+>", "", html)
        return {"raw_text": plain, "raw_html": html}
    return {"raw_text": "", "raw_html": None}


def mime_meta_from_msg(msg) -> Dict[str, Any]:
    has_attachments = False
    attachments = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        content_type = part.get_content_type()
        if filename:
            has_attachments = True
            size = None
            payload = part.get_payload(decode=True)
            if payload is not None:
                size = len(payload)
            attachments.append({"filename": filename, "content_type": content_type, "size": size})
    return {"has_attachments": has_attachments, "attachments": attachments}


def compute_email_id_and_hash(raw_bytes: bytes, msg) -> (str, str):
    message_id = msg.get("Message-ID") or msg.get("Message-Id") or None
    if message_id:
        email_id = sha1_hex(message_id.encode("utf-8"))
    else:
        headers = {k: v for k, v in msg.items()}
        norm = normalize_headers_for_hash(headers).encode("utf-8") + b"\n" + raw_bytes
        email_id = sha1_hex(norm)
    content_hash = sha1_hex(raw_bytes)
    return email_id, content_hash


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def process_maildir(maildir_root: Path, output_dir: Path, registry_path: str, limit: int = 0, only_prefix: Optional[str] = None, dry_run: bool = False):
    files = []
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
    logger.info("Found %d message files", len(files))
    if limit and limit > 0:
        files = files[:limit]

    processed = 0
    for fpath in files:
        try:
            raw_bytes = fpath.read_bytes()
            msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
            email_id, content_hash = compute_email_id_and_hash(raw_bytes, msg)
            registry_entry = get_registry_entry(registry_path, email_id)
            if registry_entry and registry_entry.get("content_hash") == content_hash and registry_entry.get("last_completed_step") == "step1_ingest":
                logger.debug("Skipping unchanged %s", email_id)
                continue

            headers = {k: v for k, v in msg.items()}
            body = extract_body(msg)
            mime_meta = mime_meta_from_msg(msg)
            stat = fpath.stat()
            out = {
                "email_id": email_id,
                "content_hash": content_hash,
                "maildir_path": str(fpath),
                "file_mtime": stat.st_mtime,
                "file_size": stat.st_size,
                "headers": headers,
                "mime_meta": mime_meta,
                "body": body,
                "processing": {
                    "ingested_at": datetime.utcnow().isoformat() + "Z",
                    "schema_version": "1",
                    "parsed_ok": True,
                },
            }

            subdir = email_id[:2]
            out_path = output_dir / subdir / f"{email_id}.json"
            if dry_run:
                logger.info("[dry-run] would write %s", out_path)
            else:
                atomic_write_json(out_path, out)
                append_registry_entry(registry_path, {
                    "email_id": email_id,
                    "content_hash": content_hash,
                    "last_completed_step": "step1_ingest",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                logger.info("Wrote %s", out_path)

            processed += 1
        except Exception as e:
            logger.exception("Failed processing %s: %s", fpath, e)
            # Write minimal failed JSON if possible
            try:
                failed = {
                    "email_id": None,
                    "content_hash": None,
                    "maildir_path": str(fpath),
                    "processing": {"ingested_at": datetime.utcnow().isoformat() + "Z", "parsed_ok": False, "parse_error": str(e)},
                }
                if not dry_run:
                    failed_path = output_dir / "failed" / f"{fpath.name}.json"
                    atomic_write_json(failed_path, failed)
                    append_registry_entry(registry_path, {
                        "email_id": f"{fpath.name}-failed",
                        "content_hash": None,
                        "last_completed_step": "step1_ingest",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "error": str(e)
                    })
            except Exception:
                logger.exception("Failed to write failed record for %s", fpath)

    logger.info("Processed %d files", processed)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--maildir-root", required=True, help="Path to Maildir root (contains cur/ and new/)")
    parser.add_argument("--output-dir", required=True, help="Output base directory for emails_raw_json")
    parser.add_argument("--state-dir", required=True, help="State directory (processing registry JSONL)")
    parser.add_argument("--limit", type=int, default=0, help="Process only N files (0 = all)")
    parser.add_argument("--only-prefix", default=None, help="Process only files whose filename starts with this prefix")
    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs; show planned actions")
    args = parser.parse_args(argv)

    maildir_root = Path(args.maildir_root)
    output_dir = Path(args.output_dir)
    state_dir = Path(args.state_dir)
    registry_path = str(Path(state_dir) / "processing_registry.jsonl")

    process_maildir(maildir_root, output_dir, registry_path, limit=args.limit, only_prefix=args.only_prefix, dry_run=args.dry_run)


if __name__ == "__main__":
    main()


