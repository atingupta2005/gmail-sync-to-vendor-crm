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

from registry import append_registry_entry

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")


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
    logger.info("Found %d message files in %s", len(files), maildir_root)

    if limit:
        files = files[:limit]

    processed = 0

    for fpath in files:
        try:
            stat = fpath.stat()
            fast_key = f"{fpath.name}:{stat.st_mtime}:{stat.st_size}"

            prev = registry_cache.get(fast_key)
            if prev:
                continue

            raw_bytes = fpath.read_bytes()
            msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
            email_id, content_hash = compute_email_id_and_hash(raw_bytes, msg)

            prev = registry_cache.get(email_id)
            if prev and prev.get("content_hash") == content_hash:
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

            processed += 1

        except Exception as e:
            logger.warning("Failed parsing %s: %s", fpath, e)

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

    logger.info("Processed %d emails", processed)


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

    # load registry ONCE
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
        logger.warning("Ingest: sleeping for 5 minutes")
        time.sleep(300)
