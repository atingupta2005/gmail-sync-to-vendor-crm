#!/usr/bin/env python3
"""
Read-only corpus analysis for Step 2A tuning.
Scans raw email JSON files and prints summary statistics.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List

INPUT_DIR = Path("data/emails_raw_json")
MAX_FILES = 0          # 0 = all files
MAX_BODY_CHARS = 500   # limit body text sampled per email

STOPWORDS = {
    "the", "and", "to", "of", "in", "for", "on", "with", "at",
    "is", "are", "this", "that", "it", "as", "be", "by", "or",
    "from", "we", "you", "your", "our", "please", "regards",
    "thanks", "thank", "hi", "hello", "dear"
}


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def iter_email_files(base: Path):
    count = 0
    for mailbox in base.iterdir():
        if not mailbox.is_dir():
            continue
        for shard in mailbox.iterdir():
            if not shard.is_dir():
                continue
            for f in shard.iterdir():
                if f.is_file() and f.suffix == ".json":
                    yield f
                    count += 1
                    if MAX_FILES and count >= MAX_FILES:
                        return


def main():
    subject_words = Counter()
    body_words = Counter()
    sender_domains = Counter()

    total = 0
    with_attachments = 0
    attachment_only = 0
    short_body = 0

    vague_phrases = Counter()

    for path in iter_email_files(INPUT_DIR):
        try:
            with path.open("r", encoding="utf-8") as fh:
                email = json.load(fh)
        except Exception:
            continue

        total += 1

        headers = email.get("headers", {})
        subject = headers.get("Subject") or ""
        body = (email.get("body", {}).get("raw_text") or "")[:MAX_BODY_CHARS]

        subject_words.update(tokenize(subject))
        body_words.update(tokenize(body))

        from_hdr = headers.get("From") or ""
        m = re.search(r"@([a-zA-Z0-9.-]+)", from_hdr)
        if m:
            sender_domains[m.group(1).lower()] += 1

        has_attach = email.get("mime_meta", {}).get("has_attachments", False)
        if has_attach:
            with_attachments += 1

        if has_attach and len(body.strip()) < 40:
            attachment_only += 1

        if len(body.strip()) < 40:
            short_body += 1

        lower_body = body.lower()
        for phrase in [
            "please find attached",
            "as discussed",
            "attached herewith",
            "sharing the document",
            "find attached"
        ]:
            if phrase in lower_body:
                vague_phrases[phrase] += 1

    print("\n=== Corpus Summary ===")
    print(f"Total emails scanned: {total}")
    print(f"Emails with attachments: {with_attachments} ({with_attachments/total:.1%})")
    print(f"Attachment-only emails: {attachment_only} ({attachment_only/total:.1%})")
    print(f"Very short body emails (<40 chars): {short_body} ({short_body/total:.1%})")

    print("\n=== Top Subject Keywords ===")
    for w, c in subject_words.most_common(30):
        print(f"{w:20} {c}")

    print("\n=== Top Body Keywords ===")
    for w, c in body_words.most_common(30):
        print(f"{w:20} {c}")

    print("\n=== Top Sender Domains ===")
    for d, c in sender_domains.most_common(20):
        print(f"{d:30} {c}")

    print("\n=== Vague Attachment Phrases ===")
    for p, c in vague_phrases.most_common():
        print(f"{p:30} {c}")


if __name__ == "__main__":
    main()
