#!/usr/bin/env python3
"""
Read-only keyword discovery on prefiltered emails.
Used ONLY to suggest candidate keywords for Step 2A refinement.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import List

INPUT_DIR = Path("data/emails_prefiltered")
MAX_FILES = 0          # 0 = all
MAX_BODY_CHARS = 400

STOPWORDS = {
    "the","and","for","with","this","that","from","your","have","will",
    "please","regards","thanks","thank","dear","hi","hello","sir",
    "sent","message","mail","email","com","www","http","https",
    "attached","attachment","file","find","fwd","re","fw",
    "as","is","are","was","were","be","been","being",
    "we","you","they","them","our","us","it","on","at","to","of","in"
}

GENERIC_NOISE = {
    "gmail","yahoo","outlook","hotmail","ebay","linkedin","profile",
    "resume","cv","signature","unsubscribe"
}


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [
        w for w in words
        if w not in STOPWORDS
        and w not in GENERIC_NOISE
        and not w.isdigit()
    ]


def iter_prefiltered_files(base: Path):
    count = 0
    for shard in base.iterdir():
        if not shard.is_dir():
            continue
        for f in shard.iterdir():
            if f.is_file() and f.suffix == ".json":
                yield f
                count += 1
                if MAX_FILES and count >= MAX_FILES:
                    return


def main():
    subject_unigrams = Counter()
    body_unigrams = Counter()
    body_bigrams = Counter()
    attachment_terms = Counter()

    total = 0

    for path in iter_prefiltered_files(INPUT_DIR):
        try:
            with path.open("r", encoding="utf-8") as fh:
                email = json.load(fh)
        except Exception:
            continue

        total += 1

        headers = email.get("headers", {})
        subject = headers.get("Subject") or ""
        body = (email.get("body", {}).get("raw_text") or "")[:MAX_BODY_CHARS]

        subj_tokens = tokenize(subject)
        body_tokens = tokenize(body)

        subject_unigrams.update(subj_tokens)
        body_unigrams.update(body_tokens)

        for i in range(len(body_tokens) - 1):
            bg = f"{body_tokens[i]} {body_tokens[i+1]}"
            body_bigrams[bg] += 1

        mime = email.get("mime_meta", {})
        for att in mime.get("attachments", []):
            name = att.get("filename") or ""
            attachment_terms.update(tokenize(name))

    print("\n=== Prefiltered Corpus Summary ===")
    print(f"Emails analyzed: {total}")

    print("\n=== Top Subject Keywords (Candidates) ===")
    for w, c in subject_unigrams.most_common(30):
        print(f"{w:25} {c}")

    print("\n=== Top Body Keywords (Candidates) ===")
    for w, c in body_unigrams.most_common(30):
        print(f"{w:25} {c}")

    print("\n=== Top Body Bigrams (Intent Signals) ===")
    for w, c in body_bigrams.most_common(25):
        print(f"{w:35} {c}")

    print("\n=== Attachment Filename Keywords ===")
    for w, c in attachment_terms.most_common(20):
        print(f"{w:25} {c}")


if __name__ == "__main__":
    main()
