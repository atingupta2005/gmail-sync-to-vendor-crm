#!/usr/bin/env python3
"""
02b_collect_vendor_training_data.py

FINAL – DO NOT MODIFY

Purpose:
- Collects vendor classification training data
- Produces ONE human-editable + machine-usable JSONL file
- Uses existing Step2B outputs as source of truth

Output:
- data/vendor_training_review.jsonl

Review rule:
- Humans may ONLY edit `final_label`
- Do NOT change any other field
"""

import json
import re
from pathlib import Path
from typing import Dict, Any


# =============================
# CONFIG (DO NOT CHANGE)
# =============================

INPUT_DIR = Path("data/emails_prefiltered")
OUTPUT_FILE = Path("data/vendor_training_review.jsonl")


# =============================
# TEXT HELPERS (STABLE)
# =============================

def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def build_model_input(email: Dict[str, Any]) -> str:
    headers = email.get("headers", {}) or {}
    subject = str(headers.get("subject", "")).strip()
    from_addr = str(headers.get("from", "")).strip()

    body = email.get("body", {}) or {}
    text = body.get("raw_text", "") or ""
    if not text and body.get("raw_html"):
        text = strip_html(body["raw_html"])

    parts = [
        "[SUBJECT]\n" + subject,
        "[FROM]\n" + from_addr,
        "[BODY]\n" + text,
    ]

    return "\n\n".join(p for p in parts if p.strip())


# =============================
# MAIN (DO NOT CHANGE)
# =============================

def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for path in INPUT_DIR.rglob("*.json"):
            try:
                email = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                skipped += 1
                continue

            bert = email.get("bert")
            if not bert:
                skipped += 1
                continue

            predicted = bert.get("label")
            if predicted not in ("vendor", "non_vendor"):
                skipped += 1
                continue

            record = {
                "email_id": email.get("email_id"),
                "text": build_model_input(email),
                "predicted_label": predicted,
                # 👇 THE ONLY FIELD HUMANS MAY EDIT
                "final_label": predicted,
            }

            if not record["text"].strip():
                skipped += 1
                continue

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written % 500 == 0:
                print(f"[collect] {written} samples written...")

    print("========================================")
    print("Vendor training data collection COMPLETE")
    print(f"Written : {written}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {OUTPUT_FILE.resolve()}")
    print("========================================")


if __name__ == "__main__":
    main()
