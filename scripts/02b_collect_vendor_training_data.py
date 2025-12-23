#!/usr/bin/env python3
"""
02b_collect_vendor_training_data.py

FINAL – CORRECTED FOR PIPELINE STRUCTURE

- Reads email text from data/emails_prefiltered/
- Reads labels from data/state/step2b_vendor_scoring.jsonl
- Produces ONE human-editable training file

Output:
- data/vendor_training_review.jsonl

Human rule:
- ONLY edit `final_label`
"""

import json
import re
from pathlib import Path
from typing import Dict, Any


EMAIL_DIR = Path("data/emails_prefiltered")
STATE_FILE = Path("data/state/step2b_vendor_scoring.jsonl")
OUTPUT_FILE = Path("data/vendor_training_review.jsonl")


# -------------------------
# Helpers
# -------------------------

def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def build_text(email: Dict[str, Any]) -> str:
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


# -------------------------
# Main
# -------------------------

def main():
    # 1️⃣ Load Step2b labels
    label_by_email_id: dict[str, str] = {}

    with STATE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue

            eid = rec.get("email_id")
            label = rec.get("predicted_label")
            if eid and label in ("vendor", "non_vendor"):
                label_by_email_id[eid] = label

    print(f"Loaded labels for {len(label_by_email_id)} emails")

    # 2️⃣ Build training file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for path in EMAIL_DIR.rglob("*.json"):
            try:
                email = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                skipped += 1
                continue

            eid = email.get("email_id")
            if not eid or eid not in label_by_email_id:
                skipped += 1
                continue

            text = build_text(email)
            if not text.strip():
                skipped += 1
                continue

            predicted = label_by_email_id[eid]

            record = {
                "email_id": eid,
                "text": text,
                "predicted_label": predicted,
                # 👇 ONLY FIELD HUMANS MAY EDIT
                "final_label": predicted,
            }

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
