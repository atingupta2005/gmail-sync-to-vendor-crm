#!/usr/bin/env python3
"""
03_review_quick.py

Fast, human-friendly review loop.

Input:
- data/vendor_training_review.jsonl (output of 02b script)

Outputs:
- data/state/review_quick_log.jsonl  (your decisions; supports resume)
- Appends manual keywords to:
  - data/lists/positive_keywords_clean.txt
  - data/lists/deny_names_clean.txt

Controls (single key):
- Enter : accept predicted label (fastest)
- v     : mark as vendor
- n     : mark as non_vendor
- s     : skip (no decision)
- a     : add POSITIVE keyword (you type a phrase; must be >5 chars)
- d     : add DENY keyword (you type a phrase; must be >5 chars)
- q     : quit

Notes:
- This script does NOT try to be smart about keyword mining; it’s optimized for speed.
  If you want suggestions, we can add them later, but this is the easiest input flow.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterable, Set, Tuple


SUBJECT_RE = re.compile(r"\[SUBJECT\]\s*(.*?)(?:\n\n|\Z)", re.S | re.I)
FROM_RE = re.compile(r"\[FROM\]\s*(.*?)(?:\n\n|\Z)", re.S | re.I)
BODY_RE = re.compile(r"\[BODY\]\s*(.*)", re.S | re.I)
EMAIL_RE = re.compile(r"([a-z0-9._%+\-]+)@([a-z0-9.\-]+\.[a-z]{2,})", re.I)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def parse_fields(text_block: str) -> Tuple[str, str, str]:
    subj = ""
    frm = ""
    body = ""
    m = SUBJECT_RE.search(text_block or "")
    if m:
        subj = m.group(1).strip()
    m = FROM_RE.search(text_block or "")
    if m:
        frm = m.group(1).strip()
    m = BODY_RE.search(text_block or "")
    if m:
        body = m.group(1).strip()
    return subj, frm, body


def extract_from_email(from_field: str) -> str:
    m = EMAIL_RE.search(from_field or "")
    return m.group(0).lower() if m else ""


def normalize_line(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def append_unique(path: Path, line: str) -> bool:
    line = (line or "").strip()
    if len(line) <= 5:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if path.exists():
        existing = {normalize_line(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}

    norm = normalize_line(line)
    if not norm or norm in existing:
        return False

    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return True


def load_reviewed_ids(review_log: Path) -> Set[str]:
    seen: Set[str] = set()
    if not review_log.exists():
        return seen
    for rec in read_jsonl(review_log):
        eid = rec.get("email_id")
        if eid:
            seen.add(eid)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/vendor_training_review.jsonl"))
    ap.add_argument("--review-log", type=Path, default=Path("data/state/review_quick_log.jsonl"))
    ap.add_argument("--positive-keywords", type=Path, default=Path("data/lists/positive_keywords_clean.txt"))
    ap.add_argument("--deny-keywords", type=Path, default=Path("data/lists/deny_names_clean.txt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--body-chars", type=int, default=400)
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input file: {args.input}")

    reviewed = load_reviewed_ids(args.review_log)
    print(f"Resume: already reviewed {len(reviewed)} emails.\n")

    count = 0
    for rec in read_jsonl(args.input):
        eid = rec.get("email_id")
        if not eid or eid in reviewed:
            continue

        text_block = rec.get("text", "") or ""
        subj, frm, body = parse_fields(text_block)
        from_email = extract_from_email(frm)

        predicted = rec.get("predicted_label", "")
        rule = rec.get("rule_override")

        print("=" * 90)
        print(f"ID: {eid}")
        print(f"FROM: {from_email or '(none parsed)'}")
        print(f"SUBJ: {subj[:220]}")
        print(f"PRED: {predicted}   RULE: {json.dumps(rule, ensure_ascii=False) if rule else 'none'}")
        if args.body_chars > 0:
            snippet = (body or "")[: args.body_chars]
            print("-" * 90)
            print(snippet)
            if len(body or "") > args.body_chars:
                print("...")

        print("-" * 90)
        print("Enter=accept  v=vendor  n=non_vendor  a=add+  d=add-  s=skip  q=quit")
        cmd = input("> ").strip().lower()

        added_pos = None
        added_neg = None

        # Default: accept predicted
        if cmd == "":
            human = predicted if predicted in {"vendor", "non_vendor"} else "skip"
        elif cmd == "v":
            human = "vendor"
        elif cmd == "n":
            human = "non_vendor"
        elif cmd == "s":
            human = "skip"
        elif cmd == "q":
            print("Quit.")
            break
        elif cmd == "a":
            phrase = input("Add POSITIVE keyword (>5 chars): ").strip()
            if append_unique(args.positive_keywords, phrase):
                added_pos = phrase
                print("✅ added to positive keywords")
            else:
                print("⚠️ not added (too short or already exists)")
            human = predicted if predicted in {"vendor", "non_vendor"} else "skip"
        elif cmd == "d":
            phrase = input("Add DENY keyword (>5 chars): ").strip()
            if append_unique(args.deny_keywords, phrase):
                added_neg = phrase
                print("✅ added to deny keywords")
            else:
                print("⚠️ not added (too short or already exists)")
            human = predicted if predicted in {"vendor", "non_vendor"} else "skip"
        else:
            # If they typed something unexpected, just accept predicted to keep it easy
            human = predicted if predicted in {"vendor", "non_vendor"} else "skip"

        decision = {
            "email_id": eid,
            "predicted_label": predicted,
            "human_label": human,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "from_email": from_email,
            "subject": subj[:300],
            "added_positive_keyword": added_pos,
            "added_deny_keyword": added_neg,
        }

        args.review_log.parent.mkdir(parents=True, exist_ok=True)
        with args.review_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(decision, ensure_ascii=False) + "\n")

        reviewed.add(eid)
        count += 1

        if args.limit and count >= args.limit:
            print(f"Reached limit={args.limit}.")
            break

    print(f"\nDone. Reviewed this run: {count}. Total reviewed: {len(reviewed)}")


if __name__ == "__main__":
    main()
