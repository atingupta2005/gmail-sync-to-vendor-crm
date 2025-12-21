#!/usr/bin/env python3
"""
validate_vendor_candidates.py

Human Validation CLI for STEP 2B (Vendor Relevance Scoring)

Purpose:
- Present scored emails to a human (you) for validation
- Capture ground-truth labels in an append-only JSONL file
- Enable incremental training later (no manual dataset creation)

This script:
- Reads STEP 2B scoring log
- Joins with raw/prefiltered email JSON
- Skips already-validated emails
- Focuses on configurable probability bands (e.g. borderline cases)
- Appends human labels to data/state/human_vendor_labels.jsonl

No registry mutation. No ML. No automation beyond presentation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Set


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------
# JSON helpers
# ----------------------------

def load_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------
# Email loading
# ----------------------------

def shard_path(root: Path, email_id: str) -> Path:
    shard = email_id[:2].lower() if len(email_id) >= 2 else "00"
    return root / shard / f"{email_id}.json"


def load_email(email_id: str, *roots: Path) -> Optional[Dict]:
    for root in roots:
        p = shard_path(root, email_id)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
    return None


# ----------------------------
# Rendering
# ----------------------------

def render_email(email: Dict, prob: float) -> None:
    headers = email.get("headers", {}) or {}
    body = email.get("body", {}) or {}

    print("\n" + "=" * 80)
    print(f"Vendor probability: {prob:.3f}")
    print("-" * 80)

    print(f"From   : {headers.get('from', '')}")
    print(f"To     : {headers.get('to', '')}")
    print(f"Subject: {headers.get('subject', '')}")
    print(f"Date   : {headers.get('date', '')}")

    print("-" * 80)
    text = body.get("raw_text", "") or ""
    if not text and body.get("raw_html"):
        text = body.get("raw_html", "")

    text = text.strip()
    if len(text) > 2000:
        text = text[:2000] + "\n\n[TRUNCATED]"
    print(text)
    print("=" * 80)


# ----------------------------
# CLI logic
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Human validation CLI for STEP 2B vendor scoring.")
    ap.add_argument("--scoring-log", required=True, help="Path to step2b_vendor_scoring.jsonl")
    ap.add_argument("--raw-dir", required=True, help="Path to data/emails_raw_json/")
    ap.add_argument("--prefiltered-dir", required=True, help="Path to data/emails_prefiltered/")
    ap.add_argument("--labels-out", required=True, help="Path to human_vendor_labels.jsonl")
    ap.add_argument("--min-prob", type=float, default=0.4, help="Minimum vendor_probability to consider")
    ap.add_argument("--max-prob", type=float, default=0.8, help="Maximum vendor_probability to consider")
    ap.add_argument("--limit", type=int, default=0, help="Max number of emails to validate (0 = no limit)")
    args = ap.parse_args()

    scoring_log = Path(args.scoring_log).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    prefiltered_dir = Path(args.prefiltered_dir).resolve()
    labels_out = Path(args.labels_out).resolve()

    # Load already-labeled email_ids
    labeled: Set[str] = set()
    for rec in load_jsonl(labels_out):
        eid = rec.get("email_id")
        if eid:
            labeled.add(eid)

    count = 0

    for rec in load_jsonl(scoring_log):
        email_id = rec.get("email_id")
        prob = rec.get("vendor_probability")

        if not email_id or prob is None:
            continue
        if email_id in labeled:
            continue
        if not (args.min_prob <= prob <= args.max_prob):
            continue

        email = load_email(email_id, raw_dir, prefiltered_dir)
        if not email:
            continue

        render_email(email, prob)

        while True:
            resp = input("Label [v]endor / [n]on-vendor / [s]kip / [q]uit: ").strip().lower()
            if resp in ("v", "n", "s", "q"):
                break

        if resp == "q":
            print("Exiting.")
            return 0
        if resp == "s":
            continue

        label = "vendor" if resp == "v" else "non_vendor"
        conf = input("Confidence (0.0–1.0, default 0.9): ").strip()
        try:
            conf_f = float(conf) if conf else 0.9
        except Exception:
            conf_f = 0.9
        notes = input("Notes (optional): ").strip()

        append_jsonl(labels_out, {
            "email_id": email_id,
            "human_label": label,
            "confidence": conf_f,
            "validated_at": utc_now_iso(),
            "notes": notes,
        })

        labeled.add(email_id)
        count += 1

        if args.limit and count >= args.limit:
            break

    print(f"\nValidation complete. Labeled {count} emails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
