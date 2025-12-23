#!/usr/bin/env python3
"""
02b_create_review_subset.py

Creates a smaller subset for human review from the full dataset.
"""

import json
from pathlib import Path

INPUT = Path("data/vendor_training_review.jsonl")
OUTPUT = Path("data/vendor_training_review_subset.jsonl")

MAX_ROWS = 5000  # 👈 REVIEW THIS MANY ONLY


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with INPUT.open("r", encoding="utf-8") as src, OUTPUT.open("w", encoding="utf-8") as dst:
        for line in src:
            if written >= MAX_ROWS:
                break

            dst.write(line)
            written += 1

    print(f"Subset created: {written} rows")
    print(f"File: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
