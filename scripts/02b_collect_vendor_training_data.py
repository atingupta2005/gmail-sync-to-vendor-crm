#!/usr/bin/env python3
"""
02b_collect_vendor_training_data.py (FINAL - Python 3.8 compatible)

What it does
- Reads email JSONs from: data/emails_prefiltered/
- Reads model labels from: data/state_excl/step2b_vendor_scoring.jsonl
- Applies external allow/deny lists with fuzzy matching (ALLOW wins)
- ALSO applies a strict positive-keywords file (invoice/PO/SOW etc.) to force vendor
- Writes ONE human-editable training file: data/vendor_training_review.jsonl

Human rule:
- ONLY edit `final_label`

Rules / Precedence
1) ALLOW by domain  -> label = vendor
2) ALLOW by fuzzy vendor-name keyword -> label = vendor
3) ALLOW by strict positive keywords (substring OR fuzzy) -> label = vendor
4) DENY by domain   -> label = non_vendor
5) DENY by fuzzy deny-name keyword -> label = non_vendor
6) Else fallback to model predicted_label

Notes
- Domain matching supports subdomains (sender endswith ".domain.com")
- For best fuzzy quality & speed install RapidFuzz: pip install rapidfuzz
"""

from __future__ import annotations  # <-- REQUIRED for Python 3.8 with list[str], dict[str,str]

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

EMAIL_DIR = Path("data/emails_prefiltered")
STATE_FILE = Path("data/state/step2b_vendor_scoring.jsonl")
OUTPUT_FILE = Path("data/vendor_training_review.jsonl")

DEFAULT_ALLOW_NAMES = Path("data/lists/positive_vendor_names_clean.txt")
DEFAULT_ALLOW_KEYWORDS = Path("data/lists/positive_keywords_clean.txt")
DEFAULT_ALLOW_DOMAINS = Path("data/lists/positive_vendor_domains_clean.txt")

DEFAULT_DENY_DOMAINS = Path("data/lists/deny_domains_clean.txt")
DEFAULT_DENY_NAMES = Path("data/lists/deny_names_clean_final.txt")


# -------------------------
# Optional fast fuzzy match
# -------------------------
try:
    from rapidfuzz import fuzz, process  # type: ignore
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False
    from difflib import SequenceMatcher



# -------------------------
# Helpers
# -------------------------

# --- Thread stripping (Gmail) ---
_FORWARD_MARKER_RE = re.compile(r"(?im)^[-]{5,}\s*forwarded message\s*[-]{5,}\s*$")
_ON_WROTE_RE = re.compile(r"(?im)^\s*on .+wrote:\s*$")
_ORIGINAL_MSG_RE = re.compile(r"(?im)^\s*-{2,}\s*original message\s*-{2,}\s*$")
_SIG_RE = re.compile(r"(?m)^\s*--\s*$")

def top_message_only(text: str, max_chars: int = 20000) -> str:
    """
    Keep only the newest/top message and remove quoted thread + signature.
    Designed for Gmail-style threads, with safe fallbacks.
    """
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = t[:max_chars]

    # 1) Find earliest thread boundary marker
    cut_positions = []

    m = _FORWARD_MARKER_RE.search(t)
    if m:
        cut_positions.append(m.start())

    m = _ON_WROTE_RE.search(t)
    if m:
        cut_positions.append(m.start())

    m = _ORIGINAL_MSG_RE.search(t)
    if m:
        cut_positions.append(m.start())

    # 2) Detect Outlook-ish header block inside forward/reply (From/Date/To/Subject/Cc)
    # Cut at the first "From:" line if the next few lines contain To: and Subject:
    lines = t.split("\n")
    scan_limit = min(len(lines), 200)  # only scan top region
    for i in range(scan_limit):
        if lines[i].strip().lower().startswith("from:"):
            window = "\n".join(lines[i:i+12]).lower()
            if "to:" in window and "subject:" in window:
                # find absolute char position of this line start
                pos = sum(len(lines[j]) + 1 for j in range(i))
                cut_positions.append(pos)
                break

    if cut_positions:
        t = t[:min(cut_positions)].strip()

    # 3) Strip signature if delimiter exists near the bottom
    sig = _SIG_RE.search(t)
    if sig:
        # only cut if signature marker is in last 40% to avoid killing content
        if sig.start() > int(len(t) * 0.6):
            t = t[:sig.start()].strip()

    # 4) Final cleanup
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t



def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def build_text(email: Dict[str, Any]) -> str:
    headers = email.get("headers", {}) or {}

    # Support both key styles (some pipelines use Subject/From, some subject/from)
    subject = str(headers.get("subject", "") or headers.get("Subject", "")).strip()
    from_addr = str(headers.get("from", "") or headers.get("From", "")).strip()

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


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[,./()\\-]+", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def normalize_domain(d: str) -> str:
    d = (d or "").strip().lower()
    d = re.sub(r"^[a-z]+://", "", d)
    d = d.split("/")[0]
    if d.startswith("www."):
        d = d[4:]
    return d.strip(".")


def load_list(path: Path) -> list[str]:
    if not path or not path.exists():
        print(f"WARNING: list file not found: {path}")
        return []
    items: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        items.append(t)
    return items


EMAIL_RE = re.compile(r"([a-z0-9._%+\\-]+)@([a-z0-9.\\-]+\\.[a-z]{2,})", re.I)

def extract_sender_domain(email_obj: Dict[str, Any]) -> Optional[str]:
    headers = email_obj.get("headers", {}) or {}

    # Try: From, Return-Path, Reply-To (Gmail often uses capitalized keys)
    candidates = [
        str(headers.get("from") or headers.get("From") or ""),
        str(headers.get("return-path") or headers.get("Return-Path") or ""),
        str(headers.get("reply-to") or headers.get("Reply-To") or ""),
    ]

    for c in candidates:
        m = EMAIL_RE.search(c)
        if m:
            return normalize_domain(m.group(2))
    return None



def domain_matches(sender_domain: str, listed_domain: str) -> bool:
    """Exact or subdomain match."""
    if not sender_domain or not listed_domain:
        return False
    sender_domain = normalize_domain(sender_domain)
    listed_domain = normalize_domain(listed_domain)
    return sender_domain == listed_domain or sender_domain.endswith("." + listed_domain)


def fuzzy_score(keyword: str, text: str) -> float:
    """
    Score in [0, 100].
    We use partial ratio because keywords are short and email text is long.
    """
    if not keyword or not text:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return float(fuzz.partial_ratio(keyword, text))
    return 100.0 * SequenceMatcher(None, keyword, text).ratio()


def best_fuzzy_match(text_norm: str, keywords_norm: list[str]) -> Tuple[Optional[str], float]:
    if not text_norm or not keywords_norm:
        return None, 0.0

    if _HAS_RAPIDFUZZ:
        # much faster than looping in Python
        res = process.extractOne(text_norm, keywords_norm, scorer=fuzz.partial_ratio)
        if not res:
            return None, 0.0
        kw, score, _ = res
        return kw, float(score)

    # difflib fallback
    best_kw: Optional[str] = None
    best_score: float = 0.0
    for kw in keywords_norm:
        sc = fuzzy_score(kw, text_norm)
        if sc > best_score:
            best_kw, best_score = kw, sc
    return best_kw, best_score



def strict_keyword_hit(text_norm: str, keywords_norm: list[str]) -> Optional[str]:
    """Strict keyword match: substring (fast, deterministic)."""
    for kw in keywords_norm:
        if kw and kw in text_norm:
            return kw
    return None


def apply_rules(
    email_obj: Dict[str, Any],
    text: str,
    predicted_label: str,
    allow_domains: list[str],
    deny_domains: list[str],
    allow_vendor_names_norm: list[str],
    deny_keywords_norm: list[str],
    allow_threshold: float,
    deny_threshold: float
) -> Tuple[str, Optional[dict]]:

    sender_dom = extract_sender_domain(email_obj)

    # Use a short candidate for fuzzy (reduces random matches in footers)
    headers = email_obj.get("headers", {}) or {}
    subject = str(headers.get("subject", "") or headers.get("Subject", "")).strip()
    from_addr = str(headers.get("from", "") or headers.get("From", "")).strip()

    body_full = (email_obj.get("body", {}) or {}).get("raw_text", "") or ""
    body_top = top_message_only(body_full)  # <-- strips old thread/forwards/signature

    candidate = f"{subject}\n{from_addr}\n{body_top[:500]}"
    text_norm = normalize_text(candidate)

    # 1) DENY by domain (PRIORITY)
    if sender_dom:
        for d in deny_domains:
            if domain_matches(sender_dom, d):
                return "non_vendor", {
                    "rule": "deny_domain",
                    "sender_domain": sender_dom,
                    "matched_domain": normalize_domain(d),
                }

    # 2) DENY by fuzzy deny-name (PRIORITY)
    if deny_keywords_norm:
        kw, sc = best_fuzzy_match(text_norm, deny_keywords_norm)
        if kw and sc >= deny_threshold:
            return "non_vendor", {
                "rule": "deny_fuzzy_keyword",
                "matched_keyword": kw,
                "score": sc,
            }

    # 3) ALLOW by domain
    if sender_dom:
        for d in allow_domains:
            if domain_matches(sender_dom, d):
                return "vendor", {
                    "rule": "allow_domain",
                    "sender_domain": sender_dom,
                    "matched_domain": normalize_domain(d),
                }

    # 4) ALLOW by fuzzy vendor-name
    if allow_vendor_names_norm:
        kw, sc = best_fuzzy_match(text_norm, allow_vendor_names_norm)
        if kw and sc >= allow_threshold:
            return "vendor", {
                "rule": "allow_fuzzy_vendor_name",
                "matched_keyword": kw,
                "score": sc,
            }

    # 5) fallback
    return predicted_label, None

# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email-dir", type=Path, default=EMAIL_DIR)
    ap.add_argument("--state-file", type=Path, default=STATE_FILE)
    ap.add_argument("--output-file", type=Path, default=OUTPUT_FILE)

    ap.add_argument("--allow-names", type=Path, default=DEFAULT_ALLOW_NAMES)
    ap.add_argument("--allow-domains", type=Path, default=DEFAULT_ALLOW_DOMAINS)

    ap.add_argument("--deny-domains", type=Path, default=DEFAULT_DENY_DOMAINS)
    ap.add_argument("--deny-names", type=Path, default=DEFAULT_DENY_NAMES)

    ap.add_argument("--allow-threshold", type=float, default=88.0)
    ap.add_argument("--deny-threshold", type=float, default=90.0)

    args = ap.parse_args()

    allow_domains = sorted({normalize_domain(x) for x in load_list(args.allow_domains) if x.strip()})
    deny_domains = sorted({normalize_domain(x) for x in load_list(args.deny_domains) if x.strip()})

    allow_vendor_names = sorted({normalize_text(x) for x in load_list(args.allow_names) if x.strip()})
    deny_names = sorted({normalize_text(x) for x in load_list(args.deny_names) if x.strip()})

    # Load Step2b labels
    label_by_email_id: dict[str, str] = {}

    with args.state_file.open("r", encoding="utf-8") as f:
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
    print(f"DENY  names={len(deny_names)}  DENY  domains={len(deny_domains)}")
    if not _HAS_RAPIDFUZZ:
        print("WARNING: rapidfuzz not installed; using difflib fallback.")

    # Build training file
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with args.output_file.open("w", encoding="utf-8") as out:
        for path in args.email_dir.rglob("*.json"):
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

            predicted2, rule_meta = apply_rules(
                email_obj=email,
                text=text,
                predicted_label=predicted,
                allow_domains=allow_domains,
                deny_domains=deny_domains,
                allow_vendor_names_norm=allow_vendor_names,
                deny_keywords_norm=deny_names,
                allow_threshold=args.allow_threshold,
                deny_threshold=args.deny_threshold,
            )



            record = {
                "email_id": eid,
                "text": text,
                "predicted_label": predicted2,
                "rule_override": rule_meta,
                "final_label": predicted2,  # ONLY FIELD HUMANS MAY EDIT
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written % 500 == 0:
                print(f"[collect] {written} samples written...")

    print("========================================")
    print("Vendor training data collection COMPLETE")
    print(f"Written : {written}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {args.output_file.resolve()}")
    print("========================================")


if __name__ == "__main__":
    main()
