#!/usr/bin/env python3
"""
STEP 3 — Cleanup & Reduction (Contact-Preserving)

Removes conversation noise while preserving ALL vendor identity and contact information.
NO AI usage. Deterministic only.

IMPROVEMENTS INCLUDED (as requested):
1) Creates/updates a separate vendor contacts file for vendors you worked with in the past
   (focused on training requirements), extracting:
   - Email Subject
   - Sender email id (From / Reply-To)
   - Person name (heuristic from signature)
   - Vendor/company name (heuristic from signature)
   - Phone numbers
   - Websites/URLs
   - Evidence references (email_id, subject, date, source path)

2) Deterministic deduplication/merge:
   - If one email has only phone + from email and another email from same sender has another phone + website,
     the final output merges UNION of phones/emails/websites across all evidence.
   - Uses overlap matching via lookup maps (email/phone/domain) to merge contacts even when keys differ.

3) Keeps your script-first, incremental, idempotent behavior:
   - Uses registry JSONL to skip already processed emails with same content_hash.
   - Contacts export runs only for processed emails in the run (not skipped).

4) Adds --loop flag instead of always running forever.
"""

import time
import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from email.utils import getaddresses
from urllib.parse import urlparse

# -------------------------
# Logging setup
# -------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("cleanup_vendor_emails")

# Progress heartbeat frequency (safe to increase to reduce log volume)
PROGRESS_EVERY_N = 500


# -------------------------
# Utility helpers
# -------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def shard_path(base: Path, email_id: str) -> Path:
    return base / email_id[:2] / f"{email_id}.json"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def stable_dedup_list(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# -------------------------
# Contact detection (CRITICAL)
# -------------------------

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"""
    (?:
        \+?\d{1,3}[\s.-]?
    )?
    (?:\(?\d{2,4}\)?[\s.-]?)?
    \d{3,4}[\s.-]?\d{3,4}
""", re.VERBOSE)

URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)


def contains_contact_info(text: str) -> bool:
    return bool(
        EMAIL_RE.search(text or "")
        or PHONE_RE.search(text or "")
        or URL_RE.search(text or "")
    )


def extract_emails(text: str) -> List[str]:
    if not text:
        return []
    found = [m.group(0).lower() for m in EMAIL_RE.finditer(text)]
    return stable_dedup_list(found)


def extract_phones(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    seen = set()
    for m in PHONE_RE.finditer(text):
        raw = m.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        # require at least 8 digits to avoid dates
        if len(digits) < 8:
            continue
        norm = re.sub(r"\s+", " ", raw)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    found = []
    for m in URL_RE.finditer(text):
        url = m.group(0).strip().rstrip(").,;]>")
        if url.lower().startswith("www."):
            url = "https://" + url
        found.append(url)
    return stable_dedup_list(found)


def domain_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def digits_only(phone: str) -> str:
    return re.sub(r"\D", "", phone or "")


def parse_header_addresses(value: Any) -> List[str]:
    """
    Parses header fields like From/To/Cc/Reply-To that may be strings or lists.
    Returns lowercased email addresses (deduped).
    """
    if value is None:
        return []
    if isinstance(value, list):
        s = ", ".join(str(x) for x in value if x is not None)
    elif isinstance(value, str):
        s = value
    else:
        s = str(value)

    addrs = []
    for _, addr in getaddresses([s]):
        addr = (addr or "").strip().lower()
        if addr and EMAIL_RE.fullmatch(addr):
            addrs.append(addr)
    return stable_dedup_list(addrs)


JOB_SUBJECT_REJECT_PATTERNS = [
    r"^\s*job\s*\|",
    r"^\s*✉️\s*job\s*\|",
    r"\b(opening|openings)\b",
    r"\burgent\s+req\b",
    r"\bhiring\b",
    r"\brecruit(ment|er|ing)\b",
    r"\bfreelance\b",
    r"\bretainer(s)?\b",
    r"\bapply\s+now\b",
]
_JOB_RE = re.compile("|".join(f"(?:{p})" for p in JOB_SUBJECT_REJECT_PATTERNS), re.IGNORECASE)

# -------------------------
# Training requirement detection (deterministic)
# -------------------------

# =========================
# DROP-IN REPLACEMENT BLOCK
# (Replace the existing TRAINING_KEYWORDS + is_training_requirement() with this)
# =========================

TRAINING_INTENT_KEYWORDS = [
    "training", "corporate training", "trainer", "faculty", "instructor", "facilitator",
    "workshop", "bootcamp", "course", "program", "enablement", "session", "sessions",
    "batch", "cohort", "class"
]

TRAINING_REQUEST_KEYWORDS = [
    "requirement", "need", "looking for", "we are looking", "request", "rfp", "rfq",
    "proposal", "quotation", "quote", "commercials", "budget", "cost", "pricing", "rate",
    "availability", "schedule", "dates", "duration", "onsite", "on-site", "virtual", "remote"
]

# Technology / domain keywords (used to boost recall + tag contacts)
TECH_KEYWORDS = [
    # AI / GenAI
    "generative ai", "genai", "llm", "llms", "large language model", "large language models",
    "rag", "retrieval augmented generation", "vector database", "vector databases",
    "ai agent", "ai agents", "agents", "responsible ai", "mlops",
    "deep learning", "machine learning",

    # Data
    "data engineering", "lakehouse", "databricks", "spark", "kafka", "airflow", "dbt",
    "data mesh", "real-time analytics", "realtime analytics",

    # Cloud
    "multi-cloud", "multicloud", "cloud architecture",
    "azure", "aws", "gcp", "oci", "azure openai", "azure ml", "sagemaker",

    # DevOps
    "devops", "devsecops", "docker", "kubernetes", "terraform", "iac", "ci/cd",
    "cicd", "gitops", "platform engineering", "observability", "prometheus", "grafana",

    # Development & Data tools
    "python", "java", "scala", "javascript", "typescript", "node.js", "nodejs",
    "django", "flask", "react", "power bi", "tableau",
    "cosmos db", "mongodb", "postgresql", "neo4j",

    # Blockchain
    "blockchain", "hyperledger", "ethereum", "corda", "smart contract", "smart contracts"
]

DEFAULT_EXCLUDE_EMAILS = {
    "atingupta2005@gmail.com",
    "u13@atttrainings.com",
}

def detect_topics(text: str) -> List[str]:
    """
    Deterministically detect technology/domain topics mentioned in text.
    Returns matched keywords (deduped, stable order).
    """
    hay = (text or "").lower()
    hits: List[str] = []
    seen = set()
    for kw in TECH_KEYWORDS:
        if kw in hay and kw not in seen:
            seen.add(kw)
            hits.append(kw)
    return hits


def _contains_any(hay: str, keywords: List[str]) -> bool:
    return any(k in hay for k in keywords)

def is_inbox_email(email_obj: Dict[str, Any]) -> bool:
    """
    Best-effort Inbox detection using whatever metadata Step 2B preserved.
    Used ONLY for contacts export (cleanup still runs for all).
    """
    candidates: List[str] = []

    # common fields likely present in your pipeline
    for k in ["maildir_path", "source_maildir_path", "source_email_path", "source_path"]:
        v = email_obj.get(k)
        if isinstance(v, str) and v:
            candidates.append(v)

    meta = email_obj.get("meta", {})
    if isinstance(meta, dict):
        for k in ["maildir_path", "source_maildir_path", "folder", "mailbox"]:
            v = meta.get(k)
            if isinstance(v, str) and v:
                candidates.append(v)

    hdrs = email_obj.get("headers", {})
    if isinstance(hdrs, dict):
        v = hdrs.get("folder") or hdrs.get("mailbox")
        if isinstance(v, str) and v:
            candidates.append(v)

    hay = " | ".join(candidates).lower()

    # strict-ish inbox markers (avoid accidental matches)
    if "/inbox/" in hay:
        return True
    if "inbox/cur" in hay or "inbox/new" in hay:
        return True
    if hay.strip() == "inbox":
        return True
    if "folder=inbox" in hay or "mailbox=inbox" in hay:
        return True

    return False

MARKETING_EXCLUDE_KEYWORDS = [
    "save", "discount", "offer", "code:", "coupon", "register", "signup", "sign up",
    "unsubscribe", "webinar", "conference", "conf is back", "trending certification",
    "limited time", "sale", "promotion"
]


JOB_SUBJECT_RE = re.compile(r"^\s*(?:job\s*\||✉️\s*job\s*\|)", re.IGNORECASE)

def is_job_like_subject(subject: str) -> bool:
    return bool(JOB_SUBJECT_RE.search(subject or ""))



def is_training_requirement(subject: str, text: str) -> bool:
    """
    Deterministic "training requirement" gate:
      - reject job-board / hiring posts by subject
      - then: (training intent AND request intent) OR (training intent AND tech keyword)
    """
    subj = (subject or "").strip().lower()

    # Hard reject obvious job-board / hiring posts
    if _JOB_RE.search(subj):
        return False

    hay = f"{subject or ''}\n{text or ''}".lower()

    has_intent = _contains_any(hay, TRAINING_INTENT_KEYWORDS)
    has_request = _contains_any(hay, TRAINING_REQUEST_KEYWORDS)
    topics = detect_topics(hay)
    has_tech = len(topics) > 0

    return (has_intent and has_request) or (has_intent and has_tech)


# -------------------------
# Cleanup logic
# -------------------------

QUOTED_REPLY_PATTERNS = [
    re.compile(r"^>+", re.MULTILINE),
    re.compile(r"^On .* wrote:$", re.MULTILINE),
]

FORWARDED_MARKERS = [
    re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}", re.IGNORECASE),
    re.compile(r"^Begin forwarded message:", re.IGNORECASE),
]

SIGNATURE_DELIMITERS = [
    re.compile(r"^--\s*$", re.MULTILINE),

    # Common sign-offs (must be a standalone line)
    re.compile(r"^(?:best\s+regards|kind\s+regards|warm\s+regards)\s*,?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(?:regards|thanks|thank\s+you)\s*,?\s*$", re.MULTILINE | re.IGNORECASE),

    # Variants you hit: "Thanks and Regards" / "Thanks & Regards"
    re.compile(r"^thanks\s*(?:and|&)\s*regards\s*,?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^regards\s*(?:and|&)\s*thanks\s*,?\s*$", re.MULTILINE | re.IGNORECASE),
]


DISCLAIMER_KEYWORDS = [
    "confidential",
    "intended recipient",
    "privileged",
    "legal disclaimer",
]


def remove_quoted_and_forwarded(text: str) -> Tuple[str, bool]:
    original = text
    for pat in FORWARDED_MARKERS + QUOTED_REPLY_PATTERNS:
        split = pat.split(text, maxsplit=1)
        if len(split) > 1:
            logger.debug("Removing quoted/forwarded block")
            text = split[0]
    return (text or "").strip(), (text != original)


def split_signature(text: str) -> Tuple[str, str, bool]:
    """
    Extract signature WITHOUT deleting it.
    """
    for pat in SIGNATURE_DELIMITERS:
        parts = pat.split(text or "", maxsplit=1)
        if len(parts) > 1:
            logger.debug("Signature delimiter detected")
            body = parts[0].strip()
            signature = parts[1].strip()
            return body, signature, True
    return (text or "").strip(), "", False


def remove_disclaimer_safely(text: str) -> Tuple[str, bool]:
    """
    Remove disclaimer ONLY if it contains no contact info.
    """
    lower = (text or "").lower()
    for kw in DISCLAIMER_KEYWORDS:
        idx = lower.find(kw)
        if idx != -1:
            tail = (text or "")[idx:]
            if not contains_contact_info(tail):
                logger.debug("Removing disclaimer block without contact info")
                return (text or "")[:idx].strip(), True
            logger.debug("Disclaimer retained due to contact info")
    return (text or ""), False


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def truncate_body_only(body: str, max_chars: int = 2000) -> str:
    if len(body) <= max_chars:
        return body
    logger.debug("Truncating body text to max_chars=%d", max_chars)
    cut = body[:max_chars]
    last_period = cut.rfind(".")
    if last_period > 200:
        return cut[: last_period + 1]
    return cut


def cleanup_text(raw_text: str) -> Dict[str, Any]:
    meta = {
        "removed_quoted_blocks": False,
        "removed_disclaimer": False,
        "signature_extracted": False,
        "contact_preserved": True,
    }

    text = raw_text or ""

    text, changed = remove_quoted_and_forwarded(text)
    meta["removed_quoted_blocks"] = changed

    body, signature, changed = split_signature(text)
    meta["signature_extracted"] = changed

    body, changed = remove_disclaimer_safely(body)
    meta["removed_disclaimer"] = changed

    body = normalize_whitespace(body)
    signature = normalize_whitespace(signature)

    body = truncate_body_only(body)

    cleaned = body
    if signature:
        cleaned = f"{body}\n\n--- SIGNATURE ---\n{signature}"

    return {
        "cleaned_text": cleaned,
        "signature_text": signature,
        "meta": meta,
    }


# -------------------------
# Registry handling (unchanged)
# -------------------------

def load_registry(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    registry = {}
    if not registry_path.exists():
        return registry
    with registry_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "email_id" in rec:
                    registry[rec["email_id"]] = rec
            except Exception:
                continue
    return registry


def append_registry(registry_path: Path, record: Dict[str, Any]) -> None:
    ensure_parent(registry_path)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------------------
# Header helpers (NEW)
# -------------------------
def pick_headers(email_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort extraction of header fields from Step 2B candidate JSON.
    Handles case-insensitive raw header keys like 'Subject', 'From', 'Reply-To'.
    """
    src = None
    if isinstance(email_obj.get("headers"), dict):
        src = email_obj["headers"]
    elif isinstance(email_obj.get("header"), dict):
        src = email_obj["header"]
    elif isinstance(email_obj.get("meta", {}).get("headers"), dict):
        src = email_obj["meta"]["headers"]

    # Build case-insensitive map
    src_ci = {str(k).strip().lower(): v for k, v in (src or {}).items()}

    keymap = {
        "from": ["from"],
        "to": ["to"],
        "cc": ["cc"],
        "bcc": ["bcc"],
        "reply_to": ["reply-to", "reply_to", "replyto"],
        "subject": ["subject"],
        "date": ["date"],
        "message_id": ["message-id", "message_id", "messageid"],
    }

    out: Dict[str, Any] = {}
    for out_key, in_keys in keymap.items():
        for ik in in_keys:
            if ik in src_ci:
                out[out_key] = src_ci[ik]
                break

    # fallback if flattened at top-level (also case-insensitive)
    top_ci = {str(k).strip().lower(): k for k in email_obj.keys()}
    for out_key, in_keys in keymap.items():
        if out_key in out:
            continue
        for ik in in_keys:
            if ik in top_ci:
                out[out_key] = email_obj.get(top_ci[ik])
                break

    return out




def header_str(hdrs: Dict[str, Any], key: str) -> str:
    v = hdrs.get(key)
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x is not None)
    return str(v)


# -------------------------
# Vendor contact export (NEW, deterministic)
# -------------------------

def load_contacts_store(index_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, str]]]:
    """
    Loads contacts store:
      {
        "contacts": { contact_key: record, ... },
        "lookups":  { "email": {...}, "phone": {...}, "domain": {...} }
      }
    Backward compatible: if old dict format, treat as contacts only and rebuild lookups during run.
    """
    obj = safe_read_json(index_path)
    if isinstance(obj, dict) and "contacts" in obj and "lookups" in obj:
        contacts = obj.get("contacts") if isinstance(obj.get("contacts"), dict) else {}
        lookups = obj.get("lookups") if isinstance(obj.get("lookups"), dict) else {}
        lookups.setdefault("email", {})
        lookups.setdefault("phone", {})
        lookups.setdefault("domain", {})
        return contacts, lookups

    if isinstance(obj, dict):
        return obj, {"email": {}, "phone": {}, "domain": {}}

    return {}, {"email": {}, "phone": {}, "domain": {}}


def append_contacts_event(jsonl_path: Path, event: Dict[str, Any]) -> None:
    ensure_parent(jsonl_path)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def choose_contact_key(
    sender_emails: List[str],
    candidate_emails: List[str],
    phones: List[str],
    urls: List[str],
    email_id: str
) -> str:
    # Strong preference: sender identity from From/Reply-To
    if sender_emails:
        return f"email:{sender_emails[0]}"
    if candidate_emails:
        return f"email:{candidate_emails[0]}"
    if phones:
        d = digits_only(phones[0])
        if d:
            return f"phone:{d}"
    if urls:
        d = domain_from_url(urls[0])
        if d:
            return f"domain:{d}"
    return f"email_id:{email_id}"


def heuristic_name_vendor_from_signature(signature_text: str) -> Tuple[str, str]:
    """
    Conservative heuristics:
    - If signature has "Name | Title | Company"
    - Else take first non-empty line as person name if it looks like a name
    - Vendor/company name from a line with common suffix or uppercase-ish
    """
    sig = signature_text or ""
    lines = [ln.strip() for ln in sig.splitlines() if ln.strip()]
    if not lines:
        return "", ""

    # "Name | Title | Company"
    for ln in lines[:6]:
        if "|" in ln and len(ln) <= 120:
            parts = [p.strip() for p in ln.split("|") if p.strip()]
            if len(parts) >= 2:
                person = parts[0]
                company = parts[-1] if len(parts) >= 3 else ""
                return person[:80], company[:120]

    def looks_like_name(s: str) -> bool:
        if contains_contact_info(s):
            return False
        if len(s) > 80:
            return False
        toks = s.split()
        if not (2 <= len(toks) <= 4):
            return False
        alpha_ratio = sum(ch.isalpha() for ch in s) / max(1, len(s))
        return alpha_ratio > 0.7

    person_name = ""
    for ln in lines[:5]:
        if looks_like_name(ln):
            person_name = ln
            break

    company_suffix = re.compile(r"\b(inc|llc|ltd|gmbh|pvt|plc|corp|co)\b\.?", re.IGNORECASE)
    vendor_name = ""
    for ln in lines[:10]:
        if contains_contact_info(ln):
            continue
        if company_suffix.search(ln):
            vendor_name = ln
            break

    if not vendor_name:
        for ln in lines[:10]:
            if contains_contact_info(ln):
                continue
            letters = [c for c in ln if c.isalpha()]
            if letters and (sum(c.isupper() for c in letters) / len(letters) > 0.6) and len(ln) <= 80:
                vendor_name = ln
                break

    return person_name[:80], vendor_name[:120]


def build_contact_record(
    email_id: str,
    hdrs: Dict[str, Any],
    cleaned_text: str,
    signature_text: str,
    source_path: str,
    exclude_emails: List[str],
) -> Dict[str, Any]:
    """
    Build a deterministic vendor contact record.

    Key goals:
    - Prefer sender identity (From/Reply-To) for contact_key.
    - Prevent "thread participant" pollution by NOT blindly adding every email seen in body.
    - Default-exclude your own addresses even if caller forgets to pass --exclude-email.
    - Allow a tiny amount of body email inclusion only when it matches sender domain(s).
    """

    subject = header_str(hdrs, "subject")
    date = header_str(hdrs, "date")

    # Sender identity (preferred)
    from_emails = parse_header_addresses(hdrs.get("from"))
    reply_to_emails = parse_header_addresses(hdrs.get("reply_to"))
    sender_emails = stable_dedup_list(from_emails + reply_to_emails)

    # Extract from signature/body
    sig_emails = extract_emails(signature_text)
    body_emails = extract_emails(cleaned_text)

    phones = stable_dedup_list(extract_phones(signature_text) + extract_phones(cleaned_text))
    urls = stable_dedup_list(extract_urls(signature_text) + extract_urls(cleaned_text))

    # Exclusions (CLI + defaults)
    exclude_set = {e.lower() for e in (exclude_emails or [])}
    exclude_set |= {e.lower() for e in DEFAULT_EXCLUDE_EMAILS}

    # ---- Candidate emails: sender + signature only (safe) ----
    candidate_emails = stable_dedup_list(sender_emails + sig_emails)

    # Normalize domains helper
    def _dom(email: str) -> str:
        try:
            return email.split("@", 1)[1].strip().lower()
        except Exception:
            return ""

    # ---- OPTIONAL: only include body emails if they match allowed domain(s) ----
    sender_domains = { _dom(e) for e in sender_emails if "@" in e }
    sender_domains.discard("")

    # Heuristics from signature
    person_name, vendor_name = heuristic_name_vendor_from_signature(signature_text)

    # --- NEW: sanitize vendor_name (avoid "Mon-Fri...", URLs, addresses, disclaimers, etc.) ---
    def _looks_like_bad_vendor(v: str) -> bool:
        if not v:
            return True
        s = v.strip()
        if len(s) < 2:
            return True
        low = s.lower()

        # obvious non-vendor patterns
        if "http://" in low or "https://" in low or "www." in low:
            return True
        if "@" in low:
            return True
        # time/day / greetings / thanks / regards
        if any(x in low for x in ["mon-fri", "mon - fri", "thanks", "regards", "warm regards"]):
            return True
        # address-ish (pin code / city-state-zip formats)
        if any(x in low for x in ["new delhi", "mumbai", "pune", "bangalore", "hyderabad", "chennai"]):
            # not always bad, but usually in your “bad vendor” examples
            return True
        # lots of digits => probably phone/address/disclaimer
        digit_count = sum(ch.isdigit() for ch in s)
        if digit_count >= 4:
            return True
        # disclaimer-ish
        if any(x in low for x in ["disclaimer", "confidential", "privileged", "bse ltd"]):
            return True
        return False

    if _looks_like_bad_vendor(vendor_name):
        vendor_name = ""

    # --- NEW: vendor fallback from sender domain (only if still missing) ---
    def _vendor_from_sender_domain(sender_list: List[str]) -> str:
        # choose first non-free-email domain; turn "stripedata.net" -> "STRIPEDATA"
        free = {
            "gmail.com","googlemail.com","yahoo.com","ymail.com","outlook.com","hotmail.com",
            "live.com","rediffmail.com","icloud.com"
        }
        for e in sender_list:
            d = _dom(e)
            if not d:
                continue
            # base domain naive: last two labels
            parts = d.split(".")
            base = ".".join(parts[-2:]) if len(parts) > 2 else d
            if base in free:
                continue
            core = base.split(".", 1)[0]
            core = re.sub(r"[^a-z0-9]+", " ", core.lower()).strip()
            if core:
                return core.upper()
        return ""

    if not vendor_name:
        vendor_name = _vendor_from_sender_domain(sender_emails)

    # If we have a vendor name derived from sender domain, allow that domain too (still safe)
    allowed_domains = set(sender_domains)

    if vendor_name and sender_emails:
        # derive domain again from first sender email (vendor_name is not reliable for domain)
        first_sender_dom = _dom(sender_emails[0])
        if first_sender_dom:
            allowed_domains.add(first_sender_dom)

    if allowed_domains:
        for e in body_emails:
            d = _dom(e)
            if d and d in allowed_domains:
                candidate_emails.append(e)
        candidate_emails = stable_dedup_list(candidate_emails)

    # Apply exclusions last
    candidate_emails = [e for e in candidate_emails if e and e.lower() not in exclude_set]
    sender_emails = [e for e in sender_emails if e and e.lower() not in exclude_set]

    # Strong preference: key from sender identity
    contact_key = choose_contact_key(sender_emails, candidate_emails, phones, urls, email_id)

    combined_for_topics = f"{subject}\n{cleaned_text}\n{signature_text}"
    topics_detected = detect_topics(combined_for_topics)

    return {
        "contact_key": contact_key,
        "person_name": person_name,
        "vendor_name": vendor_name,
        "emails": candidate_emails,
        "phones": phones,
        "websites": urls,
        "last_subject": subject,
        "first_seen": None,   # filled by upsert
        "last_seen": None,    # filled by upsert
        "topics_detected": topics_detected,
        "evidence": [
            {
                "email_id": email_id,
                "subject": subject,
                "date": date,
                "source_email_path": source_path,
            }
        ],
    }



# --- fallback: vendor from sender domain (only if vendor_name is missing) ---
import re

GENERIC_DOMAINS = {
    "gmail.com","googlemail.com","yahoo.com","ymail.com","outlook.com","hotmail.com",
    "live.com","rediffmail.com","icloud.com"
}

def _base_domain(dom: str) -> str:
    dom = (dom or "").strip().lower()
    dom = re.sub(r"^www\.", "", dom)
    parts = dom.split(".")
    if len(parts) <= 2:
        return dom
    # naive last-2-label base; good enough for this pipeline
    return ".".join(parts[-2:])

def _vendor_from_email_domain(email: str) -> str:
    if not email or "@" not in email:
        return ""
    dom = email.split("@", 1)[1].lower()
    bd = _base_domain(dom)
    if bd in GENERIC_DOMAINS:
        return ""
    core = bd.split(".", 1)[0]
    core = re.sub(r"[^a-z0-9]+", " ", core).strip()
    if not core:
        return ""
    return core.upper()



def find_existing_keys(
    lookups: Dict[str, Dict[str, str]],
    emails: List[str],
    phones: List[str],
    urls: List[str],
) -> List[str]:
    hits: List[str] = []
    email_map = lookups.get("email", {})
    phone_map = lookups.get("phone", {})
    domain_map = lookups.get("domain", {})

    for e in emails:
        k = email_map.get((e or "").lower())
        if k and k not in hits:
            hits.append(k)

    for p in phones:
        d = digits_only(p)
        if len(d) >= 8:
            k = phone_map.get(d)
            if k and k not in hits:
                hits.append(k)

    for u in urls:
        d = domain_from_url(u)
        if d:
            k = domain_map.get(d)
            if k and k not in hits:
                hits.append(k)

    return hits


def rebuild_all_lookups(contacts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    lookups = {"email": {}, "phone": {}, "domain": {}}
    for ck, c in contacts.items():
        for e in c.get("emails", []) or []:
            lookups["email"][e.lower()] = ck
        for p in c.get("phones", []) or []:
            d = digits_only(p)
            if len(d) >= 8:
                lookups["phone"][d] = ck
        for u in c.get("websites", []) or []:
            d = domain_from_url(u)
            if d:
                lookups["domain"][d] = ck
    return lookups

def merge_contacts_into(
    contacts: Dict[str, Dict[str, Any]],
    target_key: str,
    source_key: str,
    max_evidence: int,
) -> None:
    if source_key == target_key:
        return
    src = contacts.get(source_key)
    tgt = contacts.get(target_key)
    if not src or not tgt:
        return

    tgt["emails"] = stable_dedup_list((tgt.get("emails", []) or []) + (src.get("emails", []) or []))
    tgt["phones"] = stable_dedup_list((tgt.get("phones", []) or []) + (src.get("phones", []) or []))
    tgt["websites"] = stable_dedup_list((tgt.get("websites", []) or []) + (src.get("websites", []) or []))
    tgt["topics_detected"] = stable_dedup_list(
        (tgt.get("topics_detected", []) or []) + (src.get("topics_detected", []) or [])
    )


    if not tgt.get("person_name") and src.get("person_name"):
        tgt["person_name"] = src["person_name"]
    if not tgt.get("vendor_name") and src.get("vendor_name"):
        tgt["vendor_name"] = src["vendor_name"]

    # timestamps
    if src.get("first_seen") and tgt.get("first_seen"):
        tgt["first_seen"] = min(tgt["first_seen"], src["first_seen"])
    elif src.get("first_seen") and not tgt.get("first_seen"):
        tgt["first_seen"] = src["first_seen"]

    if src.get("last_seen") and tgt.get("last_seen"):
        tgt["last_seen"] = max(tgt["last_seen"], src["last_seen"])
    elif src.get("last_seen") and not tgt.get("last_seen"):
        tgt["last_seen"] = src["last_seen"]

    # evidence append-only with cap (keep most recent N)
    ev = (tgt.get("evidence", []) or []) + (src.get("evidence", []) or [])
    if max_evidence > 0 and len(ev) > max_evidence:
        ev = ev[-max_evidence:]
    tgt["evidence"] = ev

    # keep last_subject if target missing
    if not tgt.get("last_subject") and src.get("last_subject"):
        tgt["last_subject"] = src["last_subject"]

    contacts[target_key] = tgt
    # remove source
    try:
        del contacts[source_key]
    except KeyError:
        pass


def update_lookups_for_contact(
    lookups: Dict[str, Dict[str, str]],
    contact_key: str,
    contact: Dict[str, Any],
) -> None:
    lookups.setdefault("email", {})
    lookups.setdefault("phone", {})
    lookups.setdefault("domain", {})

    for e in contact.get("emails", []) or []:
        lookups["email"][(e or "").lower()] = contact_key

    for p in contact.get("phones", []) or []:
        d = digits_only(p)
        if len(d) >= 8:
            lookups["phone"][d] = contact_key

    for u in contact.get("websites", []) or []:
        d = domain_from_url(u)
        if d:
            lookups["domain"][d] = contact_key


def upsert_contact(
    contacts: Dict[str, Dict[str, Any]],
    lookups: Dict[str, Dict[str, str]],
    record: Dict[str, Any],
    now_iso: str,
    max_evidence: int,
) -> Dict[str, Any]:
    emails = record.get("emails", []) or []
    phones = record.get("phones", []) or []
    urls = record.get("websites", []) or []

    matched_keys = find_existing_keys(lookups, emails, phones, urls)

    desired_key = record.get("contact_key") or ""
    canonical_key = None

    # prefer email-based key
    if isinstance(desired_key, str) and desired_key.startswith("email:"):
        canonical_key = desired_key
    elif matched_keys:
        canonical_key = matched_keys[0]
    else:
        canonical_key = desired_key if desired_key else f"email_id:{record['evidence'][0]['email_id']}"

    # Ensure canonical exists
    if canonical_key not in contacts:
        record["first_seen"] = now_iso
        record["last_seen"] = now_iso
        record["contact_key"] = canonical_key
        contacts[canonical_key] = record
    else:
        existing = contacts[canonical_key]
        existing["emails"] = stable_dedup_list((existing.get("emails", []) or []) + emails)
        existing["phones"] = stable_dedup_list((existing.get("phones", []) or []) + phones)
        existing["websites"] = stable_dedup_list((existing.get("websites", []) or []) + urls)
        existing["topics_detected"] = stable_dedup_list(
            (existing.get("topics_detected", []) or []) + (record.get("topics_detected", []) or [])
        )


        if not existing.get("person_name") and record.get("person_name"):
            existing["person_name"] = record["person_name"]
        if not existing.get("vendor_name") and record.get("vendor_name"):
            existing["vendor_name"] = record["vendor_name"]

        if record.get("last_subject"):
            existing["last_subject"] = record["last_subject"]

        if not existing.get("first_seen"):
            existing["first_seen"] = now_iso
        existing["last_seen"] = now_iso

        ev = (existing.get("evidence", []) or [])
        ev.extend(record.get("evidence", []) or [])
        if max_evidence > 0 and len(ev) > max_evidence:
            ev = ev[-max_evidence:]
        existing["evidence"] = ev

        existing["contact_key"] = canonical_key
        contacts[canonical_key] = existing

    # Merge any other matched contacts into canonical
    for k in matched_keys:
        if k != canonical_key and k in contacts:
            merge_contacts_into(contacts, canonical_key, k, max_evidence=max_evidence)

    # Update lookups for canonical
    update_lookups_for_contact(lookups, canonical_key, contacts[canonical_key])

    return contacts[canonical_key]


def write_contacts_snapshot(contacts: Dict[str, Dict[str, Any]], snapshot_path: Path) -> None:
    items = list(contacts.values())

    def sort_key(x: Dict[str, Any]) -> Tuple[str, str]:
        return (str(x.get("last_seen") or ""), str(x.get("contact_key") or ""))

    items.sort(key=sort_key, reverse=True)
    atomic_write_json(snapshot_path, {"generated_at": utc_now(), "count": len(items), "contacts": items})


# -------------------------
# Main
# -------------------------

def main() -> int:
    start_ts = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)

    # Contacts export controls
    ap.add_argument("--export-contacts", action="store_true", help="Enable vendor contacts export")
    ap.add_argument("--training-only", action="store_true", help="Only export contacts if email looks like training requirement")
    ap.add_argument("--contacts-max-evidence", type=int, default=25, help="Max evidence items stored per contact_key")
    ap.add_argument("--exclude-email", action="append", default=[], help="Email to exclude from extracted contacts (repeatable)")

    # Loop control
    ap.add_argument("--loop", action="store_true", help="Run forever with 5 min sleep (heartbeat)")
    args = ap.parse_args()

    candidates_dir = Path(args.candidates_dir)
    output_dir = Path(args.output_dir)
    state_dir = Path(args.state_dir)

    registry_path = state_dir / "processing_registry_step3_cleanup.jsonl"

    # contacts artifacts
    contacts_events_path = state_dir / "vendor_contacts.jsonl"
    contacts_store_path = state_dir / "vendor_contacts_index.json"
    contacts_snapshot_path = state_dir / "vendor_contacts_latest.json"

    registry = load_registry(registry_path)

    contacts: Dict[str, Dict[str, Any]] = {}
    lookups: Dict[str, Dict[str, str]] = {"email": {}, "phone": {}, "domain": {}}
    if args.export_contacts:
        contacts, lookups = load_contacts_store(contacts_store_path)

    processed = 0
    failed = 0
    skipped_total = 0
    skipped_already_processed = 0
    output_written = 0
    total_input = 0
    contacts_upserts = 0

    logger.info(
        "STEP 3 START: candidates_dir=%s output_dir=%s limit=%s export_contacts=%s training_only=%s",
        candidates_dir,
        output_dir,
        args.limit or "none",
        bool(args.export_contacts),
        bool(args.training_only),
    )

    for json_path in candidates_dir.rglob("*.json"):
        if args.limit and processed >= args.limit:
            logger.info("Processing limit reached (%d); stopping early", args.limit)
            break
        total_input += 1


        try:
            with json_path.open("r", encoding="utf-8") as f:
                email = json.load(f)

            email_id = email["email_id"]
            content_hash = email["content_hash"]

            reg = registry.get(email_id)
            if reg and reg.get("last_completed_step") == "step3_cleanup" and reg.get("content_hash") == content_hash:
                skipped_total += 1
                skipped_already_processed += 1
                logger.debug("Skipping %s: already processed with same content_hash", email_id)
                continue

            raw_text = email.get("body", {}).get("raw_text") or ""
            hdrs = pick_headers(email)

            cleanup = cleanup_text(raw_text)

            output = {
                "email_id": email_id,
                "content_hash": content_hash,
                "source_email_path": str(json_path),

                # carry forward headers (subject, from, etc.)
                "headers": hdrs,

                "cleaned_text": cleanup["cleaned_text"],
                "signature_text": cleanup["signature_text"],
                "cleanup_meta": {
                    "original_length": len(raw_text),
                    "cleaned_length": len(cleanup["cleaned_text"]),
                    **cleanup["meta"],
                },
                "processing": {
                    "cleaned_at": utc_now(),
                    "schema_version": 3,
                },
            }

            out_path = shard_path(output_dir, email_id)
            ensure_parent(out_path)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            output_written += 1

            # Contacts export (deterministic) — INBOX ONLY (hard-coded)
            if args.export_contacts and is_inbox_email(email):
                subject = header_str(hdrs, "subject")

                # hard drop job-like subjects from contacts export
                if is_job_like_subject(subject):
                    continue

                gate_text = f"{cleanup['cleaned_text']}\n{cleanup['signature_text']}"
                gate_ok = True
                if args.training_only:
                    gate_ok = is_training_requirement(subject, gate_text)

                if gate_ok:
                    now_iso = utc_now()
                    record = build_contact_record(
                        email_id=email_id,
                        hdrs=hdrs,
                        cleaned_text=cleanup["cleaned_text"],
                        signature_text=cleanup["signature_text"],
                        source_path=str(json_path),
                        exclude_emails=args.exclude_email,
                    )

                    merged = upsert_contact(
                        contacts=contacts,
                        lookups=lookups,
                        record=record,
                        now_iso=now_iso,
                        max_evidence=args.contacts_max_evidence,
                    )

                    append_contacts_event(
                        contacts_events_path,
                        {
                            "ts": now_iso,
                            "event": "upsert_contact",
                            "email_id": email_id,
                            "contact_key": merged.get("contact_key"),
                            "subject": subject,
                        },
                    )
                    contacts_upserts += 1


            append_registry(
                registry_path,
                {
                    "email_id": email_id,
                    "content_hash": content_hash,
                    "last_completed_step": "step3_cleanup",
                    "timestamp": utc_now(),
                },
            )

            processed += 1

        except Exception as e:
            failed += 1
            logger.warning("Recoverable failure for %s: %s", json_path, e)
            append_registry(
                registry_path,
                {
                    "email_id": email_id if "email_id" in locals() else "unknown",
                    "error": str(e),
                    "last_completed_step": "step3_cleanup_error",
                    "timestamp": utc_now(),
                },
            )

        if total_input % PROGRESS_EVERY_N == 0:
            logger.info(
                "Progress: seen=%d processed=%d skipped=%d (already_processed=%d) failed=%d contacts_upserts=%d",
                total_input,
                processed,
                skipped_total,
                skipped_already_processed,
                failed,
                contacts_upserts,
            )

    # Persist contacts store + snapshot
    if args.export_contacts:
        lookups = rebuild_all_lookups(contacts)
        atomic_write_json(contacts_store_path, {"contacts": contacts, "lookups": lookups})
        write_contacts_snapshot(contacts, contacts_snapshot_path)
        logger.info(
            "Contacts export complete: upserts=%d store=%s snapshot=%s events=%s",
            contacts_upserts,
            contacts_store_path,
            contacts_snapshot_path,
            contacts_events_path,
        )

    elapsed = time.time() - start_ts
    invariant_ok = total_input == processed + skipped_total + failed

    logger.info(
        "STEP 3 COMPLETE: total_input=%d processed=%d skipped=%d failed=%d output_written=%d contacts_upserts=%d "
        "invariant_ok=%s elapsed_sec=%.2f",
        total_input,
        processed,
        skipped_total,
        failed,
        output_written,
        contacts_upserts,
        invariant_ok,
        elapsed,
    )

    logger.info(
        "Logging performance notes: DEBUG logs inside the hot loop are safe to disable. "
        "Increase PROGRESS_EVERY_N to reduce log volume during large runs."
    )

    return 0


if __name__ == "__main__":
    import sys
    if "--loop" in sys.argv:
        while True:
            main()
            logger.warning("Cleanup job sleeping for 5 minutes (heartbeat)")
            time.sleep(300)
    else:
        raise SystemExit(main())
