#!/usr/bin/env python3
"""
02b_bert_vendor_scoring.py

Vendor Relevance Scoring (Human-Bootstrapped)

- Reads prefiltered emails from data/emails_prefiltered/
- Builds concise classifier input text
- Calls a remote inference API expecting {"vendor_probability": float}
- Logs decisions to data/state/step2b_vendor_scoring.jsonl
- Copies passing emails to data/emails_candidates/
- Updates processing registry
- Skips unchanged emails or unchanged model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from dotenv import load_dotenv

import requests

LOG = logging.getLogger("step2b_vendor_scoring")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------
# Config Loading
# ----------------------------

def _load_yaml_minimal(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # noqa
    except ImportError as e:
        raise RuntimeError("PyYAML required to load config.yaml (pip install pyyaml)") from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(d: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class Step2BConfig:
    def __init__(self, cfg: dict, link_method: str):
        # remote inference
        self.endpoint = str(get_nested(cfg, ["bert", "endpoint"], "")).strip()
        if not self.endpoint:
            raise ValueError("config: bert.endpoint missing")

        embedding_cfg = dict(cfg["embedding"])
        self.auth_token = os.environ["HF_TOKEN"]
        
        LOG.info("HF auth token present: %s", bool(self.auth_token))

        self.model_version = str(get_nested(cfg, ["bert", "model_version"], "")).strip()
        if not self.model_version:
            raise ValueError("config: bert.model_version missing")

        self.threshold = float(get_nested(cfg, ["bert", "threshold"], 0.6))
        self.timeout_seconds = float(get_nested(cfg, ["bert", "timeout_seconds"], 20))
        self.max_retries = int(get_nested(cfg, ["bert", "max_retries"], 3))
        self.retry_backoff = float(get_nested(cfg, ["bert", "retry_backoff_base_seconds"], 1.5))

        # input text construction
        self.max_total_chars = int(get_nested(cfg, ["bert", "input", "max_total_chars"], 6000))
        self.max_body_chars = int(get_nested(cfg, ["bert", "input", "max_body_chars"], 4000))
        self.signature_max_lines = int(get_nested(cfg, ["bert", "input", "signature_max_lines"], 12))
        self.signature_min_signal_hits = int(get_nested(cfg, ["bert", "input", "signature_min_signal_hits"], 2))

        if link_method not in ("hardlink", "symlink", "copy"):
            raise ValueError("link_method must be hardlink|symlink|copy")
        self.link_method = link_method


# ----------------------------
# Registry Loading
# ----------------------------

class RegistryEntry:
    def __init__(self, rec: dict):
        self.email_id = rec.get("email_id")
        self.content_hash = rec.get("content_hash")
        self.last_completed_step = rec.get("last_completed_step")
        self.model_versions: dict[str, str] = rec.get("model_versions") or {}
        self.error = rec.get("error")


def load_registry(registry_path: Path) -> dict[str, RegistryEntry]:
    latest: dict[str, RegistryEntry] = {}
    if not registry_path.exists():
        return latest
    with registry_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                eid = rec.get("email_id")
                if eid:
                    latest[eid] = RegistryEntry(rec)
            except Exception:
                continue
    return latest


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------
# Email Helpers
# ----------------------------

def safe_read_json(p: Path) -> tuple[Optional[dict], Optional[str]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, "not a JSON object"
        return data, None
    except Exception as e:
        return None, str(e)


def extract_domain(hdrs: dict[str, Any]) -> str:
    raw_from = str(hdrs.get("from", "") or "")
    m = re.search(r"<([^>]+)>", raw_from)
    email = m.group(1).strip() if m else raw_from.strip()
    dom = email.split("@")[-1].lower() if "@" in email else ""
    return dom


def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}")
SIGNOFF_RE = re.compile(r"(?i)\b(thanks|regards|best|sincerely|kind regards)\b")


def detect_signature(body: str, max_lines: int, min_hits: int) -> str:
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    hits = 0
    if any(PHONE_RE.search(l) for l in tail):
        hits += 1
    if any(SIGNOFF_RE.search(l) for l in tail):
        hits += 1
    if hits >= min_hits:
        return "\n".join(tail)
    return ""


def build_input_text(email_obj: dict[str, Any], cfg: Step2BConfig) -> str:
    hdrs = email_obj.get("headers") or {}
    subj = str(hdrs.get("subject", "") or "").strip()
    dom = extract_domain(hdrs)

    body = email_obj.get("body") or {}
    raw_txt = str(body.get("raw_text", "") or "").strip()
    if not raw_txt and body.get("raw_html"):
        raw_txt = strip_html(body.get("raw_html", ""))

    if len(raw_txt) > cfg.max_body_chars:
        raw_txt = raw_txt[:cfg.max_body_chars].rstrip() + " …"

    sig = detect_signature(raw_txt, cfg.signature_max_lines, cfg.signature_min_signal_hits)

    parts = [
        "[SUBJECT]\n" + subj,
        "[FROM_DOMAIN]\n" + dom,
        "[BODY]\n" + raw_txt,
    ]
    if sig:
        parts.append("[SIGNATURE]\n" + sig)

    text = "\n\n".join(parts)
    if len(text) > cfg.max_total_chars:
        text = text[:cfg.max_total_chars].rstrip() + " …"
    return text


# ----------------------------
# Inference
# ----------------------------

def normalize_resp(resp: dict[str, Any]) -> float:
    """
    Only format A expected:
    { "vendor_probability": 0.83 }
    """
    p = resp.get("vendor_probability")
    if p is None:
        raise ValueError("inference response missing vendor_probability")
    return float(p)

def extract_vendor_probability_from_hf(resp) -> float:
    """
    Supports HF zero-shot formats:
    1) [{"label": "...", "score": ...}, ...]
    2) {"labels": [...], "scores": [...]}
    """
    # Format 1: list of {label, score}
    if isinstance(resp, list):
        for item in resp:
            label = str(item.get("label", "")).lower()
            if "vendor" in label:
                return float(item.get("score"))
        raise ValueError("Vendor label not found in HF list response")

    # Format 2: dict with labels/scores
    if isinstance(resp, dict):
        labels = resp.get("labels", [])
        scores = resp.get("scores", [])
        for label, score in zip(labels, scores):
            if "vendor" in label.lower():
                return float(score)
        raise ValueError("Vendor label not found in HF dict response")

    raise ValueError("Unrecognized HF response format")


def call_inference(text: str, cfg: Step2BConfig) -> float:
    headers = {
        "Authorization": f"Bearer {cfg.auth_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": [
                "vendor related business email",
                "non-vendor personal or automated email",
            ],
            "hypothesis_template": "This email is {}.",
        },
    }

    last_err = None
    for attempt in range(cfg.max_retries + 1):
        try:
            r = requests.post(
                cfg.endpoint,
                headers=headers,
                json=payload,
                timeout=cfg.timeout_seconds,
            )
            r.raise_for_status()
            data = r.json()

            return extract_vendor_probability_from_hf(data)

        except Exception as e:
            last_err = str(e)
            if attempt < cfg.max_retries:
                time.sleep(2 ** attempt)
                continue

    raise RuntimeError(f"Hugging Face inference failed: {last_err}")



# ----------------------------
# File iteration
# ----------------------------

def iter_prefiltered(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for p in root.rglob("*.json"):
        yield p


def shard_path(root: Path, email_id: str) -> Path:
    shard = email_id[:2].lower() if email_id else "00"
    return root / shard / f"{email_id}.json"


def link_or_copy(src: Path, dst: Path, method: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if method == "copy":
        shutil.copy2(src, dst)
    else:
        try:
            # hardlink or symlink
            if method == "symlink":
                dst.symlink_to(src.resolve())
            else:
                dst.link_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)


# ----------------------------
# Main
# ----------------------------

STEP = "step2b_vendor_scoring"


def should_skip(entry: Optional[RegistryEntry], content_hash: str, model_version: str) -> bool:
    if not entry:
        return False
    if entry.last_completed_step != STEP:
        return False
    if entry.content_hash != content_hash:
        return False
    prev = entry.model_versions.get("step2b_model_version")
    if prev != model_version:
        return False
    if entry.error:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefiltered-dir", required=True)
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--link-method", default="hardlink", choices=["hardlink", "symlink", "copy"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    cfg_yaml = _load_yaml_minimal(Path(args.config))
    cfg = Step2BConfig(cfg_yaml, args.link_method)

    pre_dir = Path(args.prefiltered_dir).expanduser().resolve()
    cand_dir = Path(args.candidates_dir).expanduser().resolve()
    st_dir = Path(args.state_dir).expanduser().resolve()
    registry_path = st_dir / "processing_registry.jsonl"
    decision_log = st_dir / "step2b_vendor_scoring.jsonl"

    registry = load_registry(registry_path)

    processed = skipped = passed = failed = 0

    for p in iter_prefiltered(pre_dir):
        if args.limit and processed >= args.limit:
            break

        email_obj, err = safe_read_json(p)
        if email_obj is None:
            failed += 1
            continue

        eid = email_obj.get("email_id")
        chash = email_obj.get("content_hash")
        if not eid:
            failed += 1
            continue

        entry = registry.get(eid)
        if should_skip(entry, chash, cfg.model_version):
            skipped += 1
            continue

        try:
            text = build_input_text(email_obj, cfg)
            prob = call_inference(text, cfg)
            label = "vendor" if prob >= cfg.threshold else "non_vendor"

            append_jsonl(decision_log, {
                "email_id": eid,
                "vendor_probability": prob,
                "predicted_label": label,
                "threshold_used": cfg.threshold,
                "model_version": cfg.model_version,
                "timestamp": utc_now_iso(),
            })

            if label == "vendor":
                link_or_copy(p, shard_path(cand_dir, eid), cfg.link_method)
                passed += 1

            append_jsonl(registry_path, {
                "email_id": eid,
                "content_hash": chash,
                "last_completed_step": STEP,
                "model_versions": {"step2b_model_version": cfg.model_version},
                "updated_at": utc_now_iso(),
                "error": None,
            })

            registry[eid] = RegistryEntry({
                "email_id": eid,
                "content_hash": chash,
                "last_completed_step": STEP,
                "model_versions": {"step2b_model_version": cfg.model_version},
            })
            processed += 1
        except Exception as e:
            failed += 1
            logging.error("error scoring %s: %s", eid, e)

    LOG.info("Done: processed=%d skipped=%d passed=%d failed=%d", processed, skipped, passed, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    
    raise SystemExit(main())
