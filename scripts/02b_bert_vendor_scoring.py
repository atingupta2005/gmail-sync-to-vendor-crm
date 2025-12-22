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

SESSION = requests.Session()
BATCH_SIZE = 2

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("step2b_vendor_scoring")

# ----------------------------
# LOGGING ADDITIONS (safe, constant-volume)
# ----------------------------

# Emit heartbeat/progress logs every N records SEEN (not processed) so incremental runs are visible.
PROGRESS_EVERY_N = 200

# Skip/fail reason keys (for counters + summary)
SKIP_REASON_ALREADY_SCORED_UNCHANGED = "already_scored_unchanged"

FAIL_REASON_READ_ERROR = "read_error"
FAIL_REASON_MISSING_EMAIL_ID = "missing_email_id"
FAIL_REASON_SCORING_EXCEPTION = "scoring_exception"


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
        
        logger.info("HF auth token present: %s", bool(self.auth_token))

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


def write_candidate_email(
    src_path: Path,
    dst_path: Path,
    email_obj: dict[str, Any],
    prob: float,
    cfg: Step2BConfig,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    out = dict(email_obj)  # shallow copy is fine

    out["bert"] = {
        "vendor_probability": prob,
        "threshold": cfg.threshold,
        "model_version": cfg.model_version,
        "endpoint": cfg.endpoint,
        "label": "vendor",
        "scored_at": utc_now_iso(),
    }

    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


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


def ultra_safe_cleanup_for_bert(text: str, max_chars: int = 3500) -> str:
    if not text:
        return ""

    original_len = len(text)

    # Normalize whitespace only
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Hard length cap (cost + latency protection)
    if len(text) > max_chars:
        logger.debug(
            "BERT input truncated: original_len=%d max_chars=%d",
            original_len,
            max_chars,
        )
        text = text[:max_chars].rstrip() + " …"

    return text.strip()



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
            if label.startswith("vendor"):
                return float(item.get("score"))
        raise ValueError("Vendor label not found in HF list response")

    # Format 2: dict with labels/scores
    if isinstance(resp, dict):
        labels = resp.get("labels", [])
        scores = resp.get("scores", [])
        for label, score in zip(labels, scores):
            if label.lower().startswith("vendor"):
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
            r = SESSION.post(
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
                # DEBUG only: safe to disable, avoids INFO spam in hot path
                logger.debug(
                    "Inference attempt failed; retrying attempt=%d/%d timeout_s=%.1f err=%s",
                    attempt + 1,
                    cfg.max_retries + 1,
                    cfg.timeout_seconds,
                    last_err,
                )
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

def call_inference_batch(texts: list[str], cfg: Step2BConfig) -> list[float]:
    headers = {
        "Authorization": f"Bearer {cfg.auth_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": texts,
        "parameters": {
            "candidate_labels": [
                "vendor related business email",
                "non-vendor personal or automated email",
            ],
            "hypothesis_template": "This email is {}.",
        },
    }

    r = SESSION.post(
        cfg.endpoint,
        headers=headers,
        json=payload,
        timeout=cfg.timeout_seconds,
    )
    r.raise_for_status()
    data = r.json()

    return [extract_vendor_probability_from_hf(d) for d in data]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefiltered-dir", required=True)
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--link-method", default="hardlink", choices=["hardlink", "symlink", "copy"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-level", default="DEBUG")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )

    # LOGGING ADDITION: honor --log-level (logging-only change)
    try:
        logging.getLogger().setLevel(getattr(logging, str(args.log_level).upper()))
    except Exception:
        logger.warning("Invalid --log-level=%r; using existing logging level", args.log_level)

    cfg_yaml = _load_yaml_minimal(Path(args.config))
    cfg = Step2BConfig(cfg_yaml, args.link_method)

    logger.info("Vendor scoring started (model=%s)", cfg.model_version)

    pre_dir = Path(args.prefiltered_dir).expanduser().resolve()
    cand_dir = Path(args.candidates_dir).expanduser().resolve()
    st_dir = Path(args.state_dir).expanduser().resolve()
    registry_path = st_dir / "processing_registry_step2b_vendor_scoring.jsonl"
    decision_log = st_dir / "step2b_vendor_scoring.jsonl"

    registry = load_registry(registry_path)

    # ----------------------------
    # LOGGING ADDITION: required counters + breakdowns
    # ----------------------------
    input_discovered = 0   # total files discovered in prefiltered-dir
    total_input = 0        # total inputs actually accounted for (seen) (supports invariant)
    processed = 0
    failed = 0
    skipped_total = 0
    skipped_by_reason: dict[str, int] = {SKIP_REASON_ALREADY_SCORED_UNCHANGED: 0}
    failed_by_reason: dict[str, int] = {
        FAIL_REASON_READ_ERROR: 0,
        FAIL_REASON_MISSING_EMAIL_ID: 0,
        FAIL_REASON_SCORING_EXCEPTION: 0,
    }
    output_written = 0

    # Keep existing semantic counters (passed exists in original)
    passed = 0

    # Existing "skipped" name used in legacy logs; keep as alias to skipped_total to avoid confusion
    skipped = 0

    step_start = time.time()

    # LOGGING ADDITION: discover inputs once to log input size
    # (Does not change behavior; iteration order remains rglob-produced order.)
    all_paths = list(iter_prefiltered(pre_dir))
    input_discovered = len(all_paths)

    logger.info(
        "Step start: step=%s input_files=%d registry_entries=%d limit=%d link_method=%s endpoint_host=%s timeout_s=%.1f max_retries=%d threshold=%.3f model_version=%s",
        STEP,
        input_discovered,
        len(registry),
        args.limit,
        cfg.link_method,
        (str(cfg.endpoint).split("/")[2] if "://" in str(cfg.endpoint) else cfg.endpoint),
        cfg.timeout_seconds,
        cfg.max_retries,
        cfg.threshold,
        cfg.model_version,
    )
    logger.info(
        "Incremental behavior: will SKIP records only when last_completed_step=%s AND content_hash matches AND model_version matches AND no prior error.",
        STEP,
    )
    logger.info(
        "Heartbeat: INFO progress every %d records seen (DEBUG skip decisions and per-record diagnostics are safe to disable).",
        PROGRESS_EVERY_N,
    )

    records_seen = 0
    failed_legacy = 0  # maintain original variable name usage semantics in end return

    batch_texts = []
    batch_records = []
    for p in all_paths:
        records_seen += 1

        # Existing behavior: limit applies to processed count
        if args.limit and processed >= args.limit:
            logger.info(
                "Limit reached: processed=%d limit=%d seen=%d/%d skipped=%d failed=%d output_written=%d",
                processed,
                args.limit,
                records_seen,
                input_discovered,
                skipped_total,
                failed,
                output_written,
            )
            break

        # LOGGING ADDITION: heartbeat based on records seen (incremental runs remain visible)
        if records_seen == 1 or (records_seen % PROGRESS_EVERY_N == 0):
            logger.info(
                "Progress: seen=%d/%d processed=%d skipped=%d (already_scored_unchanged=%d) failed=%d output_written=%d",
                records_seen,
                input_discovered,
                processed,
                skipped_total,
                skipped_by_reason[SKIP_REASON_ALREADY_SCORED_UNCHANGED],
                failed,
                output_written,
            )

        email_obj, err = safe_read_json(p)
        if email_obj is None:
            failed += 1
            failed_legacy += 1
            failed_by_reason[FAIL_REASON_READ_ERROR] += 1
            # WARNING: recoverable per-record failure with identifier and reason
            logger.warning("Read failed: path=%s reason=%s", p, err)
            continue

        eid = email_obj.get("email_id")
        chash = email_obj.get("content_hash")
        if not eid:
            failed += 1
            failed_legacy += 1
            failed_by_reason[FAIL_REASON_MISSING_EMAIL_ID] += 1
            logger.warning("Record missing email_id; marking failed: path=%s", p)
            continue

        entry = registry.get(eid)
        if should_skip(entry, chash, cfg.model_version):
            skipped_total += 1
            skipped += 1  # keep original name aligned
            skipped_by_reason[SKIP_REASON_ALREADY_SCORED_UNCHANGED] += 1
            # DEBUG: skip decision (safe to disable)
            logger.debug("Skip: email_id=%s reason=%s", eid, SKIP_REASON_ALREADY_SCORED_UNCHANGED)
            continue

        try:
            text = ultra_safe_cleanup_for_bert(build_input_text(email_obj, cfg))

            batch_texts.append(text)
            batch_records.append((eid, email_obj, p, chash))

            if len(batch_texts) < BATCH_SIZE:
                continue

            logger.debug(
                "Calling batch inference: batch_size=%d email_ids=%s",
                len(batch_records),
                [eid for (eid, _, _, _) in batch_records],
            )

            probs = call_inference_batch(batch_texts, cfg)

            for prob, (eid, email_obj, p, chash) in zip(probs, batch_records):
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
                    write_candidate_email(
                        src_path=p,
                        dst_path=shard_path(cand_dir, eid),
                        email_obj=email_obj,
                        prob=prob,
                        cfg=cfg,
                    )
                    passed += 1
                    output_written += 1

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

                if processed % 10 == 0:
                    logger.info(
                        "Progress: processed=%d skipped=%d passed=%d failed=%d",
                        processed, skipped, passed, failed_legacy
                    )

            # IMPORTANT: clear batch after processing
            batch_texts.clear()
            batch_records.clear()


        except Exception as e:
            failed += 1
            failed_legacy += 1
            failed_by_reason[FAIL_REASON_SCORING_EXCEPTION] += 1
            # WARNING: recoverable per-record failure; include record identifier and reason
            logger.warning("Error scoring email_id=%s: %s", eid, e, exc_info=logger.isEnabledFor(logging.DEBUG))


    # Flush leftover batch at end (if any)
    if batch_texts:
        probs = call_inference_batch(batch_texts, cfg)

        for prob, (eid, email_obj, p, chash) in zip(probs, batch_records):
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
                write_candidate_email(
                    src_path=p,
                    dst_path=shard_path(cand_dir, eid),
                    email_obj=email_obj,
                    prob=prob,
                    cfg=cfg,
                )
                passed += 1
                output_written += 1

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

    # LOGGING ADDITION: total input accounted for invariant uses actual seen records (works even with early limit stop)
    total_input = records_seen

    elapsed_s = time.time() - step_start
    invariant_ok = (total_input == processed + skipped_total + failed)

    # Final summary / accounting (INFO) - REQUIRED
    logger.info(
        "Step summary: step=%s input_discovered=%d total_input=%d processed=%d skipped_total=%d skipped_breakdown=%s failed=%d failed_breakdown=%s output_written=%d passed=%d invariant_ok=%s invariant='%d = %d + %d + %d' elapsed_s=%.3f",
        STEP,
        input_discovered,
        total_input,
        processed,
        skipped_total,
        skipped_by_reason,
        failed,
        failed_by_reason,
        output_written,
        passed,
        invariant_ok,
        total_input,
        processed,
        skipped_total,
        failed,
        elapsed_s,
    )

    # Logging performance hints (INFO) - REQUIRED
    logger.info(
        "Logging performance hints: DEBUG logs inside the hot loop (skip decisions, per-record diagnostics, inference retries) are safe to disable. "
        "To reduce INFO volume, increase PROGRESS_EVERY_N (currently %d). "
        "Keep the final Step summary enabled for auditability (constant-volume per run).",
        PROGRESS_EVERY_N,
    )

    # Preserve original final log line (compat + familiarity)
    logger.info("Done: processed=%d skipped=%d passed=%d failed=%d", processed, skipped, passed, failed_legacy)
    return 0 if failed_legacy == 0 else 1


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    
    while True:
        main()
        time.sleep(300)
        logger.info("Vendor Scoring: sleeping for 5 minutes")
