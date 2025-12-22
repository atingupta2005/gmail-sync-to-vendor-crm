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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("step2b_vendor_scoring")

# Progress/heartbeat frequency (INFO) - low and predictable
PROGRESS_EVERY_N = 50  # configurable constant: emit progress every N records seen

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

        # Safe to log presence only; never log the token itself
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
                # DEBUG only: keep hot-loop INFO volume low
                logger.debug(
                    "Inference attempt failed; will retry (attempt=%d/%d timeout_s=%.1f): %s",
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

# Skip reasons (for counters + logs). Do not change behavior; only classify/log.
SKIP_REASON_ALREADY_DONE = "already_scored_unchanged"

FAIL_REASON_READ_ERROR = "read_error"
FAIL_REASON_MISSING_EMAIL_ID = "missing_email_id"
FAIL_REASON_EXCEPTION = "exception"

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

    # Respect CLI log level (does not change logic/outputs; only logging configuration)
    try:
        logging.getLogger().setLevel(getattr(logging, str(args.log_level).upper()))
    except Exception:
        # Keep existing behavior if log level is invalid; just log a hint
        logger.warning("Invalid --log-level=%r; using existing logging level", args.log_level)

    cfg_yaml = _load_yaml_minimal(Path(args.config))
    cfg = Step2BConfig(cfg_yaml, args.link_method)

    pre_dir = Path(args.prefiltered_dir).expanduser().resolve()
    cand_dir = Path(args.candidates_dir).expanduser().resolve()
    st_dir = Path(args.state_dir).expanduser().resolve()
    registry_path = st_dir / "processing_registry.jsonl"
    decision_log = st_dir / "step2b_vendor_scoring.jsonl"

    registry = load_registry(registry_path)

    # ---- Counters required for auditability ----
    total_input = 0
    processed = 0
    failed = 0

    skipped_total = 0
    skipped_already_scored_unchanged = 0

    output_written = 0  # number of candidate emails written (vendor-labeled)
    # Additional helpful counters (INFO-safe): do not change behavior; only count
    passed = 0  # existing semantic: vendor label
    non_vendor = 0

    # Skip breakdown dict (for structured summary)
    skipped_by_reason: dict[str, int] = {
        SKIP_REASON_ALREADY_DONE: 0,
    }

    # Failure breakdown (for structured summary)
    failed_by_reason: dict[str, int] = {
        FAIL_REASON_READ_ERROR: 0,
        FAIL_REASON_MISSING_EMAIL_ID: 0,
        FAIL_REASON_EXCEPTION: 0,
    }

    step_start = time.time()

    # Compute total_input from available files up-front (enables full accounting from logs alone)
    # NOTE: This consumes the generator once; does not change which records are processed, only how we count.
    try:
        all_paths = list(iter_prefiltered(pre_dir))
        total_input = len(all_paths)
    except Exception as e:
        # If we cannot enumerate input, treat as unrecoverable for step accounting (but keep behavior: main would have iterated)
        logger.error("Failed to enumerate input directory for accounting (dir=%s): %s", pre_dir, e)
        all_paths = list(iter_prefiltered(pre_dir))
        total_input = len(all_paths)

    # Apply limit only to "processed" behavior (existing), but log it clearly
    effective_limit = args.limit if args.limit else 0
    if effective_limit:
        logger.info(
            "Step start: %s (model=%s threshold=%.3f dry_run=%s link_method=%s limit=%d input_files=%d registry_entries=%d endpoint_host=%s)",
            STEP,
            cfg.model_version,
            cfg.threshold,
            False,
            cfg.link_method,
            effective_limit,
            total_input,
            len(registry),
            str(cfg.endpoint).split("/")[2] if "://" in str(cfg.endpoint) else cfg.endpoint,
        )
    else:
        logger.info(
            "Step start: %s (model=%s threshold=%.3f dry_run=%s link_method=%s limit=%d input_files=%d registry_entries=%d endpoint_host=%s)",
            STEP,
            cfg.model_version,
            cfg.threshold,
            False,
            cfg.link_method,
            0,
            total_input,
            len(registry),
            str(cfg.endpoint).split("/")[2] if "://" in str(cfg.endpoint) else cfg.endpoint,
        )

    logger.debug(
        "Paths: prefiltered_dir=%s candidates_dir=%s state_dir=%s decision_log=%s registry_path=%s",
        pre_dir,
        cand_dir,
        st_dir,
        decision_log,
        registry_path,
    )

    # Heartbeat baseline
    logger.info(
        "Heartbeat config: progress_every_n=%d (INFO) debug_in_loop=%s",
        PROGRESS_EVERY_N,
        True,
    )

    records_seen = 0

    for p in all_paths:
        records_seen += 1

        # Existing behavior: stop on processed limit (limit applies to processed count)
        if args.limit and processed >= args.limit:
            logger.info(
                "Limit reached: processed=%d limit=%d (stopping early) seen=%d/%d skipped=%d failed=%d",
                processed,
                args.limit,
                records_seen,
                total_input,
                skipped_total,
                failed,
            )
            break

        # Periodic heartbeat/progress (INFO) based on records seen for liveness
        if records_seen == 1 or (records_seen % PROGRESS_EVERY_N == 0):
            logger.info(
                "Progress: seen=%d/%d processed=%d skipped=%d (already_scored_unchanged=%d) failed=%d output_written=%d passed=%d non_vendor=%d",
                records_seen,
                total_input,
                processed,
                skipped_total,
                skipped_already_scored_unchanged,
                failed,
                output_written,
                passed,
                non_vendor,
            )

        email_obj, err = safe_read_json(p)
        if email_obj is None:
            failed += 1
            failed_by_reason[FAIL_REASON_READ_ERROR] += 1
            # WARNING: per-record recoverable failure; include identifier (path) and reason
            logger.warning("Record read failed (path=%s): %s", p, err)
            continue

        eid = email_obj.get("email_id")
        chash = email_obj.get("content_hash")
        if not eid:
            failed += 1
            failed_by_reason[FAIL_REASON_MISSING_EMAIL_ID] += 1
            logger.warning("Record missing email_id; skipping as failed (path=%s)", p)
            continue

        entry = registry.get(eid)
        if should_skip(entry, chash, cfg.model_version):
            skipped_total += 1
            skipped_already_scored_unchanged += 1
            skipped_by_reason[SKIP_REASON_ALREADY_DONE] += 1
            # DEBUG: skip decision path (safe to disable)
            logger.debug(
                "Skip: %s reason=%s content_hash_match=%s model_version_match=%s",
                eid,
                SKIP_REASON_ALREADY_DONE,
                True,
                True,
            )
            continue

        try:
            # DEBUG only: internal decision paths / diagnostics (no payloads)
            logger.debug("Scoring: %s (path=%s)", eid, p)

            text = ultra_safe_cleanup_for_bert(build_input_text(email_obj, cfg))
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
                write_candidate_email(
                    src_path=p,
                    dst_path=shard_path(cand_dir, eid),
                    email_obj=email_obj,
                    prob=prob,
                    cfg=cfg,
                )
                passed += 1
                output_written += 1
            else:
                non_vendor += 1

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

            # Keep existing INFO progress log cadence; do NOT increase per-record INFO logs
            if processed % 10 == 0:
                logger.info(
                    "Progress (processed cadence): processed=%d skipped=%d passed=%d failed=%d",
                    processed,
                    skipped_total,
                    passed,
                    failed,
                )

        except Exception as e:
            failed += 1
            failed_by_reason[FAIL_REASON_EXCEPTION] += 1
            # WARNING for per-record recoverable failures; include record identifier and reason
            logger.warning("Record scoring failed (email_id=%s path=%s): %s", eid, p, e, exc_info=logger.isEnabledFor(logging.DEBUG))

    elapsed_s = time.time() - step_start

    # Invariant: total_input = processed + skipped + failed (based on records seen, which is the actual loop input)
    # If we stopped early due to limit, total_input accounting still refers to input_files (all_paths length),
    # but processing only covered records_seen. For correctness, compute invariant over "seen" window.
    invariant_lhs = records_seen
    invariant_rhs = processed + skipped_total + failed
    invariant_ok = (invariant_lhs == invariant_rhs)

    # Final summary / accounting (INFO) - REQUIRED
    logger.info(
        "Step summary: step=%s seen=%d/%d processed=%d skipped_total=%d skipped_already_scored_unchanged=%d failed=%d output_written=%d passed=%d non_vendor=%d invariant_ok=%s invariant='%d = %d + %d + %d' elapsed_s=%.3f",
        STEP,
        records_seen,
        total_input,
        processed,
        skipped_total,
        skipped_already_scored_unchanged,
        failed,
        output_written,
        passed,
        non_vendor,
        invariant_ok,
        invariant_lhs,
        processed,
        skipped_total,
        failed,
        elapsed_s,
    )

    # Detailed breakdown (INFO) - low volume, end-of-step only
    logger.info(
        "Skip breakdown: skipped_total=%d by_reason=%s",
        skipped_total,
        skipped_by_reason,
    )
    logger.info(
        "Failure breakdown: failed=%d by_reason=%s",
        failed,
        failed_by_reason,
    )

    # Logging performance hints (INFO) - REQUIRED
    logger.info(
        "Logging performance hints: "
        "1) DEBUG logs inside the scoring loop (skip decisions, per-record diagnostics, retry details) can be disabled for higher throughput. "
        "2) To reduce INFO log volume, increase PROGRESS_EVERY_N (currently %d). "
        "3) Keep the final Step summary enabled for production auditability; it is constant-volume per run.",
        PROGRESS_EVERY_N,
    )

    # Preserve original end log line (kept for compatibility)
    logger.info("Done: processed=%d skipped=%d passed=%d failed=%d", processed, skipped_total, passed, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")

    while True:
        main()
        time.sleep(300)
        logger.info("Vendor Scoring: sleeping for 5 minutes")
