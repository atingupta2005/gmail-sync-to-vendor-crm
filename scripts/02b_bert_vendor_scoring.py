#!/usr/bin/env python3
"""
02b_bert_vendor_scoring.py

Vendor Relevance Scoring (Human-Bootstrapped)
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
from collections import Counter
from dotenv import load_dotenv

import requests

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("step2b_vendor_scoring")

PROGRESS_EVERY_N = 20
STEP = "step2b_vendor_scoring"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------
# Config Loading
# ----------------------------

def _load_yaml_minimal(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML required to load config.yaml") from e
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
        self.endpoint = str(get_nested(cfg, ["bert", "endpoint"], "")).strip()
        if not self.endpoint:
            raise ValueError("config: bert.endpoint missing")

        self.auth_token = os.environ["HF_TOKEN"]
        logger.info("HF auth token present=%s", bool(self.auth_token))

        self.model_version = str(get_nested(cfg, ["bert", "model_version"], "")).strip()
        if not self.model_version:
            raise ValueError("config: bert.model_version missing")

        self.threshold = float(get_nested(cfg, ["bert", "threshold"], 0.6))
        self.timeout_seconds = float(get_nested(cfg, ["bert", "timeout_seconds"], 20))
        self.max_retries = int(get_nested(cfg, ["bert", "max_retries"], 3))
        self.retry_backoff = float(get_nested(cfg, ["bert", "retry_backoff_base_seconds"], 1.5))

        self.max_total_chars = int(get_nested(cfg, ["bert", "input", "max_total_chars"], 6000))
        self.max_body_chars = int(get_nested(cfg, ["bert", "input", "max_body_chars"], 4000))
        self.signature_max_lines = int(get_nested(cfg, ["bert", "input", "signature_max_lines"], 12))
        self.signature_min_signal_hits = int(get_nested(cfg, ["bert", "input", "signature_min_signal_hits"], 2))

        if link_method not in ("hardlink", "symlink", "copy"):
            raise ValueError("link_method must be hardlink|symlink|copy")
        self.link_method = link_method


# ----------------------------
# Registry
# ----------------------------

class RegistryEntry:
    def __init__(self, rec: dict):
        self.email_id = rec.get("email_id")
        self.content_hash = rec.get("content_hash")
        self.last_completed_step = rec.get("last_completed_step")
        self.model_versions = rec.get("model_versions") or {}
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
# Helpers
# ----------------------------

def safe_read_json(p: Path) -> tuple[Optional[dict], Optional[str]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, "not an object"
        return data, None
    except Exception as e:
        return None, str(e)


def iter_prefiltered(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    yield from root.rglob("*.json")


def shard_path(root: Path, email_id: str) -> Path:
    return root / email_id[:2].lower() / f"{email_id}.json"


# ----------------------------
# Skip Logic
# ----------------------------

def should_skip(entry: Optional[RegistryEntry], content_hash: str, model_version: str) -> bool:
    if not entry:
        return False
    if entry.last_completed_step != STEP:
        return False
    if entry.content_hash != content_hash:
        return False
    if entry.model_versions.get("step2b_model_version") != model_version:
        return False
    if entry.error:
        return False
    return True


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefiltered-dir", required=True)
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--link-method", default="hardlink", choices=["hardlink", "symlink", "copy"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    step_start = time.monotonic()

    cfg_yaml = _load_yaml_minimal(Path(args.config))
    cfg = Step2BConfig(cfg_yaml, args.link_method)

    pre_dir = Path(args.prefiltered_dir)
    cand_dir = Path(args.candidates_dir)
    st_dir = Path(args.state_dir)
    registry_path = st_dir / "processing_registry.jsonl"
    decision_log = st_dir / "step2b_vendor_scoring.jsonl"

    registry = load_registry(registry_path)

    # ---- counters ----
    total_input = 0
    processed = 0
    failed = 0
    skipped_total = 0
    skipped_reasons = Counter()
    output_written = 0

    logger.info(
        "step_start step=%s model_version=%s threshold=%.3f prefiltered_dir=%s",
        STEP,
        cfg.model_version,
        cfg.threshold,
        pre_dir,
    )

    for p in iter_prefiltered(pre_dir):
        if args.limit and total_input >= args.limit:
            break

        total_input += 1

        email_obj, err = safe_read_json(p)
        if email_obj is None:
            failed += 1
            logger.warning("record_failed step=%s path=%s reason=%s", STEP, p, err)
            continue

        eid = email_obj.get("email_id")
        chash = email_obj.get("content_hash")
        if not eid:
            failed += 1
            logger.warning("record_failed step=%s path=%s reason=missing_email_id", STEP, p)
            continue

        entry = registry.get(eid)
        if should_skip(entry, chash, cfg.model_version):
            skipped_total += 1
            skipped_reasons["unchanged"] += 1
            logger.debug("record_skipped reason=unchanged email_id=%s", eid)
            continue

        try:
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

        except Exception as e:
            failed += 1
            logger.warning(
                "record_failed step=%s email_id=%s error=%s",
                STEP,
                eid,
                e,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )

        if total_input % PROGRESS_EVERY_N == 0:
            logger.info(
                "step_progress step=%s seen=%d processed=%d skipped=%d failed=%d passed=%d",
                STEP,
                total_input,
                processed,
                skipped_total,
                failed,
                output_written,
            )

    elapsed = time.monotonic() - step_start
    invariant_ok = total_input == processed + skipped_total + failed

    logger.info(
        "step_summary step=%s total_input=%d processed=%d skipped_total=%d failed=%d output_written=%d skipped_breakdown=%s invariant_ok=%s elapsed_s=%.2f",
        STEP,
        total_input,
        processed,
        skipped_total,
        failed,
        output_written,
        dict(skipped_reasons),
        invariant_ok,
        elapsed,
    )

    if not invariant_ok:
        logger.error(
            "step_invariant_violation step=%s lhs=%d rhs=%d",
            STEP,
            total_input,
            processed + skipped_total + failed,
        )

    logger.info(
        "logging_hints step=%s "
        "Disable DEBUG to remove per-record diagnostics; "
        "increase PROGRESS_EVERY_N to reduce INFO volume; "
        "inference behavior unaffected by logging configuration.",
        STEP,
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    load_dotenv(".env")
    while True:
        main()
        logger.info("Vendor Scoring idle: sleeping for 5 minutes")
        time.sleep(300)
