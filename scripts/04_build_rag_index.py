from dotenv import load_dotenv
import hashlib
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from functools import wraps
from time import perf_counter
import argparse
import logging
import json
import os
import time
import yaml
import requests

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger("ingest")

# -------------------------------------------------------------------
# Pipeline logging constants (safe + bounded volume)
# -------------------------------------------------------------------

# Heartbeat/progress log frequency for the main loop (bounded INFO volume).
# Increase this number to reduce log volume; decrease for more granular monitoring.
PROGRESS_EVERY_RECORDS = 200

# INFO log frequency for discovered files (kept low + bounded).
FOUND_EMAIL_LOG_EVERY = 1000

# -------------------------------------------------------------------
# Utility logging helpers
# -------------------------------------------------------------------

def _fmt_skip_breakdown(skipped_by_reason: Dict[str, int]) -> str:
    if not skipped_by_reason:
        return "{}"
    # Stable ordering to keep logs diff-friendly
    items = ", ".join(f"{k}={skipped_by_reason[k]}" for k in sorted(skipped_by_reason))
    return "{" + items + "}"

def _invariant_ok(total_input: int, processed: int, skipped_total: int, failed: int) -> bool:
    return total_input == (processed + skipped_total + failed)

def log_progress(
    *,
    seen: int,
    total_input: int,
    processed: int,
    failed: int,
    skipped_total: int,
    skipped_by_reason: Dict[str, int],
    output_written: int,
    every: int = PROGRESS_EVERY_RECORDS,
) -> None:
    # Periodic heartbeat only (bounded INFO volume)
    if seen > 0 and (seen % every == 0):
        logger.info(
            "[PROGRESS] seen=%s/%s processed=%s skipped=%s failed=%s output_written=%s skipped_breakdown=%s time=%sZ",
            seen,
            total_input,
            processed,
            skipped_total,
            failed,
            output_written,
            _fmt_skip_breakdown(skipped_by_reason),
            datetime.utcnow().isoformat(),
        )

def debug_step(name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            logger.debug("[DEBUG] → %s.start", name)
            t0 = perf_counter()
            result = fn(*args, **kwargs)
            dt = perf_counter() - t0
            size = len(result) if isinstance(result, list) else result
            # NOTE: This is INFO today (existing behavior) and intentionally left unchanged.
            logger.info("[DEBUG] ← %s.end | result=%s | %.2fs", name, size, dt)
            return result
        return wrapper
    return decorator

# -------------------------------------------------------------------
# Email chunking
# -------------------------------------------------------------------

def chunk_email(cleaned_email: dict) -> List[Dict]:
    chunks: List[Dict] = []

    text = (cleaned_email.get("cleaned_text") or "").strip()
    if text:
        body = text[:1800]
        last_period = body.rfind(".")
        if last_period > 200:
            body = body[: last_period + 1]
        chunks.append({"chunk_index": 0, "text": body})

    lines = text.splitlines()
    if len(lines) >= 4:
        tail = "\n".join(lines[-8:]).strip()
        if any(k in tail.lower() for k in ["@", "tel", "phone", "mobile", "www", "http"]) and len(tail) <= 300:
            chunks.append({"chunk_index": 1, "text": tail})

    subject = (cleaned_email.get("subject") or "").strip()
    if subject:
        chunks.append({"chunk_index": 2, "text": subject[:150]})

    return chunks[:3]

# -------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------

_SESSION = requests.Session()

def embed_texts(
    texts: List[str],
    *,
    endpoint: str,
    auth_token: str,
    timeout_seconds: int,
    max_retries: int,
) -> List[List[float]]:

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    payload = {"inputs": texts}

    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.post(endpoint, headers=headers, json=payload, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json()

            vectors = []
            for item in data:
                if isinstance(item[0], (int, float)):
                    vectors.append(item)
                else:
                    pooled = [sum(col) / len(item) for col in zip(*item)]
                    vectors.append(pooled)

            return vectors

        except Exception as e:
            if attempt == max_retries:
                logger.exception("Embedding failed after %s attempts", max_retries)
                raise
            sleep_s = min(2 ** (attempt - 1), 8)
            logger.warning("Embedding attempt %s failed: %s — retrying in %ss", attempt, e, sleep_s)
            time.sleep(sleep_s)

# -------------------------------------------------------------------
# Pinecone
# -------------------------------------------------------------------

from pinecone import Pinecone

def init_pinecone_index(*, api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

# -------------------------------------------------------------------
# Vector construction
# -------------------------------------------------------------------

def embed_email_chunks(*, cleaned_email: dict, embedding_cfg: dict) -> List[Dict]:
    chunks = chunk_email(cleaned_email)
    if not chunks:
        return []

    vectors = embed_texts(
        [c["text"] for c in chunks],
        endpoint=embedding_cfg["endpoint"],
        auth_token=embedding_cfg["auth_token"],
        timeout_seconds=embedding_cfg["timeout_seconds"],
        max_retries=embedding_cfg["max_retries"],
    )

    if len(vectors) != len(chunks):
        raise ValueError("Embedding count mismatch")

    return [
        {
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "embedding": v,
        }
        for c, v in zip(chunks, vectors)
    ]

def build_vector_records(*, email_id: str, embedded_chunks: List[Dict], metadata: dict) -> List[Dict]:
    return [
        {
            "id": f"{email_id}::chunk_{c['chunk_index']}",
            "values": c["embedding"],
            "metadata": {
                **metadata,
                "email_id": email_id,
                "chunk_index": c["chunk_index"],
            },
        }
        for c in embedded_chunks
    ]

def upsert_vectors(*, index, records: List[Dict], buffer: List[Dict], batch_size: int = 500):
    buffer.extend(records)
    if len(buffer) >= batch_size:
        logger.info("UPSERT FLUSH: %s", len(buffer))
        index.upsert(vectors=buffer, namespace="emails")
        buffer.clear()

# -------------------------------------------------------------------
# Email processing
# -------------------------------------------------------------------

def process_single_cleaned_email(
    *,
    cleaned_email: dict,
    embedding_cfg: dict,
    pinecone_index,
    base_metadata: dict,
    upsert_buffer: List[Dict],
) -> int:

    embedded_chunks = embed_email_chunks(
        cleaned_email=cleaned_email,
        embedding_cfg=embedding_cfg,
    )

    if not embedded_chunks:
        return 0

    records = build_vector_records(
        email_id=cleaned_email["email_id"],
        embedded_chunks=embedded_chunks,
        metadata=base_metadata,
    )

    upsert_vectors(
        index=pinecone_index,
        records=records,
        buffer=upsert_buffer,
    )

    return len(records)

# -------------------------------------------------------------------
# Disk IO + registry
# -------------------------------------------------------------------

def load_cleaned_email(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def derive_base_metadata(cleaned_email: dict) -> dict:
    meta = {}
    if cleaned_email.get("subject"):
        meta["subject"] = cleaned_email["subject"]
    sender = cleaned_email.get("from") or cleaned_email.get("sender")
    if sender and "@" in sender:
        meta["sender_domain"] = sender.split("@")[-1].lower()
    if cleaned_email.get("date"):
        meta["email_date"] = cleaned_email["date"]
    if isinstance(cleaned_email.get("bert_probability"), (int, float)):
        meta["bert_probability"] = float(cleaned_email["bert_probability"])
    return meta

def iter_cleaned_emails(cleaned_dir: Path):
    for path in cleaned_dir.rglob("*.json"):
        try:
            yield path, load_cleaned_email(path)
        except Exception:
            logger.exception("Failed to load %s", path)

def load_registry(path: Path) -> dict:
    if not path.exists():
        return {}
    registry = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                registry[rec["email_id"]] = rec
            except Exception:
                pass
    return registry

def append_registry(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------
# Main processing loop
# -------------------------------------------------------------------

def process_all_cleaned_emails(
    *,
    cleaned_dir: Path,
    embedding_cfg: dict,
    pinecone_index,
    registry_path: Path,
) -> int:

    step_started_at = perf_counter()

    # Counters required by prompt
    total_input = 0
    processed = 0
    failed = 0
    skipped_total = 0
    skipped_by_reason: Dict[str, int] = {}
    output_written = 0  # number of vector records produced (and scheduled/written via upsert)

    # Existing locals (kept to avoid any behavioral changes)
    upsert_buffer: List[Dict] = []
    registry = load_registry(registry_path)
    processed_emails = 0
    total_vectors = 0

    # Determine total_input up front so logs can be audited from logs alone
    # (This is not a logic/control-flow change; it only enumerates the same generator once for counting.)
    try:
        total_input = sum(1 for _ in cleaned_dir.rglob("*.json"))
    except Exception:
        logger.exception("Failed to count input files under cleaned_dir=%s", cleaned_dir)
        # Keep going; total_input may remain 0/partial

    logger.info(
        "[STEP START] process_all_cleaned_emails cleaned_dir=%s total_input=%s registry_path=%s model_version=%s endpoint=%s progress_every=%s found_log_every=%s",
        str(cleaned_dir),
        total_input,
        str(registry_path),
        embedding_cfg.get("model_version"),
        embedding_cfg.get("endpoint"),
        PROGRESS_EVERY_RECORDS,
        FOUND_EMAIL_LOG_EVERY,
    )

    seen = 0

    for path, cleaned_email in iter_cleaned_emails(cleaned_dir):
        seen += 1

        if seen % FOUND_EMAIL_LOG_EVERY == 0:
            # Keep low-volume: one log per N discovered inputs
            logger.info("FOUND CLEANED EMAIL: %s", path)

        # Validate record identifier availability without logging payload/PII
        email_id = None
        try:
            email_id = cleaned_email["email_id"]
        except Exception:
            failed += 1
            logger.error(
                "Unrecoverable record: missing email_id (cannot proceed) path=%s seen=%s/%s",
                str(path),
                seen,
                total_input,
            )
            log_progress(
                seen=seen,
                total_input=total_input,
                processed=processed,
                failed=failed,
                skipped_total=skipped_total,
                skipped_by_reason=skipped_by_reason,
                output_written=output_written,
            )
            continue

        content_hash = cleaned_email.get("content_hash")
        model_version = embedding_cfg["model_version"]

        cleaned_text = (cleaned_email.get("cleaned_text") or "").strip()
        cleaned_text_hash = sha1_text(cleaned_text)

        reg = registry.get(email_id)
        if (
            reg
            and reg.get("cleaned_text_hash") == cleaned_text_hash
            and reg.get("embedding_model_version") == model_version
        ):
            skipped_total += 1
            reason = "already_indexed_same_hash_and_model"
            skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
            logger.debug(
                "SKIP email_id=%s reason=%s seen=%s/%s",
                email_id,
                reason,
                seen,
                total_input,
            )
            log_progress(
                seen=seen,
                total_input=total_input,
                processed=processed,
                failed=failed,
                skipped_total=skipped_total,
                skipped_by_reason=skipped_by_reason,
                output_written=output_written,
            )
            continue

        try:
            n = process_single_cleaned_email(
                cleaned_email=cleaned_email,
                embedding_cfg=embedding_cfg,
                pinecone_index=pinecone_index,
                base_metadata=derive_base_metadata(cleaned_email),
                upsert_buffer=upsert_buffer,
            )

            if n == 0:
                skipped_total += 1
                reason = "no_chunks_or_no_vectors_produced"
                skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
                logger.debug(
                    "SKIP email_id=%s reason=%s seen=%s/%s",
                    email_id,
                    reason,
                    seen,
                    total_input,
                )
            else:
                processed += 1
                output_written += n

            append_registry(
                registry_path,
                {
                    "email_id": email_id,
                    "cleaned_text_hash": cleaned_text_hash,
                    "embedding_model_version": model_version,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Existing counters (kept)
            processed_emails += 1
            total_vectors += n

            log_progress(
                seen=seen,
                total_input=total_input,
                processed=processed,
                failed=failed,
                skipped_total=skipped_total,
                skipped_by_reason=skipped_by_reason,
                output_written=output_written,
            )

        except Exception as e:
            # Recoverable per-record failure: log record identifier + reason, no payload
            failed += 1
            logger.warning(
                "Per-record failure email_id=%s path=%s error=%s",
                email_id,
                str(path),
                e,
            )
            logger.exception("Failed email %s", email_id)

            log_progress(
                seen=seen,
                total_input=total_input,
                processed=processed,
                failed=failed,
                skipped_total=skipped_total,
                skipped_by_reason=skipped_by_reason,
                output_written=output_written,
            )

    if upsert_buffer:
        logger.info("FINAL UPSERT FLUSH: %s", len(upsert_buffer))
        try:
            pinecone_index.upsert(vectors=upsert_buffer, namespace="emails")
            upsert_buffer.clear()
        except Exception:
            # Unrecoverable output failure
            logger.exception("Unrecoverable failure: final upsert flush failed (buffer_size=%s)", len(upsert_buffer))
            raise

    elapsed_s = perf_counter() - step_started_at

    invariant_ok = _invariant_ok(total_input, processed, skipped_total, failed)

    logger.info(
        "[STEP SUMMARY] total_input=%s processed=%s skipped_total=%s skipped_breakdown=%s failed=%s output_written=%s invariant_ok=%s elapsed_s=%.3f",
        total_input,
        processed,
        skipped_total,
        _fmt_skip_breakdown(skipped_by_reason),
        failed,
        output_written,
        invariant_ok,
        elapsed_s,
    )

    if not invariant_ok:
        logger.error(
            "Invariant violation: total_input(%s) != processed(%s)+skipped(%s)+failed(%s)",
            total_input,
            processed,
            skipped_total,
            failed,
        )

    logger.info(
        "[LOGGING PERFORMANCE] To improve throughput: (1) disable DEBUG logs in production hot paths (skip decisions and internal diagnostics are DEBUG), "
        "(2) increase PROGRESS_EVERY_RECORDS=%s to reduce heartbeat volume, "
        "(3) keep per-record INFO logs out of loops (this step uses periodic INFO only).",
        PROGRESS_EVERY_RECORDS,
    )

    return total_vectors

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned-dir", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    load_dotenv(".env")

    cfg = load_config(Path(args.config))
    embedding_cfg = cfg["embedding"]
    embedding_cfg["auth_token"] = os.environ["HF_TOKEN"]

    index = init_pinecone_index(
        api_key=os.environ["PINECONE_API_KEY"],
        index_name=embedding_cfg["vector_db"]["index_name"],
    )

    logger.info("INDEX STATS BEFORE: %s", index.describe_index_stats())

    total = process_all_cleaned_emails(
        cleaned_dir=Path(args.cleaned_dir),
        embedding_cfg=embedding_cfg,
        pinecone_index=index,
        #registry_path=Path("data/state/processing_registry.jsonl"),
        registry_path=Path("data/state/processing_registry_step4_rag.jsonl")
    )

    logger.info("Indexed %s vectors", total)
    logger.info("INDEX STATS AFTER: %s", index.describe_index_stats())

if __name__ == "__main__":
    while True:
        main()
        time.sleep(300)
        logger.info("RAG Index: sleeping for 5 minutes")
