from dotenv import load_dotenv
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
# Utility logging helpers
# -------------------------------------------------------------------

def log_progress(processed: int, total_vectors: int, every: int = 50) -> None:
    if processed % every == 0:
        logger.info(
            "[PROGRESS] emails=%s vectors=%s time=%sZ",
            processed,
            total_vectors,
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

    upsert_buffer: List[Dict] = []
    registry = load_registry(registry_path)
    processed_emails = 0
    total_vectors = 0

    for path, cleaned_email in iter_cleaned_emails(cleaned_dir):
        if processed_emails % 100 == 0:
            logger.info("FOUND CLEANED EMAIL: %s", path)

        email_id = cleaned_email["email_id"]
        content_hash = cleaned_email.get("content_hash")
        model_version = embedding_cfg["model_version"]

        reg = registry.get(email_id)
        if reg and reg.get("content_hash") == content_hash and reg.get("embedding_model_version") == model_version:
            continue

        try:
            n = process_single_cleaned_email(
                cleaned_email=cleaned_email,
                embedding_cfg=embedding_cfg,
                pinecone_index=pinecone_index,
                base_metadata=derive_base_metadata(cleaned_email),
                upsert_buffer=upsert_buffer,
            )

            append_registry(
                registry_path,
                {
                    "email_id": email_id,
                    "content_hash": content_hash,
                    "embedding_model_version": model_version,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            processed_emails += 1
            total_vectors += n
            log_progress(processed_emails, total_vectors)

        except Exception as e:
            logger.exception("Failed email %s", email_id)

    if upsert_buffer:
        logger.info("FINAL UPSERT FLUSH: %s", len(upsert_buffer))
        pinecone_index.upsert(vectors=upsert_buffer, namespace="emails")

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
        registry_path=Path("data/state/processing_registry.jsonl"),
    )

    logger.info("Indexed %s vectors", total)
    logger.info("INDEX STATS AFTER: %s", index.describe_index_stats())

if __name__ == "__main__":
    while True:
        main()
        time.sleep(300)
        logger.info("RAG Index: sleeping for 5 minutes")
