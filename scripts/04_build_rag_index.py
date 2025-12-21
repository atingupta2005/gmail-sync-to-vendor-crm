from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict
import os

from functools import wraps
from time import perf_counter

def log_progress(processed, total_vectors, every=50):
    if processed % every == 0:
        print(
            f"[PROGRESS] emails={processed} "
            f"vectors={total_vectors} "
            f"time={datetime.utcnow().isoformat()}Z"
        )

def debug_step(name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            print(f"[DEBUG] → {name}.start")
            t0 = perf_counter()
            result = fn(*args, **kwargs)
            dt = perf_counter() - t0

            if isinstance(result, list):
                size = len(result)
            elif result is None:
                size = None
            else:
                size = result

            print(f"[DEBUG] ← {name}.end | result={size} | {dt:.2f}s")
            return result
        return wrapper
    return decorator


def chunk_email(cleaned_email: dict) -> List[Dict]:
    """
    Deterministically split a cleaned email into <= 3 chunks.
    Returns a list of {"chunk_index": int, "text": str}
    """
    chunks = []

    # --- Chunk 0: main body (always) ---
    text = (cleaned_email.get("cleaned_text") or "").strip()
    if text:
        max_chars = 1800
        body = text[:max_chars]
        # try to cut on sentence boundary
        last_period = body.rfind(".")
        if last_period > 200:
            body = body[: last_period + 1]
        chunks.append({"chunk_index": 0, "text": body})

    # --- Chunk 1: signature (optional, conservative) ---
    # heuristic: last ~8 lines, only if it looks like a signature
    lines = text.splitlines()
    if len(lines) >= 4:
        tail = "\n".join(lines[-8:]).strip()
        signal = any(k in tail.lower() for k in ["@", "tel", "phone", "mobile", "www", "http"])
        if signal and len(tail) <= 300:
            chunks.append({"chunk_index": 1, "text": tail})

    # --- Chunk 2: subject/context (optional) ---
    subject = (cleaned_email.get("subject") or "").strip()
    if subject:
        chunks.append({"chunk_index": 2, "text": subject[:150]})

    return chunks[:3]


import time
import requests

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
            resp = _SESSION.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()

            vectors = []
            for item in data:
                if isinstance(item[0], (int, float)):
                    vectors.append(item)
                else:
                    dim = len(item[0])
                    pooled = [0.0] * dim
                    for token_vec in item:
                        for i, v in enumerate(token_vec):
                            pooled[i] += v
                    vectors.append([v / len(item) for v in pooled])

            return vectors

        except Exception:
            if attempt == max_retries:
                raise


from pinecone import Pinecone


def init_pinecone_index(
    *,
    api_key: str,
    index_name: str,
):
    """
    Initialize Pinecone client and return an Index handle.
    Assumes index already exists.
    """
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def embed_email_chunks(
    cleaned_email: dict,
    *,
    embedding_cfg: dict,
) -> List[Dict]:
    """
    Chunk a cleaned email and embed each chunk.
    Returns list of dicts:
      {
        "chunk_index": int,
        "text": str,
        "embedding": List[float],
      }
    """

    chunks = chunk_email(cleaned_email)
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]

    vectors = embed_texts(
        texts,
        endpoint=embedding_cfg["endpoint"],
        auth_token=embedding_cfg["auth_token"],
        timeout_seconds=embedding_cfg["timeout_seconds"],
        max_retries=embedding_cfg["max_retries"],
    )

    if not vectors:
        raise ValueError("Empty embedding response")
    
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match chunk count")

    results = []
    for c, v in zip(chunks, vectors):
        results.append(
            {
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "embedding": v,
            }
        )

    return results


def build_vector_records(
    *,
    email_id: str,
    embedded_chunks: List[Dict],
    metadata: dict,
) -> List[Dict]:
    """
    Build Pinecone-ready vector records.
    """
    records = []

    for c in embedded_chunks:
        vector_id = f"{email_id}::chunk_{c['chunk_index']}"

        record = {
            "id": vector_id,
            "values": c["embedding"],
            "metadata": {
                **metadata,
                "email_id": email_id,
                "chunk_index": c["chunk_index"],
            },
        }
        records.append(record)

    return records

def upsert_vectors(
    *,
    index,
    records: List[Dict],
    buffer: List[Dict],
    batch_size: int = 500,
):
    """
    Buffer vectors and upsert in large batches.
    """
    if not records:
        return

    buffer.extend(records)

    if len(buffer) >= batch_size:
        print("UPSERT FLUSH:", len(buffer))
        index.upsert(vectors=buffer, namespace="emails")
        buffer.clear()

def process_single_cleaned_email(
    *,
    cleaned_email: dict,
    embedding_cfg: dict,
    pinecone_index,
    base_metadata: dict,
    upsert_buffer: List[Dict],
):
    """
    End-to-end processing for a single cleaned email:
    chunk -> embed -> build vectors -> buffered upsert.
    """
    email_id = cleaned_email["email_id"]

    embedded_chunks = embed_email_chunks(
        cleaned_email,
        embedding_cfg=embedding_cfg,
    )

    if not embedded_chunks:
        return 0

    records = build_vector_records(
        email_id=email_id,
        embedded_chunks=embedded_chunks,
        metadata=base_metadata,
    )

    upsert_vectors(
        index=pinecone_index,
        records=records,
        buffer=upsert_buffer,
    )

    return len(records)



import json
from pathlib import Path


def load_cleaned_email(path: Path) -> dict:
    """
    Load a single cleaned email JSON file from disk.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def derive_base_metadata(cleaned_email: dict) -> dict:
    """
    Extract metadata to attach to every vector for this email.
    """
    meta = {}

    # subject (if present)
    subject = cleaned_email.get("subject")
    if subject:
        meta["subject"] = subject

    # sender domain (if present upstream)
    sender = cleaned_email.get("from") or cleaned_email.get("sender")
    if sender and "@" in sender:
        meta["sender_domain"] = sender.split("@")[-1].lower()

    # email date (best-effort)
    email_date = cleaned_email.get("date") or cleaned_email.get("sent_at")
    if email_date:
        meta["email_date"] = email_date

    # BERT probability (from Step 2B, if carried forward)
    bert_prob = cleaned_email.get("bert_probability")
    if isinstance(bert_prob, (int, float)):
        meta["bert_probability"] = float(bert_prob)

    return meta


def iter_cleaned_emails(cleaned_dir: Path):
    """
    Yield (path, cleaned_email_dict) for each cleaned email JSON.
    """
    for path in cleaned_dir.rglob("*.json"):
        try:
            yield path, load_cleaned_email(path)
        except Exception as e:
            # swallow for now; proper error handling comes later
            continue

def load_registry(path: Path) -> dict:
    registry = {}
    if not path.exists():
        return registry
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                registry[rec["email_id"]] = rec
            except Exception:
                continue
    return registry


def append_registry(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_all_cleaned_emails(
    *,
    cleaned_dir: Path,
    embedding_cfg: dict,
    pinecone_index,
    registry_path: Path,
):
    upsert_buffer = []
    processed_emails = 0

    registry = load_registry(registry_path)
    total_vectors = 0

    for path, cleaned_email in iter_cleaned_emails(cleaned_dir):
        if processed_emails % 100 == 0:
            print("FOUND CLEANED EMAIL:", path)

        email_id = cleaned_email["email_id"]
        content_hash = cleaned_email.get("content_hash")
        model_version = embedding_cfg["model_version"]

        reg = registry.get(email_id)
        if (
            reg
            and reg.get("last_completed_step") == "step4_rag_index"
            and reg.get("content_hash") == content_hash
            and reg.get("embedding_model_version") == model_version
        ):
            continue  # safe skip

        # No delete needed — upsert overwrites by ID

        base_metadata = derive_base_metadata(cleaned_email)

        try:
            n = process_single_cleaned_email(
                cleaned_email=cleaned_email,
                embedding_cfg=embedding_cfg,
                pinecone_index=pinecone_index,
                base_metadata=base_metadata,
                upsert_buffer=upsert_buffer,
            )

            append_registry(
                registry_path,
                {
                    "email_id": email_id,
                    "content_hash": content_hash,
                    "last_completed_step": "step4_rag_index",
                    "embedding_model_version": model_version,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            processed_emails += 1
            total_vectors += n

            log_progress(processed_emails, total_vectors)

        except Exception as e:
            append_registry(
                registry_path,
                {
                    "email_id": email_id,
                    "last_completed_step": "step4_rag_index_error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            continue

    if upsert_buffer:
        print("FINAL UPSERT FLUSH:", len(upsert_buffer))
        pinecone_index.upsert(vectors=upsert_buffer, namespace="emails")
        upsert_buffer.clear()

    return total_vectors


from pinecone.core.openapi.shared.exceptions import NotFoundException


def delete_email_vectors(
    *,
    index,
    email_id: str,
):
    """
    No-op.
    Deleting before upsert is unnecessary because
    upsert with deterministic IDs overwrites existing vectors.
    """
    return


import argparse
import os
import yaml

import os
import re

_ENV_RE = re.compile(r"\$\{([^}]+)\}")

def expand_env_vars(obj):
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        def repl(m):
            return os.environ.get(m.group(1), m.group(0))
        return _ENV_RE.sub(repl, obj)
    return obj


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned-dir", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    load_dotenv(dotenv_path=".env")

    cfg = expand_env_vars(load_config(Path(args.config)))


    embedding_cfg = dict(cfg["embedding"])
    embedding_cfg["auth_token"] = os.environ["HF_TOKEN"]
    
    pinecone_cfg = embedding_cfg["vector_db"]

    index = init_pinecone_index(
        api_key=os.environ["PINECONE_API_KEY"],
        index_name=pinecone_cfg["index_name"],
    )

    print("Before processing: INDEX STATS:")
    print("--------------------------------")
    print(index.describe_index_stats())
    print("--------------------------------")

    total = process_all_cleaned_emails(
        cleaned_dir=Path(args.cleaned_dir),
        embedding_cfg=embedding_cfg,
        pinecone_index=index,
        registry_path=Path("data/state/processing_registry.jsonl"),
    )

    print(f"Indexed {total} vectors")
    
    print("After processing: INDEX STATS:")
    print("--------------------------------")
    print(index.describe_index_stats())
    print("--------------------------------")


if __name__ == "__main__":
    while True:
        main()
        time.sleep(300)
        print("RAG Index: sleeping for 5 minutes")

