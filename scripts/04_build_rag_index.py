from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict

from functools import wraps
from time import perf_counter

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

@debug_step(
    "embed_texts"
)
def embed_texts(
    texts: List[str],
    *,
    endpoint: str,
    auth_token: str,
    timeout_seconds: int,
    max_retries: int,
) -> List[List[float]]:
    """
    Call remote embedding API and return normalized embeddings.
    Expects provider to return a list of vectors (or equivalent).
    """
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    payload = {"inputs": texts}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print("CALLING EMBEDDING API WITH", len(texts), "TEXTS")

            resp = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()

            # data can be:
            # 1) List[List[float]]              -> already pooled
            # 2) List[List[List[float]]]        -> token embeddings

            vectors = []

            for item in data:
                # Case 1: already pooled vector
                if isinstance(item, list) and item and isinstance(item[0], (int, float)):
                    vectors.append(item)
                    continue

                # Case 2: token embeddings -> mean pool
                if isinstance(item, list) and item and isinstance(item[0], list):
                    dim = len(item[0])
                    pooled = [0.0] * dim
                    for token_vec in item:
                        for i, v in enumerate(token_vec):
                            pooled[i] += v
                    pooled = [v / len(item) for v in pooled]
                    vectors.append(pooled)
                    continue

                raise ValueError(f"Unsupported embedding item format: {type(item)}")

            return vectors




            raise ValueError(f"Unexpected embedding response format: {type(data)}")

        except Exception as e:
            print("🔥 EMBED ERROR:", repr(e))
            print("🔥 RAW RESPONSE TYPE:", type(data) if 'data' in locals() else None)
            print("🔥 RAW RESPONSE VALUE:", data if 'data' in locals() else None)
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

@debug_step("embed_email_chunks")
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
    if not embedding_cfg.get("auth_token"):
        raise RuntimeError("embedding.auth_token is missing")

    chunks = chunk_email(cleaned_email)
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]

    vectors = embed_texts(
        texts,
        endpoint=embedding_cfg["endpoint"],
        auth_token=os.environ["HF_TOKEN"],
        timeout_seconds=embedding_cfg["timeout_seconds"],
        max_retries=embedding_cfg["max_retries"],
    )
    print(
        "EMBED DEBUG:",
        "chunks =", len(chunks),
        "raw_vector_type =", type(vectors[0]),
        "raw_vector_len =", len(vectors[0]),
    )


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

@debug_step("upsert_vectors")
def upsert_vectors(
    *,
    index,
    records: List[Dict],
):
    """
    Upsert vector records into Pinecone.
    """
    if not records:
        return

    print(
        "UPSERT DEBUG:",
        "num_records =", len(records),
        "vector_dim =", len(records[0]["values"]),
    )

    index.upsert(vectors=records, namespace="emails")


@debug_step(
    "process_single_cleaned_email"
)
def process_single_cleaned_email(
    *,
    cleaned_email: dict,
    embedding_cfg: dict,
    pinecone_index,
    base_metadata: dict,
):
    """
    End-to-end processing for a single cleaned email:
    chunk -> embed -> build vectors -> upsert.
    """
    email_id = cleaned_email["email_id"]

    embedded_chunks = embed_email_chunks(
        cleaned_email,
        embedding_cfg=embedding_cfg,
    )
    print("CHUNKS:", len(embedded_chunks))

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
    registry = load_registry(registry_path)
    total_vectors = 0

    for path, cleaned_email in iter_cleaned_emails(cleaned_dir):
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

        if reg:
            delete_email_vectors(
                index=pinecone_index,
                email_id=email_id,
            )

        base_metadata = derive_base_metadata(cleaned_email)

        try:
            n = process_single_cleaned_email(
                cleaned_email=cleaned_email,
                embedding_cfg=embedding_cfg,
                pinecone_index=pinecone_index,
                base_metadata=base_metadata,
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
            total_vectors += n

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

    return total_vectors


from pinecone.core.openapi.shared.exceptions import NotFoundException


def delete_email_vectors(
    *,
    index,
    email_id: str,
):
    """
    Delete all vectors for a given email_id.
    Safe to call even if nothing exists.
    """
    try:
        index.delete(filter={"email_id": email_id}, namespace="emails")
    except NotFoundException:
        # namespace or vectors do not exist yet (first run)
        pass


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
    main()

