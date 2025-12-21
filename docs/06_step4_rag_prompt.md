# STEP 4 — Embeddings + Vector Database (RAG Index Construction)

## 1. Purpose of This Step

This step builds the **semantic retrieval layer** of the system.

Its job is to:

* Convert cleaned vendor emails into embeddings
* Store them in a vector database
* Enable **retrieval‑augmented generation (RAG)** for downstream LLM reasoning

By this point:

* Emails are already relevant (Step 2B)
* Emails are already cleaned and reduced (Step 3)

This step ensures that all future LLM calls are **grounded in actual email evidence**, not hallucination.

---

## 2. Position in the Pipeline

```
STEP 3 — Cleanup & Reduction
   ↓
STEP 4 — Embeddings + Vector DB (RAG Index)
   ↓
Semantic Retrieval Layer
```

Only cleaned vendor‑candidate emails are indexed.

---

## 3. Core Design Principles

* Vector DB is **not** the source of truth
* JSON remains authoritative
* Vector DB can be rebuilt at any time
* Embeddings are **always remote** (HTTP API)
* Incremental upsert is mandatory

---

## 4. Inputs

* Cleaned email JSON files from:

  ```
  data/emails_cleaned/
  ```
* Processing registry

---

## 5. Outputs

### 5.1 Vector Database Index

Depending on configuration:

* Local FAISS index (persisted to disk), OR
* Remote vector DB (e.g., Qdrant)

### 5.2 Chunk Mapping Store

A local mapping that associates vector IDs with source data:

* email_id
* chunk_index
* metadata

This mapping is required to:

* trace retrieval results back to JSON files
* support deletions and re‑indexing

### 5.3 Registry Update

Update registry with:

* last_completed_step = "step4_rag_index"
* embedding_model_version

---

## 6. Chunking Strategy (Critical)

### 6.1 Why Chunking Matters

* Emails may still be too large for embeddings
* Fine‑grained chunks improve retrieval precision
* Chunk boundaries affect RAG quality

---

### 6.2 Chunking Rules

Each email should produce **1 to 3 chunks maximum**.

Recommended chunks:

1. Main cleaned body text
2. Signature block (if non‑empty)
3. Optional short subject/context chunk

Avoid over‑chunking.

---

### 6.3 Chunk Size Guidelines

* Target: 200–500 tokens per chunk
* Hard max enforced via truncation
* Preserve semantic coherence

---

## 7. Embedding API Usage

### 7.1 Embedding Model Assumptions

* Sentence or document embeddings
* Model examples:

  * e5‑large
  * bge‑large
  * MiniLM

Exact model is abstracted behind an API.

---

### 7.2 Embedding Request (Example)

```json
{
  "texts": ["<chunk text 1>", "<chunk text 2>"],
  "model_version": "embed_v1"
}
```

---

### 7.3 Embedding Response (Normalized)

```json
{
  "embeddings": [[0.01, 0.23, ...], [0.12, 0.44, ...]]
}
```

The script must normalize provider‑specific formats.

---

## 8. Vector DB Schema

Each vector record must store:

### 8.1 Vector

* Embedding vector (float array)

### 8.2 Metadata

```
metadata:
  email_id
  chunk_index
  subject
  sender_domain
  email_date
  bert_probability
```

Metadata is critical for filtering and explainability.

---

## 9. Incremental Upsert Logic

For each cleaned email:

1. Check registry
2. Skip if:

   * last_completed_step >= step4_rag_index
   * content_hash unchanged
   * embedding_model_version unchanged
3. Re‑embed and upsert if:

   * content changed, OR
   * embedding model version changed

---

## 10. Deletion & Rebuild Strategy

* If a cleaned email JSON is removed:

  * delete its vectors from the index
* Full rebuild is allowed but should not be required

Vector DB must remain **eventually consistent** with JSON store.

---

## 11. Error Handling

* Embedding API failures must be retried
* Retry limits configurable
* On persistent failure:

  * log error
  * mark registry
  * continue processing others

---

## 12. Configuration Requirements

This step must read from config:

* embedding.endpoint
* embedding.auth_token
* embedding.model_version
* embedding.timeout_seconds
* embedding.max_retries
* vector_db.type (faiss / qdrant)
* vector_db.path or endpoint

---

## 13. Deliverables

Cursor must generate:

* `04_build_rag_index.py`
* Chunking logic
* Embedding API client
* Vector DB abstraction layer
* Incremental registry integration
* CLI interface

---

## 14. Validation Checklist

Before proceeding to Step 5:

* Vectors exist only for cleaned vendor emails
* Chunk sizes reasonable
* Metadata present and accurate
* Re‑running script skips unchanged emails
* Changing embedding model version triggers re‑indexing

This step defines the quality of all RAG‑based reasoning.
