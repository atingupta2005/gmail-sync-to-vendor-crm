# Overall Plan — AI‑First Vendor Intelligence System (Maildir / mbsync)

## 1. Purpose

This document is the **authoritative blueprint** for the entire project. It captures *all requirements, constraints, design decisions, and rationale* so implementation (manual scripts first, Cursor-generated code later) stays aligned without rethinking fundamentals.

This file is **conceptual and architectural**. It is **not** a code-generation prompt.

---

## 2. Problem Definition

You have a large personal/work mailbox synced locally via **mbsync** in **Maildir** format. The mailbox contains **200,000+ emails** accumulated over time.

Only about **~1%** of these emails are relevant and represent **vendor interactions**, such as:

* invoices and payments
* purchase orders
* training/course delivery
* proposals, quotations, contracts
* professional service engagements

The goal is to **automatically discover, extract, deduplicate, and maintain** a continuously updated **vendor database**, where **one vendor = one person**, with strong evidence and confidence scoring.

---

## 3. Non‑Negotiable Constraints

### 3.1 Data & Source Constraints

* Email source is **local Maildir**, synced by **mbsync**.
* No Gmail API, IMAP, or cloud mailbox dependency.
* Only `cur/` and `new/` directories are scanned; `tmp/` is ignored.
* Files on disk are the **source of truth**.

### 3.2 Scale Constraints

* Must handle **200k+ emails** efficiently.
* Must assume **continuous growth** (daily incremental syncs).
* Must avoid full rescans after initial ingestion.

### 3.3 Development Constraints

* Development starts with **manual Python scripts**.
* Scripts are run manually, one step at a time.
* **No Docker dependency initially**.
* Code must be readable, debuggable, and inspectable.
* Later refactor into apps/containers must **not require logic rewrites**.

### 3.4 AI Constraints (Explicit Choice)

* The system is **AI‑first**.
* AI is **not optional** for core steps.
* The following are mandatory:

  * BERT‑style classifier for relevance filtering
  * embeddings + vector database for RAG
  * LLM for extraction, reasoning, and deduplication
* All AI models are:

  * hosted remotely (HuggingFace API or self‑hosted)
  * accessed via **HTTP APIs**
* The pipeline machine runs **no heavy ML locally**.

---

## 4. Core Design Principles

### 4.1 Persist First, Decide Later

Every email is **persisted as JSON before any filtering**.

Why:

* enables auditing and debugging
* allows changing models or rules later
* prevents accidental data loss
* guarantees reproducibility

### 4.2 Incremental by Default

Every step:

* processes **only new or changed inputs**
* records progress in a registry
* is safe to re‑run
* can resume after failure

### 4.3 Aggressive Early Reduction

Because only ~1% of emails are relevant:

* filtering must happen early
* expensive AI must never run on obvious junk

### 4.4 AI as Core Intelligence

AI is not an add‑on. It is the **primary reasoning layer**:

* BERT → relevance decision
* Vector DB → semantic memory
* LLM → extraction, reasoning, deduplication

### 4.5 JSON as Source of Truth

* One email = one JSON file
* Vendor records are JSON
* Vector DB is an **index**, never authoritative
* All AI outputs are persisted

---

## 5. Canonical Email Identity & Incrementality

### 5.1 Email Identity (`email_id`)

Each email must have a **stable identity** independent of file paths.

Priority order:

1. If `Message‑ID` exists:

   * `email_id = sha1(Message‑ID)` (optionally salted with From/Date)
2. Otherwise:

   * `email_id = sha1(normalized headers + normalized body)`

### 5.2 Content Hash (`content_hash`)

A separate hash tracks content changes:

* computed from raw email text or normalized bytes
* used to detect modifications

### 5.3 Why Two IDs

* `email_id` → identity & evidence linking
* `content_hash` → change detection & reprocessing

---

## 6. Processing Registry (Backbone)

A persistent **processing registry** records progress per email.

Purpose:

* enable incremental runs
* skip already processed work
* resume after crashes
* track which model versions were used

Each registry entry records:

* email_id
* content_hash
* last_completed_step
* timestamps
* model versions (bert, embedding, llm)
* error status (if any)

Registry format:

* append‑only JSONL file during script phase
* optionally MongoDB later

---

## 7. High‑Level Architecture

### 7.1 Logical Layers

1. Ingestion Layer
2. Filtering Layer
3. Reduction Layer
4. Semantic Index Layer
5. Reasoning & Extraction Layer
6. Deduplication Layer
7. Persistence Layer

Each layer:

* has one clear responsibility
* produces artifacts on disk
* can be run independently

---

## 8. Final Pipeline (Authoritative)

```
STEP 1
Ingest Maildir → Raw Email JSON (ALL emails)

STEP 2A
Heuristic Pre‑Filter (cheap, high recall)

STEP 2B
BERT Classification (core relevance decision)

STEP 3
Cleanup & Reduction (vendor candidates only)

STEP 4
Embeddings + Vector Database (RAG index)

STEP 5
LLM + RAG Extraction & Reasoning

STEP 6
AI‑Assisted Vendor Deduplication

STEP 7
Vendor Database Persistence
```

This order is **fixed** and intentional.

---

## 9. Rationale for Each Step

### STEP 1 — Ingest & Persist

* zero intelligence
* zero filtering
* maximum safety

### STEP 2A — Heuristic Pre‑Filter

* remove obvious junk cheaply
* reduce AI cost
* improve BERT precision

### STEP 2B — BERT Classification

* primary relevance decision
* robust to language variation
* scalable and repeatable

### STEP 3 — Cleanup & Reduction

* reduce token size
* improve embedding quality
* improve LLM reasoning

### STEP 4 — Vector DB / RAG

* semantic memory of relevant emails
* enables retrieval‑based reasoning

### STEP 5 — LLM Extraction & Reasoning

* structured vendor intelligence
* context‑aware decisions

### STEP 6 — Deduplication

* vendors appear many times
* semantic merging required

### STEP 7 — Persistence

* long‑term vendor intelligence store
* incremental updates

---

## 10. Vendor Conceptual Model

* One vendor = one **person**
* Primary key: person email (when available)
* Vendor record includes:

  * person name
  * email
  * phone numbers
  * organization (if inferred)
  * domain
  * categories (invoice, payment, training, etc.)
  * confidence score
  * evidence list (append‑only)
  * first_seen / last_seen

---

## 11. AI Infrastructure Model

### 11.1 Pipeline Machine

* runs Python scripts
* manages JSON files
* calls AI services via HTTP
* no GPU required

### 11.2 AI Machines

* BERT inference service
* embedding service
* LLM service

These may be HuggingFace APIs or self‑hosted services and can be replaced without changing pipeline logic.

---

## 12. Storage Model Summary

* Raw emails: JSON files (authoritative)
* Candidate/cleaned emails: JSON files
* Vector DB: semantic index only
* Vendors: JSON (and optional MongoDB)

---

## 13. Refactor Strategy (Later)

* Keep logic modular inside scripts.
* Wrap scripts as CLIs.
* Group CLIs into containers as desired.
* No logic rewrite—only packaging.

---

## 14. How This File Is Used

* Human reference
* Architectural guardrail
* Prevents scope creep
* Basis for all Cursor prompts

This file should be read once before implementation and revisited only when requirements change.
