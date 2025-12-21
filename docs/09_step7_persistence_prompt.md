# STEP 7 — Vendor Database Persistence & Incremental Updates

## 1. Purpose of This Step

This step persists the **final, canonical vendor records** produced by Step 6 into durable storage.

It is the **last step of the pipeline** and represents the system’s long‑term memory.

By this point:

* Vendors are already deduplicated
* Evidence is consolidated
* Confidence scores exist

This step must be **boring, reliable, and correct**.

---

## 2. Position in the Pipeline

```
STEP 6 — AI‑Assisted Vendor Deduplication
   ↓
STEP 7 — Vendor Persistence
```

No downstream processing depends on this step.

---

## 3. Core Design Principles

* Persistence is deterministic
* No AI is used here
* Append‑only evidence
* Incremental updates only
* Storage backends are pluggable

---

## 4. Inputs

* Canonical vendor records from:

  ```
  data/vendors_final/vendors.json
  ```
* Processing registry

---

## 5. Outputs

### 5.1 Primary Output (Authoritative)

* JSON file containing all canonical vendors:

  ```
  data/vendors_persisted/vendors.json
  ```

This file is the **authoritative vendor database**.

---

### 5.2 Optional Secondary Output (MongoDB)

If enabled via config:

* Vendors are upserted into MongoDB
* JSON file remains authoritative

MongoDB is a convenience layer, not a source of truth.

---

## 6. Canonical Vendor JSON Schema

Each vendor record must conform to the following structure:

```
{
  "vendor_id": string,
  "person_name": string | null,
  "person_email": string | null,
  "phone_numbers": [string],
  "organization": string | null,
  "domain": string | null,
  "categories": ["invoice" | "payment" | "training" | "course" | "contract" | "other"],
  "confidence": number,
  "evidence": [
    {
      "email_id": string,
      "reason": string,
      "timestamp": string
    }
  ],
  "first_seen": string,
  "last_seen": string,
  "last_updated": string
}
```

Rules:

* `vendor_id` must be stable
* `evidence` is append‑only
* `confidence` may be recomputed but not deleted

---

## 7. Vendor ID Strategy

The `vendor_id` must be:

* deterministic
* stable across runs

Preferred strategy:

1. If person_email exists:

   * `vendor_id = sha1(normalized_email)`
2. Else:

   * `vendor_id = sha1(person_name + organization + domain)`

Once assigned, a vendor_id must never change.

---

## 8. Incremental Persistence Rules

For each vendor record:

1. If vendor_id does not exist in persisted store:

   * insert
2. If vendor_id exists:

   * merge updates
   * append new evidence
   * update last_seen and last_updated

Never delete vendors.

---

## 9. Evidence Handling

Evidence is the **audit trail** of why a vendor exists.

Rules:

* Evidence entries are never removed
* Duplicate evidence (same email_id) must not be re‑added
* Evidence must include:

  * email_id
  * reason
  * timestamp

---

## 10. MongoDB Integration (Optional)

### 10.1 Enablement

MongoDB usage must be optional and config‑driven.

### 10.2 Upsert Rules

* Collection indexed by vendor_id
* Use upsert operations
* Mirror JSON state

### 10.3 Failure Handling

* Mongo failure must not corrupt JSON persistence
* JSON write happens first
* Mongo errors are logged only

---

## 11. Processing Registry Update

After successful persistence:

* update registry with last_completed_step = "step7_persistence"
* record timestamp

---

## 12. Error Handling

* File write failures must abort the step
* Partial writes must be avoided (write‑then‑rename strategy)
* Mongo failures must not abort JSON persistence

---

## 13. Configuration Requirements

This step must read from config:

* persistence.output_path
* mongo.enabled
* mongo.uri
* mongo.database
* mongo.collection

---

## 14. Deliverables

Cursor must generate:

* `07_save_vendors.py`
* JSON persistence logic
* Incremental merge logic
* Optional MongoDB client
* Registry update integration
* CLI interface

---

## 15. Validation Checklist

After implementation:

* vendors.json is valid JSON
* vendor_id stable across runs
* evidence accumulates correctly
* re‑running script is idempotent
* MongoDB (if enabled) mirrors JSON

This step completes the pipeline.
