# STEP 6 — AI‑Assisted Vendor Deduplication (One Vendor = One Person)

## 1. Purpose of This Step

This step merges multiple vendor extraction records into a **single canonical vendor record per person**.

By this point:

* Each relevant email has produced a vendor extraction (Step 5)
* Multiple emails may refer to the same person using:

  * different names
  * different email addresses
  * different signatures

This step ensures **one vendor = one person**, while preserving all evidence.

---

## 2. Position in the Pipeline

```
STEP 5 — LLM + RAG Vendor Extraction
   ↓
STEP 6 — AI‑Assisted Vendor Deduplication
   ↓
Canonical Vendor Records
```

This step operates on **vendor extraction records**, not emails directly.

---

## 3. Core Design Principles

* Deterministic merging first
* AI used only for ambiguous cases
* Evidence is append‑only
* No vendor record is ever deleted
* Incremental processing

---

## 4. Inputs

* Vendor extraction records from:

  ```
  data/vendors_raw/vendor_extractions.jsonl
  ```
* Existing canonical vendor records (if any)
* Processing registry

---

## 5. Outputs

### 5.1 Canonical Vendor Database

Written to:

```
data/vendors_final/vendors.json
```

Each vendor record represents **one person**.

### 5.2 Deduplication Decision Log

Append‑only JSONL file:

```
data/state/step6_dedup_decisions.jsonl
```

Each record includes:

* candidate_vendor_ids
* decision (merge / no‑merge)
* reason
* confidence
* timestamp

### 5.3 Registry Update

Update registry with:

* last_completed_step = "step6_vendor_dedup"

---

## 6. Vendor Identity Model

### 6.1 Primary Identity Key

* **Person email address** (when present)

If two vendor records share the same normalized email address:

* they MUST be merged
* no AI decision required

---

### 6.2 Secondary Identity Signals

Used when primary key is missing or different:

* phone number match
* same domain + similar name
* same organization + similar name

These cases may require AI assistance.

---

## 7. Deterministic Deduplication Rules

The following merges must happen **without AI**:

1. Exact email match (case‑insensitive)
2. Exact phone number match
3. Exact email + domain match after normalization

These merges are high confidence.

---

## 8. Ambiguous Cases (AI‑Assisted)

Examples:

* [john@abc.com](mailto:john@abc.com) vs [john.doe@abc.com](mailto:john.doe@abc.com)
* “John D” vs “John Doe”
* Different email domains but same person name and organization

For these cases, invoke the LLM.

---

## 9. LLM Merge Decision Prompt

The LLM must be asked a **binary question**:

> “Do these two vendor records represent the same person?”

### Prompt Inputs

* Vendor A summary
* Vendor B summary
* Evidence snippets (email subjects, dates)

### Expected Output (Strict JSON)

```
{
  "same_person": true | false,
  "confidence": number,
  "reason": string
}
```

---

## 10. Merge Strategy

When merging Vendor B into Vendor A:

* Preserve Vendor A’s `vendor_id`
* Merge fields:

  * combine phone numbers
  * keep most complete name
  * keep most recent organization
* Append all evidence
* Update `last_seen`

Never discard evidence.

---

## 11. Incremental Deduplication Rules

For each new vendor extraction:

1. Attempt deterministic merge
2. If ambiguous:

   * check if AI decision already exists
   * otherwise call LLM
3. Apply merge or keep separate
4. Update canonical store

Previously deduped vendors must not be re‑processed unnecessarily.

---

## 12. Error Handling

* LLM failures must not block pipeline
* If AI decision fails:

  * log error
  * keep vendors separate
  * mark for later review

---

## 13. Configuration Requirements

This step must read from config:

* llm.endpoint
* llm.auth_token
* llm.model_version
* llm.timeout_seconds
* llm.max_retries
* name_similarity_threshold

---

## 14. Deliverables

Cursor must generate:

* `06_dedupe_vendors.py`
* Deterministic merge logic
* LLM merge decision client
* Deduplication logs
* Incremental registry integration
* CLI interface

---

## 15. Validation Checklist

Before proceeding to Step 7:

* No duplicate vendors for same email
* Evidence preserved across merges
* AI decisions logged
* Re‑running script is idempotent
* Canonical vendor count stabilizes over time

This step determines data correctness. Validate carefully.
