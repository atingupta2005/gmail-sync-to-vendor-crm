# STEP 3 — Cleanup & Reduction of Vendor Candidate Emails

## 1. Purpose of This Step

This step **reduces noise and size** of emails that have already been classified as vendor‑related.

By this point:

* Relevance has already been decided (Step 2B)
* Volume is low (~1% of total emails)

The goal here is to:

* Remove irrelevant text (quoted replies, forwards, boilerplate)
* Extract a clean, human‑authored message
* Extract the signature block
* Prepare text for **embeddings and LLM reasoning**

This step directly impacts **embedding quality, RAG relevance, and LLM accuracy**.

---

## 2. Position in the Pipeline

```
STEP 2B — BERT Classification (vendor candidates)
   ↓
STEP 3 — Cleanup & Reduction
   ↓
Cleaned Vendor Emails
```

Only emails that passed Step 2B are processed here.

---

## 3. Core Design Principles

* Deterministic rules only (no AI)
* Aggressive reduction, but preserve meaning
* Never alter factual content
* Always preserve:

  * subject
  * sender information
  * core message
* Incremental and idempotent

---

## 4. Inputs

* Vendor candidate email JSON files from:

  ```
  data/emails_candidates/
  ```
* Processing registry

---

## 5. Outputs

### 5.1 Cleaned Email JSON Files

Written to:

```
data/emails_cleaned/
  ├── 00/<email_id>.json
  ├── 01/<email_id>.json
```

Each output JSON must include both original and cleaned representations.

### 5.2 Cleanup Statistics Log

Append‑only JSONL file:

```
data/state/step3_cleanup_stats.jsonl
```

Each record must include:

* email_id
* raw_length
* cleaned_length
* reduction_ratio
* timestamp

### 5.3 Registry Update

Update registry with:

* last_completed_step = "step3_cleanup"

---

## 6. Cleanup Operations (In Order)

Cleanup must be applied in a **fixed order** to ensure consistency.

---

## 7. Remove Quoted Replies

Quoted replies add large amounts of redundant text.

### Common patterns to detect

* Lines starting with `>`
* Blocks starting with:

  * `On <date>, <person> wrote:`
  * `From:` / `Sent:` / `To:` / `Subject:` blocks

### Rule

* Once a quoted‑reply marker is detected, drop everything **below** it.

---

## 8. Remove Forwarded Message Blocks

### Common markers

* `----- Forwarded message -----`
* `Begin forwarded message:`

### Rule

* Remove forwarded blocks entirely
* Preserve only the latest authored message

---

## 9. Remove Base64 / Attachment Artifacts

Some emails contain base64 remnants or inline encoded content.

### Detection heuristics

* Very long lines with no spaces
* Base64 character sets
* MIME boundary artifacts

### Rule

* Remove such blocks completely

---

## 10. Remove Boilerplate & Disclaimers

### Examples

* Corporate confidentiality disclaimers
* Virus scan footers
* Long legal notices

### Rule

* Remove known boilerplate patterns via regex list
* Configurable patterns preferred

---

## 11. Normalize Whitespace

After removals:

* Collapse multiple blank lines
* Normalize line endings
* Strip leading/trailing whitespace

---

## 12. Signature Extraction (Critical)

The signature often contains:

* vendor name
* organization
* phone numbers
* email addresses

### Common signature delimiters

* `--`
* `Thanks,`
* `Regards,`
* `Best regards,`
* `Sincerely,`

### Strategy

* Detect signature delimiter from bottom up
* Split message into:

  * main body
  * signature block

If no delimiter found:

* leave signature empty

---

## 13. Length & Token Control

### Why this matters

* Embeddings and LLMs have token limits
* Shorter text = better signal

### Rules

* Enforce max cleaned length (configurable, e.g. 2–4k chars)
* Truncate from bottom if needed
* Preserve subject and top of body preferentially

---

## 14. Output JSON Schema Additions

The cleaned JSON must include:

```
body:
  raw_text
  cleaned_text
  signature_text

cleanup_stats:
  raw_length
  cleaned_length
  reduction_ratio
```

Original raw_text must never be removed.

---

## 15. Incremental Processing Rules

For each email:

1. Check registry
2. Skip if:

   * last_completed_step >= step3_cleanup
   * content_hash unchanged
3. Re‑run if content changed

---

## 16. Error Handling

* Any cleanup failure must be logged
* Email should still be written with:

  * cleaned_text = raw_text
  * cleanup_stats indicating failure
* Processing must continue

---

## 17. Configuration Requirements

This step must read from config:

* quoted‑reply patterns
* forwarded message patterns
* boilerplate patterns
* signature delimiters
* max_cleaned_length

---

## 18. Deliverables

Cursor must generate:

* `03_cleanup_candidate_emails.py`
* Deterministic cleanup pipeline
* Incremental registry integration
* CLI interface

---

## 19. Validation Checklist

Before proceeding to Step 4:

* Cleaned text significantly shorter than raw
* No quoted chains remain
* Signature extracted when present
* Re‑running script skips unchanged emails
* Cleanup stats look reasonable

This step determines downstream AI quality. Validate carefully.
