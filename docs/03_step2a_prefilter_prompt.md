# STEP 2A — Heuristic Pre‑Filter (Cheap, High‑Recall Pruning)

## 1. Purpose of This Step

This step performs **cheap, deterministic pruning** to eliminate obvious non‑vendor emails **before** expensive AI models (BERT, embeddings, LLM) are invoked.

Its goal is **high recall**, not perfect precision.

Key idea:

> It is acceptable to keep false positives. It is NOT acceptable to drop true vendor emails.

This step dramatically reduces cost and latency for downstream AI.

---

## 2. Position in the Pipeline

This step runs **after Step 1 ingestion** and **before Step 2B BERT classification**.

```
Raw Email JSON (ALL emails)
   ↓
STEP 2A — Heuristic Pre‑Filter
   ↓
Prefiltered Emails (~10–20%)
```

Only emails that pass this step are sent to BERT.

---

## 3. Inputs

* Raw email JSON files from:

  ```
  data/emails_raw_json/
  ```

* Processing registry from Step 1

---

## 4. Outputs

### 4.1 Prefiltered Email Set

Emails that pass this step are copied or linked to:

```
data/emails_prefiltered/
  ├── 00/<email_id>.json
  ├── 01/<email_id>.json
```

### 4.2 Decision Log

Append‑only JSONL file:

```
data/state/step2a_prefilter_decisions.jsonl
```

Each record must include:

* email_id
* score
* passed (true/false)
* reasons[]
* timestamp

### 4.3 Registry Update

Update processing registry with:

* last_completed_step = "step2a_prefilter"

---

## 5. Core Design Principles

* Deterministic logic only
* No ML, no LLM
* Fully explainable decisions
* Fast execution
* Incremental processing

---

## 6. Scoring‑Based Decision Model

This step must use a **scoring system**, not a simple boolean check.

### Why scoring?

* Allows tuning without code changes
* Preserves explainability
* Avoids brittle yes/no rules

An email passes if:

```
score >= PREFILTER_THRESHOLD
```

Threshold is configurable.

---

## 7. Positive Signals (Add to Score)

### 7.1 Financial & Commercial Keywords

Examples (case‑insensitive):

* invoice
* tax invoice
* quotation
* quote
* proposal
* contract
* agreement
* billing
* bill

Suggested score: +3 each (configurable)

---

### 7.2 Purchase Order Patterns

Examples:

* purchase order
* PO
* P.O.
* PO#
* PO No

Regex examples:

* `\bPO\b`
* `purchase\s+order`

Suggested score: +3

---

### 7.3 Payment & Banking Indicators

Examples:

* payment
* paid
* bank
* account number
* IFSC
* NEFT
* IMPS
* RTGS
* UTR
* SWIFT

Suggested score: +3

---

### 7.4 Training / Course / Delivery Keywords

Examples:

* training
* trainer
* session
* workshop
* course
* delivery
* agenda
* materials

Suggested score: +2

---

### 7.5 Contact Signals

#### Phone Numbers

Regex examples:

* international formats
* country‑specific formats

Suggested score: +2

#### External Email Domains

If sender domain is **not** in a known internal / personal domain list.

Suggested score: +1

---

### 7.6 Attachment Indicators

If `mime_meta.has_attachments == true`.

Additional boost if attachment filename contains:

* invoice
* quote
* proposal
* contract

Suggested score:

* attachment present: +1
* keyworded attachment name: +2

---

## 8. Negative Signals (Subtract from Score)

### 8.1 Automated / System Emails

Examples:

* noreply
* do‑not‑reply
* automated message

Suggested score: −5

---

### 8.2 Newsletters & Marketing

Examples:

* unsubscribe
* newsletter
* promotional
* marketing

Suggested score: −5

---

### 8.3 OTP / Verification Emails

Examples:

* OTP
* verification code
* password reset
* security alert

Suggested score: −5

---

## 9. Subject vs Body Weighting

* Subject matches should carry **higher weight** than body matches.
* Example:

  * keyword in subject: full score
  * keyword in body: 50–70% score

---

## 10. Incremental Processing Rules

For each email:

1. Check registry:

   * if already processed for Step 2A AND content_hash unchanged → skip
2. Otherwise:

   * compute score
   * log decision
   * update registry

---

## 11. Error Handling

* Missing fields must not crash the script
* Any failure must:

  * log an error record
  * mark registry with error
  * continue processing others

---

## 12. Configuration Requirements

This step must read from config:

* keyword lists
* regex patterns
* scoring weights
* threshold value
* known internal domains list

No hard‑coding in logic.

---

## 13. Deliverables

Cursor must generate:

* `02a_prefilter_vendor_emails.py`
* Config‑driven scoring system
* Decision logging
* Incremental registry updates
* CLI interface

---

## 14. Validation Checklist

After implementation:

* ~10–20% of emails pass this step
* Almost no obvious vendor emails are dropped
* Decisions are explainable via reasons[]
* Re‑running script skips unchanged emails

Do NOT proceed to Step 2B until recall is acceptable.
