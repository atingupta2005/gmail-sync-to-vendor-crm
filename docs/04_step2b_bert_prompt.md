# STEP 2B — BERT‑Based Vendor Classification (Core Relevance Decision)

## 1. Purpose of This Step

This step performs the **primary relevance decision**: determining whether an email is truly **vendor‑related**.

Unlike Step 2A (heuristic, high‑recall), this step is **precision‑oriented** and uses a **BERT‑style transformer classifier**.

This step is where the email set is reduced from roughly **10–20% → ~1%**.

---

## 2. Position in the Pipeline

```
Raw Email JSON (ALL emails)
   ↓
STEP 2A — Heuristic Pre‑Filter (~10–20%)
   ↓
STEP 2B — BERT Classification (~1%)
   ↓
Vendor Candidate Emails
```

Only emails that pass this step are considered vendor candidates.

---

## 3. Core Design Principles

* BERT is the **authoritative relevance judge**
* Classification is **binary**: vendor vs non‑vendor
* Model inference is **remote** via HTTP
* Pipeline machine runs no ML locally
* All decisions are persisted and explainable

---

## 4. Inputs

* Prefiltered email JSON files from:

  ```
  data/emails_prefiltered/
  ```
* Processing registry

---

## 5. Outputs

### 5.1 Vendor Candidate Set

Emails classified as vendor‑related are copied or linked to:

```
data/emails_candidates/
  ├── 00/<email_id>.json
  ├── 01/<email_id>.json
```

### 5.2 Classification Log

Append‑only JSONL file:

```
data/state/step2b_bert_classification.jsonl
```

Each record includes:

* email_id
* vendor_probability
* predicted_label
* threshold_used
* model_version
* timestamp

### 5.3 Registry Update

Update registry with:

* last_completed_step = "step2b_bert"
* bert_model_version

---

## 6. BERT Model Assumptions

### 6.1 Model Type

* Transformer encoder (BERT‑family)
* Binary classification head
* Examples:

  * bert‑base‑uncased
  * distilbert‑base‑uncased
  * MiniLM

The exact model is abstracted behind an API.

---

### 6.2 Training Assumptions (Conceptual)

While training is out of scope for this script, the inference logic must assume:

* Strong class imbalance (~1% positive)
* Model trained with:

  * class weights or focal loss
  * balanced validation set

This affects **threshold tuning**.

---

## 7. Input Text Construction (Critical)

Raw emails are too long and noisy for BERT.

The classifier input must be **carefully constructed**.

### 7.1 Text Components

The input text must include:

1. Subject line
2. Sender domain
3. First N characters or tokens of body
4. Signature block (if detected)

Example layout:

```
[SUBJECT]
Invoice for Training Program

[FROM_DOMAIN]
abc‑training.com

[BODY]
First 300–500 tokens of body text

[SIGNATURE]
John Doe
ABC Training
+91‑XXXXXXXX
```

---

### 7.2 Token Length Safety

* Hard limit enforced (e.g., 512 tokens)
* Truncate body safely
* Preserve subject and signature preferentially

---

## 8. Remote Inference API Contract

The script must support calling a remote inference endpoint.

### 8.1 Request (Example)

```json
{
  "text": "<constructed input text>",
  "model_version": "bert_vendor_v1"
}
```

### 8.2 Response (Normalized)

The script must normalize different provider formats into:

```json
{
  "vendor_probability": 0.93
}
```

---

## 9. Thresholding Strategy

Classification decision:

```
if vendor_probability >= THRESHOLD:
    vendor = true
else:
    vendor = false
```

### Threshold Characteristics

* Threshold must be configurable
* Typical range: 0.6 – 0.8
* Lower threshold → higher recall
* Higher threshold → higher precision

Threshold tuning happens offline and should not require code changes.

---

## 10. Incremental Processing Rules

For each email:

1. Check registry entry
2. Skip if:

   * last_completed_step >= step2b_bert
   * content_hash unchanged
   * bert_model_version unchanged
3. Re‑run if:

   * content_hash changed, OR
   * bert_model_version changed

---

## 11. Error Handling

* API timeouts must be retried
* Retry count configurable
* If retries exhausted:

  * log error
  * mark registry with failure
  * continue processing other emails

No single failure must stop the batch.

---

## 12. Configuration Requirements

This step must read from config:

* bert.endpoint
* bert.auth_token (if required)
* bert.model_version
* bert.timeout_seconds
* bert.max_retries
* bert.threshold

No hard‑coded values.

---

## 13. Deliverables

Cursor must generate:

* `02b_bert_vendor_classifier.py`
* Remote inference client
* Input text builder
* Threshold logic
* Incremental registry integration
* CLI interface

---

## 14. Validation Checklist

Before proceeding to Step 3:

* Vendor candidate rate ~1%
* False negatives are rare
* Probabilities look sensible
* Re‑running script skips unchanged emails
* Changing model_version triggers reprocessing

This step defines relevance quality. Validate carefully.
