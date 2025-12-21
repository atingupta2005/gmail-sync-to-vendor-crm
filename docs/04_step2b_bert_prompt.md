## 1. Purpose of This Step (Revised)

This step produces a **vendor relevance score** for each email that passed STEP 2A.

At system start, **no manually labeled training data exists**.
Therefore, this step **does NOT require a pre-trained vendor classifier**.

Instead, it:

* uses a **bootstrappable AI model** (e.g. zero-shot or weakly supervised)
* produces a **vendor_probability ∈ [0,1]**
* treats predictions as **proposals**, not ground truth
* enables **human validation**
* accumulates labeled data for **future incremental training**

This step remains **mandatory, AI-based, and authoritative for routing**, but **not authoritative for truth**.

---

## 2. Position in the Pipeline (Unchanged)

```
Raw Email JSON (ALL emails)
   ↓
STEP 2A — Heuristic Pre-Filter (~10–20%)
   ↓
STEP 2B — Vendor Relevance Scoring (~1–5%)
   ↓
Vendor Candidate Emails
```

Only emails that pass the configurable threshold continue downstream.

---

## 3. Core Design Principles (Revised)

* AI-first, even without labeled data
* No manual class creation required
* Human-in-the-loop by design
* Fully incremental and re-runnable
* Model-agnostic via HTTP inference

---

## 4. Inputs

* Prefiltered email JSON files:

  ```
  data/emails_prefiltered/
  ```
* Processing registry

---

## 5. Outputs

### 5.1 Vendor Candidate Email Set

Emails with `vendor_probability >= threshold` are copied or linked to:

```
data/emails_candidates/
  ├── 00/<email_id>.json
  ├── 01/<email_id>.json
```

---

### 5.2 Classification Log (Append-Only)

```
data/state/step2b_vendor_scoring.jsonl
```

Each record MUST include:

```json
{
  "email_id": "...",
  "vendor_probability": 0.72,
  "predicted_label": "vendor",
  "threshold_used": 0.6,
  "model_version": "zeroshot_nli_v1",
  "timestamp": "..."
}
```

⚠️ These values are **not ground truth**.

---

### 5.3 Registry Update

Registry entry MUST record:

* last_completed_step = `"step2b_vendor_scoring"`
* content_hash
* bert_model_version
* inference_timestamp
* error status (if any)

Human validation **does NOT update the registry**.

---

## 6. Model Assumptions (Revised)

### 6.1 Initial Model (No Training Required)

The initial model MUST support inference without labeled data.

Allowed examples:

* Zero-shot NLI models (recommended)
* Weakly supervised transformer classifiers
* Any remote HTTP-based scoring service

The pipeline treats all models identically.

---

### 6.2 Future Model (Drop-In Replacement)

Once sufficient human-validated data exists, the model may be replaced with:

* a fine-tuned BERT-family vendor classifier

This change must be **transparent to pipeline logic**.

---

## 7. Input Text Construction (Still Mandatory)

Classifier input text MUST include:

1. Subject
2. Sender domain
3. Truncated body (first N tokens)
4. Signature block (if detected)

Token safety limits MUST be enforced.

---

## 8. Thresholding Strategy (Clarified)

Threshold controls **routing**, not truth.

* Lower threshold → higher recall
* Higher threshold → higher precision

Threshold MUST be configurable and tunable **without code changes**.

---

## 9. Human Validation Loop (Explicit, External)

Human validation is **out-of-band**, not part of inference.

Validated labels are stored in:

```
data/state/human_vendor_labels.jsonl
```

Each record:

```json
{
  "email_id": "...",
  "human_label": "vendor",
  "confidence": 0.9,
  "validated_at": "...",
  "notes": "Invoice email"
}
```

This file is **append-only** and authoritative for training.

---

## 10. Incremental Training Model (Authoritative)

### 10.1 Training Data Derivation

Training data is **derived**, not manually created:

```
JOIN(
  step2b_vendor_scoring.jsonl,
  human_vendor_labels.jsonl
)
```

---

### 10.2 Incremental Training Modes

Allowed:

* Periodic full retraining
* Continual fine-tuning

Each trained model MUST produce a new `model_version`.

---

### 10.3 Automatic Reprocessing

Changing `bert.model_version` MUST trigger:

* re-scoring in STEP 2B
* downstream reprocessing

No special flags or manual intervention allowed.

---

## 11. Error Handling (Unchanged)

* API retries required
* Failures logged
* Registry updated
* Batch continues

---

## 12. Validation Checklist (Updated)

Before proceeding to STEP 3:

* Vendor candidate rate ~1–5%
* False negatives are rare
* Borderline cases are available for validation
* Re-running skips unchanged emails
* Changing model_version triggers reprocessing

