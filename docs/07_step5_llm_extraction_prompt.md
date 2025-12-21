# STEP 5 — LLM + RAG Vendor Extraction & Reasoning

## 1. Purpose of This Step

This step converts **cleaned, relevant emails** into **structured vendor intelligence** using a **remote LLM grounded by RAG context**.

At this stage:

* Relevance is confirmed (Step 2B)
* Noise is reduced (Step 3)
* Semantic retrieval is available (Step 4)

The LLM is used for **understanding and reasoning**, not discovery.

---

## 2. Position in the Pipeline

```
STEP 4 — Vector DB / RAG Index
   ↓
STEP 5 — LLM + RAG Extraction & Reasoning
   ↓
Structured Vendor Candidates
```

Only emails indexed in the vector DB are processed here.

---

## 3. Core Design Principles

* LLM must never hallucinate
* All outputs must be **strict JSON**
* RAG context is mandatory for grounding
* JSON schema validation is required
* Incremental execution only

---

## 4. Inputs

* Cleaned email JSON files from:

  ```
  data/emails_cleaned/
  ```
* Vector DB retriever
* Processing registry

---

## 5. Outputs

### 5.1 Vendor Extraction Records

Append‑only JSONL file:

```
data/vendors_raw/vendor_extractions.jsonl
```

Each record represents **one extraction attempt for one email**.

### 5.2 Registry Update

Update registry with:

* last_completed_step = "step5_llm_extraction"
* llm_model_version

---

## 6. RAG Context Assembly

### 6.1 Mandatory Context Elements

For each email:

1. The email’s own cleaned text
2. Its extracted signature
3. Top‑K similar chunks retrieved from vector DB

This ensures:

* grounding
* cross‑email context
* consistency

---

### 6.2 Retrieval Strategy

* Use cosine similarity
* Typical K: 3–5
* Filter by:

  * same sender domain (if possible)
  * recent dates (optional)

---

## 7. Vendor Extraction Schema (Authoritative)

The LLM **must** return JSON matching exactly this schema:

```
{
  "person_name": string | null,
  "person_email": string | null,
  "phone_numbers": [string],
  "organization": string | null,
  "domain": string | null,
  "category": "invoice" | "payment" | "training" | "course" | "contract" | "other",
  "confidence": number,
  "evidence": [
    {
      "email_id": string,
      "reason": string
    }
  ]
}
```

Rules:

* Do NOT invent fields
* Use `null` if unknown
* `confidence` must be between 0 and 1

---

## 8. LLM Prompt Structure

The prompt sent to the LLM must include:

1. System instruction (role + rules)
2. Explicit JSON schema
3. RAG context
4. Clear extraction task

The prompt must explicitly state:

* “Do not guess”
* “Use only provided context”
* “Return JSON only”

---

## 9. LLM API Interaction

### 9.1 Request (Conceptual)

```json
{
  "prompt": "<assembled prompt>",
  "model_version": "llm_vendor_extract_v1",
  "temperature": 0.0
}
```

Low temperature is mandatory.

---

### 9.2 Response Handling

The script must:

1. Parse JSON strictly
2. Validate against schema
3. Reject responses that:

   * are not valid JSON
   * contain extra fields
   * violate schema

---

## 10. Retry & Fallback Strategy

* If JSON parsing fails:

  * retry up to N times
* If still failing:

  * log failure
  * mark registry
  * skip this email for now

Never silently accept invalid output.

---

## 11. Incremental Processing Rules

For each email:

1. Check registry
2. Skip if:

   * last_completed_step >= step5_llm_extraction
   * content_hash unchanged
   * llm_model_version unchanged
3. Re‑run if:

   * content changed, OR
   * LLM model version changed

---

## 12. Confidence Scoring Guidelines

The LLM should assign confidence based on:

* explicit presence of contact info
* clarity of organization name
* consistency across RAG context

Confidence is advisory and may be recalculated later.

---

## 13. Configuration Requirements

This step must read from config:

* llm.endpoint
* llm.auth_token
* llm.model_version
* llm.timeout_seconds
* llm.max_retries
* rag.top_k

---

## 14. Deliverables

Cursor must generate:

* `05_extract_vendors.py`
* RAG retrieval integration
* LLM API client
* Prompt builder
* JSON schema validator
* Incremental registry integration
* CLI interface

---

## 15. Validation Checklist

Before proceeding to Step 6:

* LLM outputs strict JSON
* No hallucinated vendors
* Evidence references correct email_id
* Re‑running script skips unchanged emails
* Changing llm_model_version triggers reprocessing

This step produces the raw material for vendor deduplication.
