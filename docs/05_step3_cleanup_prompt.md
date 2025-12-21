# STEP 3 — Cleanup & Reduction (Contact-Preserving)

## 1. Purpose of This Step (Revised)

This step **reduces noise while preserving all identity and contact evidence** from vendor-candidate emails.

By this point:

* Relevance has already been decided (Step 2B)
* Volume is low (~1% of total emails)

The goal is to:

* Remove redundant conversation history (quoted replies, forwarded chains)
* Remove non-semantic noise (HTML junk, MIME artifacts)
* Normalize text for embeddings and LLM reasoning
* **Preserve every possible vendor identification signal**

This step **must not remove any information that could identify, contact, or attribute a vendor**.

---

## 2. Position in the Pipeline

```
STEP 2B — BERT Classification (vendor candidates)
   ↓
STEP 3 — Cleanup & Reduction (CONTACT-PRESERVING)
   ↓
Cleaned Vendor Emails
```

---

## 3. Core Design Principles (Authoritative)

* Deterministic rules only (NO AI)
* Aggressive removal of *conversation noise*
* **Zero tolerance for loss of contact details**
* Evidence preservation over readability
* Incremental and idempotent

---

## 4. Absolute Preservation Rules (CRITICAL)

The following **MUST ALWAYS be preserved**:

### 4.1 Header Metadata (Never Clean)

These fields are **out of scope for cleanup** and must always be retained verbatim:

* `from`
* `to`
* `cc`
* `bcc`
* `reply_to`
* `subject`
* `date`
* `message_id`

Cleanup logic **must never infer or remove headers from the body**.

---

### 4.2 Contact & Identity Signals (Never Remove)

Cleanup logic **must not delete** any line containing:

* Email addresses
* Phone numbers (mobile, landline, international formats)
* Personal names
* Job titles
* Company names
* Physical addresses
* URLs (LinkedIn, company sites, calendars)
* Messaging handles (WhatsApp, Telegram, etc.)

If a line plausibly contains contact or identity information, **it must be preserved**, even if it appears inside:

* signatures
* disclaimers
* footers

---

## 5. Allowed Cleanup Operations (Strictly Scoped)

Cleanup operations MUST be applied **in order**, and MUST respect preservation rules.

---

## 6. Remove Quoted Replies (Conversation History Only)

### Detect quoted replies via:

* Lines starting with `>`
* Blocks starting with:

  * `On <date>, <person> wrote:`
  * Full RFC-822 reply headers **only when part of a reply chain**

### Rule

* Once a quoted-reply marker is detected, drop everything **below it**
* **Do NOT remove inline headers unless clearly part of a quoted reply**

---

## 7. Remove Forwarded Message Chains

### Detect via markers:

* `----- Forwarded message -----`
* `Begin forwarded message:`

### Rule

* Remove forwarded blocks entirely
* Preserve only the most recent authored message

---

## 8. Remove MIME / Base64 / Attachment Artifacts

### Detect via heuristics:

* Very long lines with no spaces
* Base64-only character runs
* MIME boundary markers

### Rule

* Remove these blocks completely
* Ensure no human-authored text is removed

---

## 9. Boilerplate & Disclaimers (Constrained)

### Important Constraint

Disclaimers **often contain contact or legal identity data**.

### Rule

* Disclaimers MAY be trimmed **only if**:

  * They occur **after the signature block**
  * AND they contain **no contact or identity signals**
* Otherwise, disclaimers MUST be preserved

---

## 10. Signature Handling (Extract, Never Delete)

### Signature Definition

A signature is **valuable structured data**, not noise.

### Strategy

* Detect signature delimiter from bottom up:

  * `--`
  * `Thanks,`
  * `Regards,`
  * `Best regards,`
  * `Sincerely,`

### Rule

* Split message into:

  * `cleaned_body_text`
  * `signature_text`
* **Never discard the signature**
* Preserve phone numbers, emails, names, titles, and companies

---

## 11. Whitespace Normalization (Safe Only)

Allowed:

* Collapse excessive blank lines
* Normalize line endings
* Trim trailing whitespace

Not allowed:

* Line-level deletion that could remove contact info

---

## 12. Length & Token Control (Non-Destructive)

### Purpose

* Control embedding and LLM token limits

### Rule

* Enforce max cleaned length (configurable)
* Truncate **from the bottom of the body only**
* **Never truncate signature**
* Prefer preserving:

  1. Subject
  2. Top of body
  3. Entire signature

---

## 13. Output JSON Requirements (Updated)

The cleaned email JSON MUST include:

```yaml
headers:
  from
  to
  cc
  subject
  date

body:
  raw_text
  cleaned_text
  signature_text

cleanup_stats:
  raw_length
  cleaned_length
  reduction_ratio
  preserved_contact_tokens: true
```

---

## 14. Incremental Processing Rules

Same as before:

1. Check registry
2. Skip if unchanged and already processed
3. Reprocess only on content change

---

## 15. Error Handling (Fail-Open)

If cleanup fails:

* Preserve raw text
* Preserve headers
* Preserve signature
* Log error
* Continue processing

---

## 16. Validation Checklist (Mandatory Before Step 4)

* No email loses phone numbers or email addresses
* Signatures are extracted, not deleted
* Headers are intact
* Cleaned text is shorter but semantically richer
* Re-runs do not alter unchanged emails

---

## 17. Guiding Principle (Final)

> **If a human could use the cleaned email to contact the vendor, Step 3 succeeded.
> If not, Step 3 failed.**

---

If you want, next I can:

* rewrite your cleanup functions to be **contact-aware**
* give you **regex guards for phone/email preservation**
* or provide a **minimal diff against your current code**

Just say the word.
