# Cursor Workflow Guide — How to Build This Project Step‑by‑Step

## 1. Purpose

This file explains **exactly how you should use Cursor** (and separate chats) to generate and implement the project safely, without mixing context or drifting from the plan.

It also defines a repeatable workflow for:

* running scripts manually
* validating outputs
* iterating safely
* keeping the project incremental

This guide is written for *you* (the developer/operator).

---

## 2. Golden Rules (Non‑Negotiable)

1. **One Cursor chat = One step.**

   * Never ask Cursor to implement multiple steps in one chat.

2. **Always paste the Master Prompt first.**

   * Use `docs/01_master_prompt_cursor.md` at the start of every chat.

3. **Always paste exactly one step prompt next.**

   * Example: Master Prompt + Step 1 Prompt only.

4. **Generate code, then run it manually.**

   * Validate outputs before moving to the next step.

5. **Do not “optimize” early.**

   * First achieve correctness, then improve performance.

6. **If you change requirements, update docs first.**

   * Treat docs/ as your source of truth.

---

## 3. Recommended Project Layout (Practical)

Even during script‑first development, keep a clean layout:

```
project_root/
  docs/
  scripts/
    01_ingest_maildir_to_json.py
    02a_prefilter_vendor_emails.py
    02b_bert_vendor_classifier.py
    03_cleanup_candidate_emails.py
    04_build_rag_index.py
    05_extract_vendors.py
    06_dedupe_vendors.py
    07_save_vendors.py
  data/
    state/
    emails_raw_json/
    emails_prefiltered/
    emails_candidates/
    emails_cleaned/
    vendors_raw/
    vendors_final/
    vendors_persisted/
  config/
    config.yaml
    config.yaml.example
  README.md
```

Notes:

* Keep generated scripts in `scripts/`.
* Keep all artifacts under `data/`.
* Keep config under `config/`.

---

## 4. How to Run Each Step (Manual Script Phase)

For every step, use this pattern:

1. Ensure config is correct
2. Run the script
3. Inspect output artifacts
4. Re‑run to confirm idempotency

### Example command pattern

```
python scripts/<script_name>.py --config config/config.yaml
```

If the generated scripts don’t use `--config`, update them to do so.

---

## 5. Cursor Chat Template (Copy/Paste Workflow)

### For Step N

1. Open a new Cursor chat
2. Paste:

   * `docs/01_master_prompt_cursor.md`
3. Paste:

   * the step prompt file for that step
4. Add one final instruction:

Use this exact line:

> Generate production‑ready Python 3.10+ code for the requested script. Output code only.

This keeps Cursor from adding extra narration.

---

## 6. Output Validation — What “Done” Means Per Step

### Step 1 — Ingestion

You are done when:

* Raw JSON exists for a sample of emails
* Each email JSON includes headers, body, and attachment metadata
* Re‑running the script skips unchanged emails
* Processing registry is created and updated

Smoke tests:

* Run once, then run again immediately → second run should be fast and mostly skip

---

### Step 2A — Prefilter

You are done when:

* Prefilter output volume is ~10–20% (roughly)
* Decision logs show explainable reasons
* You can adjust threshold without changing code
* Re‑running skips unchanged emails

Smoke tests:

* Spot check random 50 passing emails: should include many that look plausibly vendor-related

---

### Step 2B — BERT Classification

You are done when:

* Output candidate volume is ~1% (roughly)
* Probabilities look sane
* Model version changes trigger reprocessing
* API failures are retried and logged

Smoke tests:

* Manually inspect 50 passing candidates: most should be vendor-related

---

### Step 3 — Cleanup

You are done when:

* Cleaned text is much shorter than raw text
* Quoted chains mostly removed
* Signatures extracted frequently
* Stats log shows significant reduction ratio

Smoke tests:

* Compare raw vs cleaned for 20 emails

---

### Step 4 — RAG Index

You are done when:

* Vectors exist only for cleaned candidates
* Each vector has correct metadata
* Retrieval works (top-k returns relevant chunks)
* Re‑run skips unchanged

Smoke tests:

* Query vector DB with “invoice payment” and confirm results include invoice emails

---

### Step 5 — LLM Extraction

You are done when:

* Output is strict JSON
* Schema validation rejects invalid output
* Evidence references valid email_ids
* Output coverage is acceptable

Smoke tests:

* Sample 50 outputs; confirm names/emails/phones are not hallucinated

---

### Step 6 — Dedup

You are done when:

* Exact email duplicates merge deterministically
* AI merge decisions are logged
* Evidence preserved across merges
* Vendor count stabilizes over time

Smoke tests:

* Find two known duplicate vendors and confirm merge outcome

---

### Step 7 — Persistence

You are done when:

* Final JSON is valid
* Evidence append-only
* vendor_id stable
* Mongo upsert works if enabled

Smoke tests:

* Run twice; second run should be idempotent

---

## 7. Debugging Strategy

### 7.1 Always preserve artifacts

Never delete:

* raw email JSON
* decision logs
* registry

If a step is wrong, fix the step and re-run; do not wipe history unless you intentionally rebuild.

### 7.2 Use small sampling mode

Ask Cursor to include options like:

* `--limit N`
* `--only-prefix 0a`

Use these for iterative development.

### 7.3 Always log failures

Failures must go to:

* logs
* registry error field
* step-specific JSONL

---

## 8. When to Refactor Into Apps / Docker

Only refactor after:

* each step is validated
* outputs look correct
* incremental behavior is proven

Refactor checklist:

* extract reusable library modules
* wrap scripts in CLI commands
* containerize if desired

No logic changes during refactor.

---

## 9. Operating the System Incrementally (Ongoing Use)

After initial full run:

* run Step 1 daily/weekly to ingest new mail
* run Step 2A, 2B, 3, 4, 5 incrementally on new candidates
* run Step 6/7 to update vendor DB

A future improvement is a single orchestrator, but during script-first development you run manually.

---

## 10. Final Reminder

* Keep chats step-scoped
* Keep docs updated
* Validate each step output before moving on
* Preserve evidence
* Prefer correctness over speed
