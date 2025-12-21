# gmail-sync-to-vendor-crm — Runbook (STEP 1 → STEP 3)

This runbook explains how to run the pipeline **end-to-end** from **STEP 1 to STEP 3** on a machine that has your local Maildir (mbsync). It also includes **debugging + troubleshooting** and the **reasoning** behind key design decisions.

> **Core principle:** JSON files are the **source of truth**. Every downstream artifact is derived and rebuildable.

---

## 0) What this system does

You have a large mailbox synced locally in **Maildir** format (via `mbsync`) with **200k+ emails**. Only a small portion are vendor-related. The system:

1. **Ingests** all Maildir messages into **one JSON file per email** (lossless, auditable)
2. Uses **cheap heuristics** to reduce obvious junk (high recall)
3. Uses **AI scoring (HF zero-shot/NLI right now)** to route likely vendor emails into candidates (thresholded)
4. **Cleans** candidate email text (deterministic) to prepare for embeddings + LLM steps later

---

## 1) Prerequisites

* Ubuntu/Linux recommended
* Python **3.8+** (3.10+ ideal)
* `mbsync` already configured and syncing Maildir
* Virtualenv activated

Install deps:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Repository layout (important)

```text
gmail-sync-to-vendor-crm/
├── scripts/
│   ├── 01_ingest_maildir_to_json.py
│   ├── 02a_prefilter_vendor_emails.py
│   ├── 02b_bert_vendor_scoring.py
│   ├── 03_cleanup_vendor_emails.py
│   └── validate_vendor_candidates.py
├── config/
│   └── config.yaml
├── data/
│   ├── emails_raw_json/
│   ├── emails_prefiltered/
│   ├── emails_candidates/
│   ├── emails_cleaned/
│   └── state/
│       ├── processing_registry.jsonl
│       ├── step2a_prefilter_decisions.jsonl
│       ├── step2b_vendor_scoring.jsonl
│       └── human_vendor_labels.jsonl
└── requirements.txt
```

### What each folder means

* `data/emails_raw_json/`: **STEP 1 output** — all emails persisted as JSON (authoritative)
* `data/emails_prefiltered/`: **STEP 2A output** — heuristic-pass emails only
* `data/emails_candidates/`: **STEP 2B output** — AI-scored vendor candidates (thresholded)
* `data/emails_cleaned/`: **STEP 3 output** — cleaned/reduced text for semantic steps
* `data/state/`: append-only logs and registry (backbone for incrementality)

---

## 3) Configuration (`config/config.yaml`)

All scripts rely on `config/config.yaml`.

Key sections used so far:

* `maildir_root`: where Maildir folder lives
* `output.*`: output directories
* `state.*`: state files
* `prefilter.*`: scoring rules & threshold
* `bert.*`: STEP 2B scoring model settings (currently Hugging Face router endpoint)

**Hugging Face token**

Set your HF token in config (recommended) OR via environment variable (preferred if supported by your script):

```bash
export HF_TOKEN="..."
```

> If you use config-only auth, ensure `bert.auth_token` is filled.

---

## 4) STEP 1 — Ingest Maildir → Raw Email JSON

### Script

```text
scripts/01_ingest_maildir_to_json.py
```

### Purpose

* Scan Maildir `cur/` and `new/` (ignore `tmp/`)
* Parse MIME safely
* Write **one JSON per email**
* Compute and store:

  * `email_id`: stable identity (Message-ID hash preferred)
  * `content_hash`: detects changes to trigger reprocessing
* Update `data/state/processing_registry.jsonl`

### Run

Single folder:

```bash
python scripts/01_ingest_maildir_to_json.py \
  --maildir-root /home/you/Mail/Gmail/Inbox \
  --output-dir data/emails_raw_json \
  --state-dir data/state
```

Multiple folders (Inbox + Sent) if your script supports it:

```bash
python scripts/01_ingest_maildir_to_json.py \
  --maildir-roots /home/you/Mail/Gmail/Inbox /home/you/Mail/Gmail/Sent \
  --output-dir data/emails_raw_json \
  --state-dir data/state
```

### Validate

```bash
find data/emails_raw_json -type f -name "*.json" | wc -l
head -n 1 data/state/processing_registry.jsonl
```

### Common issues

* **No files ingested** → confirm the Maildir path actually contains `cur/` and `new/`:

```bash
ls -la /home/you/Mail/Gmail/Inbox
ls -la /home/you/Mail/Gmail/Inbox/cur | head
```

* **Parse errors** are OK — STEP 1 should not stop. It should mark `parsed_ok=false` and continue.

---

## 5) STEP 2A — Heuristic Pre-filter (cheap, high recall)

### Script

```text
scripts/02a_prefilter_vendor_emails.py
```

### Purpose

* Deterministic scoring of raw emails using keywords/regex/attachments
* High recall: OK to keep false positives, not OK to drop vendors
* Output only the subset worth running through AI

### Run

```bash
python scripts/02a_prefilter_vendor_emails.py \
  --raw-dir data/emails_raw_json \
  --prefiltered-dir data/emails_prefiltered \
  --state-dir data/state \
  --config config/config.yaml
```

### Validate pass rate + reasons

```bash
python - <<'PY'
import json, collections
p='data/state/step2a_prefilter_decisions.jsonl'
passed=0; total=0; reasons=collections.Counter()
for line in open(p, 'r', encoding='utf-8'):
  d=json.loads(line); total+=1
  if d.get('passed'): passed+=1
  for r in d.get('reasons', []):
    reasons[r]+=1
print('passed', passed, 'total', total, 'pass_rate', (passed/total if total else 0))
print('top_reasons:', reasons.most_common(20))
PY
```

### Troubleshooting

* **0 emails pass**

  * Lower `prefilter.threshold`
  * Ensure your keyword lists are populated
  * Confirm Step 1 produced `subject` and `raw_text`

* **Too many pass (>50%)**

  * Increase `prefilter.threshold`
  * Add negative weights for marketing/OTP

---

## 6) STEP 2B — Vendor relevance scoring (AI, human-in-the-loop)

### Script

```text
scripts/02b_bert_vendor_scoring.py
```

### What STEP 2B is (and is not)

* It produces `vendor_probability ∈ [0, 1]` for each prefiltered email
* It uses an **AI model over HTTP**
* It routes emails into candidates using a **threshold**

**Important:**

* The model output is a **proposal**, not ground truth
* Human labels are stored separately and used later for calibration and training

### Current model setup (what you actually used)

You successfully tested Hugging Face router inference with:

* Model: `facebook/bart-large-mnli`
* Endpoint pattern:

  * `https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli`

### Run

```bash
python scripts/02b_bert_vendor_scoring.py \
  --prefiltered-dir data/emails_prefiltered \
  --candidates-dir data/emails_candidates \
  --state-dir data/state \
  --config config/config.yaml \
  --link-method hardlink
```

Optional limit for testing:

```bash
--limit 30
```

### Outputs

* Scoring log:

  * `data/state/step2b_vendor_scoring.jsonl`
* Candidates (thresholded):

  * `data/emails_candidates/`

### Validate

```bash
wc -l data/state/step2b_vendor_scoring.jsonl
head -n 3 data/state/step2b_vendor_scoring.jsonl
find data/emails_candidates -type f -name "*.json" | wc -l
```

### Human validation (manual labeling)

Script:

```text
scripts/validate_vendor_candidates.py
```

Labels file (append-only):

```text
data/state/human_vendor_labels.jsonl
```

Recommended labeling confidence:

* `0.9–1.0` = very confident
* `0.6` = borderline
* avoid `0.0` in the final pass

Run validation focusing on borderline band:

```bash
python scripts/validate_vendor_candidates.py \
  --scoring-log data/state/step2b_vendor_scoring.jsonl \
  --raw-dir data/emails_raw_json \
  --prefiltered-dir data/emails_prefiltered \
  --labels-out data/state/human_vendor_labels.jsonl \
  --min-prob 0.65 \
  --max-prob 0.80
```

**Why it may say `Labeled 0 emails`:**

* The script avoids showing emails already present in the labels file.
* If you already labeled all emails in the 0.65–0.80 band, you will get 0.

### Threshold decision (what you concluded)

Based on probability distribution + labels, you locked:

* **`bert.threshold = 0.7`**

Quick distribution check:

```bash
jq '.vendor_probability' data/state/step2b_vendor_scoring.jsonl | sort -n | uniq -c
```

### Common failures & fixes

* **401 Unauthorized**

  * HF token missing/invalid
  * Check `bert.auth_token` and endpoint

* **NameResolutionError / DNS**

  * Endpoint in config points to placeholder (`bert.example`)
  * Use HF router endpoint

* **Script seems stuck**

  * Network retries/backoff in progress
  * Reduce retries temporarily or test with `--limit 5`

---

## 7) STEP 3 — Cleanup & Reduction (deterministic)

### Script

```text
scripts/03_cleanup_vendor_emails.py
```

### Purpose

* Remove noise while preserving signal:

  * quoted replies / threads
  * forwarded blocks
  * disclaimers
  * extra whitespace
* Output a compact `cleaned_text` used in embedding and LLM steps

### Run

```bash
python scripts/03_cleanup_vendor_emails.py \
  --candidates-dir data/emails_candidates \
  --output-dir data/emails_cleaned \
  --state-dir data/state
```

### Validate

```bash
find data/emails_cleaned -type f -name "*.json" | wc -l
jq '.cleaned_text' data/emails_cleaned/*/*.json | head -n 10
```

### What to expect

* Some system boilerplate may remain (e.g., marketplace templates). That’s OK.
* The goal is conservative cleanup (never delete critical evidence).

---

## 8) Current status (as of now)

* ✅ STEP 1 complete
* ✅ STEP 2A complete
* ✅ STEP 2B complete **and threshold locked (0.7)**
* ✅ STEP 3 complete
* 🔜 STEP 4 next: embeddings + vector DB (you intend to use **Pinecone free tier**)

---

## 9) Debugging cheat sheet

### “Which step produced what?”

* STEP 1 → `data/emails_raw_json/*/*.json`
* STEP 2A → `data/emails_prefiltered/*/*.json` + `data/state/step2a_prefilter_decisions.jsonl`
* STEP 2B → `data/emails_candidates/*/*.json` + `data/state/step2b_vendor_scoring.jsonl` + `human_vendor_labels.jsonl`
* STEP 3 → `data/emails_cleaned/*/*.json`

### “Why did an email get skipped?”

Because registry says the step is already complete for that `email_id` and `content_hash` is unchanged.

To inspect registry entry:

```bash
grep -F '"email_id": "<EMAIL_ID>"' data/state/processing_registry.jsonl | tail -n 5
```

### “How do I re-run a step for one email?”

Do **not** delete data. Instead, change one of:

* `content_hash` (email changed) OR
* model version (Step 2B/Step 4 later) OR
* add a script flag in future to force reprocess by email_id

For now, easiest controlled reprocessing is:

* bump the model version in config (Step 2B) or
* re-run on a new email set.

---

## 10) Reasoning behind key design choices

### Why persist all emails before filtering?

* Auditable
* Reproducible
* Allows swapping models later
* Prevents accidental data loss

### Why prefilter before AI?

* 200k emails is too expensive to score with AI
* Heuristics remove obvious junk cheaply
* AI runs only on plausible candidates

### Why keep labels separate from scoring?

* Scoring is an automated, re-runnable operation
* Human labels are ground truth
* Separation ensures reproducibility and allows retraining without rewriting pipeline logic

### Why deterministic cleanup before embeddings/LLM?

* Embeddings and LLMs work better on high-signal text
* Cleanup reduces token cost and improves semantic clustering
* Deterministic cleanup is predictable and debuggable

---

## 11) Known‑good config examples

This section captures **configs that are proven to work** based on actual runs. Use these as a baseline before experimenting.

---

### 11.1 Minimal working config (STEP 1–3)

```yaml
maildir_root: "/home/youruser/Mail/Gmail/Inbox"

output:
  emails_raw_json: "data/emails_raw_json"
  emails_prefiltered: "data/emails_prefiltered"
  emails_candidates: "data/emails_candidates"
  emails_cleaned: "data/emails_cleaned"

state:
  state_dir: "data/state"
  processing_registry: "data/state/processing_registry.jsonl"

prefilter:
  threshold: 4
  subject_weight: 1.0
  body_weight: 0.5

  positive_keywords:
    commercial:
      weight: 3
      terms: [invoice, payment, quote, proposal, contract]
    training:
      weight: 3
      terms: [training, workshop, course, session]

  negative_keywords:
    marketing:
      weight: -6
      terms: [unsubscribe, newsletter, promotion]

bert:
  endpoint: "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
  auth_token: "${HF_TOKEN}"
  model_version: "zeroshot_mnli_v1"
  timeout_seconds: 30
  max_retries: 3
  threshold: 0.7
```

**Why this works**

* Prefilter keeps recall high
* Zero‑shot MNLI gives usable probabilities without training
* Threshold `0.7` is empirically validated

---

### 11.2 Hugging Face auth (recommended)

Set token once per shell:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```

Then reference it in `config.yaml`:

```yaml
bert:
  auth_token: "${HF_TOKEN}"
```

This avoids committing secrets to disk.

---

### 11.3 Debug‑friendly config (during iteration)

Use this temporarily if debugging:

```yaml
bert:
  timeout_seconds: 10
  max_retries: 1
  threshold: 0.65
```

This makes failures fast and visible. **Do not keep this for production runs.**

---

### 11.4 Common misconfigurations to avoid

❌ Using placeholder endpoint:

```yaml
endpoint: "https://bert.example/api"
```

❌ Mismatched model + threshold (too aggressive):

```yaml
threshold: 0.85
```

❌ Missing auth token (causes 401 errors)

---

### 11.5 STEP 4 preview (do not use yet)

```yaml
embedding:
  endpoint: "https://api-inference.huggingface.co/pipeline/feature-extraction/..."
  auth_token: "${HF_TOKEN}"
  model_version: "embed_v1"

  vector_db:
    type: "pinecone"
    api_key: "${PINECONE_API_KEY}"
    environment: "us-east-1"
    index_name: "vendor-emails"
    dimension: 384
```

Use this **only after STEP 4 script is introduced**.

---

## 12) Next step preview (STEP 4)

STEP 4 will:

* chunk cleaned emails
* call embedding API
* upsert vectors into Pinecone (index only)
* update registry incrementally

Proceed only after confirming STEP 3 outputs look correct.
