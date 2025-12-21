# STEP 1 — Maildir Ingestion → Raw Email JSON

## 1. Purpose of This Step

This step is responsible for **ingesting all emails** from the local Maildir (synced via mbsync) and **persisting each email as a standalone JSON file**.

This is the **foundation of the entire system**.

Key characteristics:

* Zero intelligence
* Zero filtering
* Zero AI usage
* Lossless persistence

If this step is wrong, every downstream step is compromised.

---

## 2. What This Step Must and Must NOT Do

### MUST DO

* Read emails from Maildir (`cur/`, `new/`)
* Parse MIME safely
* Persist one JSON per email
* Generate stable identity and content hash
* Support incremental ingestion
* Record processing state

### MUST NOT DO

* No filtering
* No cleanup
* No AI calls
* No embeddings
* No summarization

---

## 3. Inputs

### 3.1 Maildir Root

* Provided via CLI or config
* Example:

  ```
  ~/Mail/account/Inbox/
  ```

### 3.2 Maildir Semantics

* Only scan:

  * `cur/`
  * `new/`
* Ignore:

  * `tmp/`
* Each file represents a complete email message

---

## 4. Outputs

### 4.1 Raw Email JSON Files

Each email must be written as **one JSON file**.

Directory structure:

```
data/emails_raw_json/
  ├── 00/
  │   ├── <email_id>.json
  ├── 01/
  └── ff/
```

* Subdirectory = first two hex chars of `email_id`
* Prevents filesystem performance issues

### 4.2 Processing Registry

Append-only JSONL file:

```
data/state/processing_registry.jsonl
```

Tracks ingestion state per email.

---

## 5. Canonical Email Identity

### 5.1 email_id (Stable Identity)

The `email_id` must remain stable across runs.

Priority:

1. If `Message-ID` exists:

   ```
   email_id = sha1(message_id)
   ```
2. Else:

   ```
   email_id = sha1(normalized_headers + normalized_body)
   ```

Normalization rules:

* lowercase headers
* trim whitespace
* normalize line endings

---

## 6. Change Detection

### 6.1 content_hash

A separate hash used to detect changes.

Computed from:

* raw email bytes OR
* normalized raw text

Purpose:

* detect email modifications
* trigger reprocessing

---

## 7. JSON Schema (Raw Email)

Each raw email JSON **must** contain the following sections.

### 7.1 Top-Level Fields

* `email_id`
* `content_hash`
* `maildir_path`
* `file_mtime`
* `file_size`

### 7.2 Headers

```
headers:
  from
  to
  cc
  subject
  date
  message_id
  in_reply_to
  references
```

Preserve original header values as strings.

### 7.3 MIME Metadata

```
mime_meta:
  has_attachments: boolean
  attachments:
    - filename
      content_type
      size (if available)
```

Do NOT store attachment content.

### 7.4 Body

```
body:
  raw_text
  raw_html (optional)
```

Extraction rules:

* Prefer `text/plain`
* Fallback to HTML stripped to text

### 7.5 Processing Metadata

```
processing:
  ingested_at
  schema_version
  parsed_ok
  parse_error (if any)
```

---

## 8. Incremental Processing Rules

Before ingesting an email:

1. Compute email_id
2. Look up registry
3. If email_id exists AND content_hash unchanged:

   * skip
4. If content_hash changed:

   * re‑ingest
   * overwrite JSON
   * update registry

---

## 9. Error Handling

* Parsing failures must NOT stop the run
* For failed emails:

  * write JSON with `parsed_ok = false`
  * record error message
  * continue processing others

---

## 10. Performance Considerations

* Stream file processing
* Do NOT load all emails into memory
* Process directory iteratively

---

## 11. Deliverables

Cursor must generate:

* `01_ingest_maildir_to_json.py`
* CLI arguments:

  * `--maildir-root`
  * `--output-dir`
  * `--state-dir`
* Logging
* Clear comments

---

## 12. Validation Checklist

After implementation, verify:

* JSON created for every email
* email_id stable across runs
* content_hash changes detected
* registry updates correctly
* broken emails don’t crash run

This step must be rock-solid before proceeding.
