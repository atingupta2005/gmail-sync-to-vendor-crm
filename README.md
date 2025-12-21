# gmail-sync-to-vendor-crm

AI‑first pipeline to extract vendor intelligence from a local Maildir (mbsync). This repository contains step‑wise scripts (script‑first development) that you run manually on the machine that hosts the Maildir.

Quickstart (on the Ubuntu VM where your Maildir is located)

1. Clone the repo:

```bash
git clone <your-repo-url>
cd gmail-sync-to-vendor-crm
```

2. Create and activate a Python virtualenv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Copy the example config and edit paths/endpoints:

```bash
cp config/config.yaml.example config/config.yaml
# edit config/config.yaml to set maildir_root and endpoints
```

4. Run a dry‑run smoke test for ingestion:

```bash
python scripts/01_ingest_maildir_to_json.py \
  --maildir-root /home/you/Mail/account/Inbox \
  --output-dir data/emails_raw_json \
  --state-dir data/state \
  --limit 50 \
  --dry-run
```

5. Run the prefilter in dry-run:

```bash
python scripts/02a_prefilter_vendor_emails.py \
  --input-dir data/emails_raw_json \
  --output-dir data/emails_prefiltered \
  --state-dir data/state \
  --config config/config.yaml \
  --limit 50 \
  --dry-run
```

Notes
- The repository expects to be run on the machine with the Maildir; do not mount the maildir over network unless you understand file semantics.  
- `config/config.yaml` is excluded from git; keep secrets out of the repo.  
- Use `--limit` and `--only-prefix` flags to iterate quickly on small samples.

If you want, run `scripts/run_smoke.sh` to perform these steps automatically (edit the script for your environment).


