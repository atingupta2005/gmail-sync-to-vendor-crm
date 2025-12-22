# Terminal 1
cd ~/gmail-sync-to-vendor-crm/
git pull
source venv/bin/activate

chmod +x scripts/00_run_mbsync_retry.sh

nohup scripts/00_run_mbsync_retry.sh > /dev/null 2>&1 &
pgrep mbsync
tail -f ~/mbsync.log

## Stop (Only if needed)
# pkill -f run_mbsync_retry
# pgrep mbsync


# Terminal 2
cd ~/gmail-sync-to-vendor-crm
source venv/bin/activate

python scripts/01_ingest_maildir_to_json.py \
  --maildir-roots \
    /home/atingupta2005/Mail/Gmail/Inbox \
    /home/atingupta2005/Mail/Gmail/[Gmail]/Sent\ Mail \
  --output-dir data/emails_raw_json \
  --state-dir data/state


# Terminal 3
cd ~/gmail-sync-to-vendor-crm
source venv/bin/activate

python scripts/02a_prefilter_vendor_emails.py \
  --input-dir data/emails_raw_json \
  --output-dir data/emails_prefiltered \
  --state-dir data/state \
  --config config/config.yaml


# Terminal 4
cd ~/gmail-sync-to-vendor-crm
source venv/bin/activate


python scripts/02b_bert_vendor_scoring.py \
  --prefiltered-dir data/emails_prefiltered \
  --candidates-dir data/emails_candidates \
  --state-dir data/state \
  --config config/config.yaml \
  --link-method copy


# Terminal 5
cd ~/gmail-sync-to-vendor-crm
source venv/bin/activate

python scripts/validate_vendor_candidates.py \
  --scoring-log data/state/bert_scoring_log.jsonl \
  --raw-dir data/emails_raw_json \
  --prefiltered-dir data/emails_prefiltered \
  --labels-out data/state/vendor_candidate_labels.jsonl


python scripts/03_cleanup_vendor_emails.py \
  --candidates-dir data/emails_candidates \
  --output-dir data/emails_cleaned \
  --state-dir data/state

# Terminal 6
cd ~/gmail-sync-to-vendor-crm
source venv/bin/activate


python scripts/04_build_rag_index.py \
  --cleaned-dir data/emails_cleaned \
  --config config/config.yaml


# Terminal 7
cd ~/gmail-sync-to-vendor-crm
source venv/bin/activate


python scripts/05_debug_semantic_search.py

find data/emails_raw_json -name "*.json" | wc -l
find data/emails_prefiltered -name "*.json" | wc -l
find data/emails_candidates -name "*.json" | wc -l
find data/emails_cleaned -name "*.json" | wc -l

chmod a+x scripts/status_check.sh
./scripts/status_check.sh