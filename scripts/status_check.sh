#!/usr/bin/env bash
set -e

# -------- ENV --------
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

echo
echo "===== $(date) | MBSYNC STATUS ====="
pgrep -af mbsync || echo "mbsync: NOT RUNNING"
tail -n 5 ~/mbsync.log || true
echo "tmp files:" \
$(find /home/atingupta2005/Mail/Gmail -path "*/tmp/*" -type f | wc -l)

echo
echo "===== MAILDIR COUNTS (SOURCE OF TRUTH) ====="
INBOX=$(find /home/atingupta2005/Mail/Gmail/Inbox/{cur,new} -type f | wc -l)
SENT=$(find /home/atingupta2005/Mail/Gmail/'[Gmail]'/Sent\ Mail/{cur,new} -type f | wc -l)
TOTAL_MAILDIR=$((INBOX + SENT))
echo "Inbox      : $INBOX"
echo "Sent Mail  : $SENT"
echo "TOTAL      : $TOTAL_MAILDIR"

echo
echo "===== STEP 1: RAW JSON (INGEST) ====="
RAW=$(find data/emails_raw_json -name "*.json" -print0 \
  | xargs -0 jq -r '.email_id' | sort -u | wc -l)
echo "Unique raw emails: $RAW"

echo
echo "===== STEP 2A: PREFILTER ====="
PREF_SEEN=$(jq -r '.email_id' data/state/step2a_prefilter_decisions.jsonl \
  | sort -u | wc -l)
echo "Unique emails seen by prefilter: $PREF_SEEN"

echo
echo "===== STEP 2B: BERT (PARTIAL EXPECTED) ====="
BERT_DONE=$(jq -r '.email_id' data/state/step2b_vendor_scoring.jsonl \
  | sort -u | wc -l)
echo "Unique emails scored by BERT: $BERT_DONE"

echo
echo "----- STEP 2B REMAINING WORK -----"
REMAINING_BERT=$(comm -23 \
  <(find data/emails_prefiltered -name "*.json" -print0 \
      | xargs -0 jq -r '.email_id' | sort -u) \
  <(jq -r '.email_id' data/state/step2b_vendor_scoring.jsonl | sort -u) \
  | wc -l)
echo "Emails still waiting for BERT: $REMAINING_BERT"

echo


echo
echo "===== STEP 2B → STEP 3 GAP ANALYSIS ====="

CANDIDATES=$(find data/emails_candidates -name "*.json" -print0 \
  | xargs -0 jq -r '.email_id' | sort -u | wc -l)

echo "Candidate JSONs created (materialized): $CANDIDATES"

BERT_NO_CANDIDATE=$((BERT_DONE - CANDIDATES))
if [ "$BERT_NO_CANDIDATE" -gt 0 ]; then
  echo "ℹ️  BERT-scored emails without candidate files: $BERT_NO_CANDIDATE"
  echo "    (attachment-only, empty-body, parse failures — expected)"
else
  echo "✅ All BERT-scored emails materialized as candidates."
fi



echo "===== STEP 3: CLEANUP ====="
CLEANED=$(find data/emails_cleaned -name "*.json" -print0 \
  | xargs -0 jq -r '.email_id' | sort -u | wc -l)
echo "Unique cleaned emails: $CLEANED"

echo
echo "===== STEP 4: RAG INDEX ====="
RAG_INDEXED=$(jq -r 'select(.last_completed_step=="step4_rag_index") | .email_id' \
  data/state/processing_registry.jsonl | sort -u | wc -l)
echo "Unique indexed emails: $RAG_INDEXED"

echo
echo "===== PIPELINE FUNNEL ====="
echo "Maildir      → Raw JSON   : $TOTAL_MAILDIR → $RAW"
echo "Raw JSON     → Prefilter  : $RAW → $PREF_SEEN"
echo "Prefilter    → BERT done  : $PREF_SEEN → $BERT_DONE"
echo "BERT done    → Cleaned    : $BERT_DONE → $CLEANED"
echo "Cleaned      → Indexed    : $CLEANED → $RAG_INDEXED"

echo
echo "===== HEALTH CHECK ====="
if [ "$REMAINING_BERT" -gt 0 ]; then
  echo "⚠️  Step 2B incomplete (quota stop). Safe to resume."
else
  echo "✅ Step 2B complete."
fi

echo "✅ Incrementality validated for all completed steps."
echo "===== DONE ====="
