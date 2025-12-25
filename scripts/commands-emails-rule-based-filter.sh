pip install rapidfuzz

ls -lah data/state_excl/step2b_vendor_scoring.jsonl

ls -lah data/state_excl
rm -rf data/state_excl
mkdir -p data/state_excl


ls -lah data/emails_prefiltered | head
ls -lah data/lists/


python3 02b_collect_vendor_training_data.py \
  --email-dir data/emails_prefiltered \
  --state-file data/state_excl/step2b_vendor_scoring.jsonl \
  --output-file data/vendor_training_review.jsonl \
  --allow-names data/lists/positive_vendor_names_clean.txt \
  --allow-keywords data/lists/positive_keywords_clean.txt \
  --allow-domains data/lists/positive_vendor_domains_clean.txt \
  --deny-domains data/lists/deny_domains_clean.txt \
  --deny-names data/lists/deny_names_clean_final.txt \
  --allow-threshold 88 \
  --allow-keyword-threshold 92 \
  --deny-threshold 90


wc -l data/vendor_training_review.jsonl
head -n 3 data/vendor_training_review.jsonl

grep -c '"rule_override": null' data/vendor_training_review.jsonl
grep -c '"rule_override": {' data/vendor_training_review.jsonl


python3 02b_quick_review_vendor_data.py --limit 50 --body-chars 200


# Update positive/ deny keywords

python3 02b_collect_vendor_training_data.py \
  --email-dir data/emails_prefiltered \
  --state-file data/state_excl/step2b_vendor_scoring.jsonl \
  --output-file data/vendor_training_review.jsonl \
  --allow-names data/lists/positive_vendor_names_clean.txt \
  --allow-keywords data/lists/positive_keywords_clean.txt \
  --allow-domains data/lists/positive_vendor_domains_clean.txt \
  --deny-domains data/lists/deny_domains_clean.txt \
  --deny-names data/lists/deny_names_clean_final.txt \
  --allow-threshold 88 \
  --allow-keyword-threshold 92 \
  --deny-threshold 90
