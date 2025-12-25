# 0) Install fuzzy dependency (use python from venv)
python3 -m pip install -U rapidfuzz

# 1) (Optional) sanity checks
ls -lah data/emails_prefiltered | head
ls -lah data/state_excl/step2b_vendor_scoring.jsonl

rm -rf data/state_excl
mkdir -p data/state_excl

rm -f data/vendor_training_review.jsonl

# 2) Run training-data collector (UPDATED script is in scripts/)
python3 scripts/02b_collect_vendor_training_data.py \
  --email-dir data/emails_prefiltered \
  --state-file data/state/step2b_vendor_scoring.jsonl \
  --output-file data/vendor_training_review.jsonl \
  --allow-names scripts/data/lists/positive_vendor_names_clean.txt \
  --allow-domains scripts/data/lists/positive_vendor_domains_clean.txt \
  --deny-domains scripts/data/lists/deny_domains_clean.txt \
  --deny-names scripts/data/lists/deny_names_clean.txt \
  --allow-threshold 94 \
  --deny-threshold 88

# 3) Verify output
wc -l data/vendor_training_review.jsonl
head -n 3 data/vendor_training_review.jsonl

# 4) Quick stats: how many records had rule overrides
grep -c '"rule_override": null' data/vendor_training_review.jsonl
grep -c '"rule_override": {' data/vendor_training_review.jsonl

# 5) Manual review (script is also in scripts/)
python3 scripts/02b_quick_review_vendor_data.py --limit 50 --body-chars 200

# Only review vendors
python3 - <<'PY'
import json
src="data/vendor_training_review.jsonl"
dst="data/vendor_training_review_only_vendor.jsonl"

n=0
with open(src,"r",encoding="utf-8") as f, open(dst,"w",encoding="utf-8") as out:
    for line in f:
        try:
            r=json.loads(line)
        except Exception:
            continue
        if r.get("predicted_label") == "vendor":
            out.write(line)
            n += 1

print("Wrote", n, "vendor-predicted rows to", dst)
PY

python3 scripts/02b_quick_review_vendor_data.py \
  --input data/vendor_training_review_only_vendor.jsonl \
  --limit 50 \
  --body-chars 200


# 6) Update your keyword files manually:
# - data/lists/positive_keywords_clean.txt
# - data/lists/deny_names_clean_final.txt
# - (optional) data/lists/positive_vendor_names_clean.txt / positive_vendor_domains_clean.txt

# 7) Re-run with the same full command (DO NOT switch back to old flags)
python3 scripts/02b_collect_vendor_training_data.py \
  --email-dir data/emails_prefiltered \
  --state-file data/state_excl/step2b_vendor_scoring.jsonl \
  --output-file data/vendor_training_review.jsonl \
  --allow-names data/lists/positive_vendor_names_clean.txt \
  --allow-keywords data/lists/positive_keywords_clean.txt \
  --allow-domains data/lists/positive_vendor_domains_clean.txt \
  --deny-domains data/lists/deny_domains_clean.txt \
  --deny-names data/lists/deny_names_clean_final.txt \
  --allow-threshold 88 \
  --deny-threshold 90
