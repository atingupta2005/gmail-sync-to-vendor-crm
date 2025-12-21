#!/usr/bin/env bash

LOG="$HOME/mbsync.log"
SHORT_SLEEP=300        # 5 minutes
LONG_SLEEP=7200        # 2 hours
MAX_RETRIES=0          # 0 = infinite

retries=0

while true; do
  echo "[$(date)] Starting mbsync" >> "$LOG"

  mbsync -a >> "$LOG" 2>&1
  exit_code=$?

  echo "[$(date)] mbsync exited with code $exit_code" >> "$LOG"

  # If success, short sleep then continue incremental sync
  if [ "$exit_code" -eq 0 ]; then
    echo "[$(date)] Sync successful, sleeping $SHORT_SLEEP sec" >> "$LOG"
    sleep "$SHORT_SLEEP"
    continue
  fi

  # Detect quota / rate limit / timeout
  if tail -n 50 "$LOG" | grep -qiE "quota|rate|timeout|socket error|temporarily unavailable"; then
    echo "[$(date)] Detected quota/network issue, sleeping $LONG_SLEEP sec" >> "$LOG"
    sleep "$LONG_SLEEP"
  else
    echo "[$(date)] Unknown error, sleeping $SHORT_SLEEP sec" >> "$LOG"
    sleep "$SHORT_SLEEP"
  fi

  retries=$((retries + 1))
  if [ "$MAX_RETRIES" -ne 0 ] && [ "$retries" -ge "$MAX_RETRIES" ]; then
    echo "[$(date)] Max retries reached, exiting" >> "$LOG"
    exit 1
  fi
done
