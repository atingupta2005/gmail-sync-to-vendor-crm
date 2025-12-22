#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# mbsync single-instance retry supervisor (FINAL, SAFE VERSION)
# ============================================================

# -------- Configuration --------
LOG="$HOME/mbsync.log"

SHORT_SLEEP=300        # 5 minutes after successful sync
LONG_SLEEP=7200        # 2 hours after quota / network issues
MAX_RETRIES=0          # 0 = infinite retries

LOCKFILE="$HOME/.mbsync_retry.lock"

# -------- Single-instance lock --------
exec 9>"$LOCKFILE" || exit 1

if ! flock -n 9; then
  echo "[$(date)] Another mbsync retry instance is already running, exiting." >> "$LOG"
  exit 0
fi

# -------- Retry loop --------
retries=0

while true; do
  echo "[$(date)] Starting mbsync" >> "$LOG"

  mbsync -a >> "$LOG" 2>&1
  exit_code=$?

  echo "[$(date)] mbsync exited with code $exit_code" >> "$LOG"

  # ---- Success ----
  if [ "$exit_code" -eq 0 ]; then
    echo "[$(date)] Sync successful, sleeping $SHORT_SLEEP seconds" >> "$LOG"
    sleep "$SHORT_SLEEP"
    continue
  fi

  # ---- Graceful termination (SIGTERM / SIGINT) ----
  if [ "$exit_code" -eq 143 ] || [ "$exit_code" -eq 130 ]; then
    echo "[$(date)] Received termination signal, exiting cleanly" >> "$LOG"
    exit 0
  fi

  # ---- Gmail quota / network handling ----
  if tail -n 50 "$LOG" | grep -qiE "quota|rate|timeout|socket error|temporarily unavailable"; then
    echo "[$(date)] Detected quota/network issue, sleeping $LONG_SLEEP seconds" >> "$LOG"
    sleep "$LONG_SLEEP"
  else
    echo "[$(date)] Unknown error, sleeping $SHORT_SLEEP seconds" >> "$LOG"
    sleep "$SHORT_SLEEP"
  fi

  retries=$((retries + 1))
  if [ "$MAX_RETRIES" -ne 0 ] && [ "$retries" -ge "$MAX_RETRIES" ]; then
    echo "[$(date)] Max retries reached, exiting" >> "$LOG"
    exit 1
  fi
done
