#!/usr/bin/env bash
set -euo pipefail

# Example deployment script. Edit REMOTE_USER and REMOTE_HOST before use.
REMOTE_USER="ubuntu"
REMOTE_HOST="your.ubuntu.host"
REMOTE_PATH="~/gmail-sync-to-vendor-crm"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Syncing repository to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
rsync -avz --delete --exclude='.git' --exclude='venv' --exclude='data' --exclude='config/config.yaml' "$ROOT_DIR/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

echo "Remote sync complete. Connect and run the smoke script on the remote:"
echo "ssh ${REMOTE_USER}@${REMOTE_HOST} 'cd ${REMOTE_PATH} && bash scripts/run_smoke.sh'"


