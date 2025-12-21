#!/usr/bin/env bash
set -euo pipefail

# Simple smoke runner: create venv, install deps, copy config, run ingest dry-run
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/venv"

echo "Root dir: $ROOT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

# ensure config exists
if [ ! -f "$ROOT_DIR/config/config.yaml" ]; then
  cp "$ROOT_DIR/config/config.yaml.example" "$ROOT_DIR/config/config.yaml"
  echo "Copied config example to config/config.yaml — please edit it before running non-dry runs"
fi

python "$ROOT_DIR/scripts/01_ingest_maildir_to_json.py" \
  --maildir-root "$(grep -m1 'maildir_root' -A0 "$ROOT_DIR/config/config.yaml" | awk -F': ' '{print $2}' | tr -d '\"')" \
  --output-dir "$ROOT_DIR/data/emails_raw_json" \
  --state-dir "$ROOT_DIR/data/state" \
  --limit 20 \
  --dry-run

echo "Smoke run finished (dry-run). Edit config/config.yaml and remove --dry-run to perform real run."


