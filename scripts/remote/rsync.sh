#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  source "$REPO_ROOT/.env"
fi

if [[ -z "${SSH_SERVER_NAME:-}" ]]; then
  echo "ERROR: SSH_SERVER_NAME is not set. Define it in .env or the environment." >&2
  exit 1
fi

REMOTE_PROJECT_PATH="${REMOTE_PROJECT_PATH:-/gpfs/yangsh/Code/deep-learning-playbook}"
REMOTE_PATH="${SSH_SERVER_NAME}:${REMOTE_PROJECT_PATH%/}/"

cd "$REPO_ROOT"

if rsync -avzP \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude 'env/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '.DS_Store' \
  --exclude '.obsidian/' \
  --include 'references/' \
  --include 'references/README.md' \
  --exclude 'references/**' \
  --exclude 'dataset/' \
  --exclude 'datasets/' \
  --exclude 'logs/' \
  --exclude 'runs/' \
  --exclude 'train/' \
  --exclude '*.pth' \
  --exclude '*.pt' \
  --exclude '*.ckpt' \
  --exclude '*.tar' \
  --exclude '*.tar.gz' \
  ./ "$REMOTE_PATH"; then
  echo "RSYNC_STATUS=SUCCESS"
  exit 0
else
  status=$?
  echo "RSYNC_STATUS=FAILED code=${status}" >&2
  exit "$status"
fi
