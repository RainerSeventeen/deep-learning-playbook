#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=/dev/null
source "$REPO_ROOT/.codex/remote.env"
: "${REMOTE_SSH_ALIAS:?Set REMOTE_SSH_ALIAS in .codex/remote.env}"
: "${REMOTE_PROJECT_PATH:?Set REMOTE_PROJECT_PATH in .codex/remote.env}"
REMOTE_PATH="${REMOTE_SSH_ALIAS}:${REMOTE_PROJECT_PATH%/}/"

cd "$REPO_ROOT"

if rsync -avzP \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude '.codex/remote.env' \
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
