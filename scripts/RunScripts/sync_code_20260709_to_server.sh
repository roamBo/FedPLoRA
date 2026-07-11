#!/usr/bin/env bash
# Sync code-only changes for the 20260709 FedPLoRA runs.
#
# Run this on the local Mac, not on the server:
#   cd /Users/hawaiii/codex/FedPLoRA/FedPLoRA-main
#   REMOTE=minghao@172.26.191.30 bash scripts/RunScripts/sync_code_20260709_to_server.sh
#
# This intentionally does not sync data/log/result/model directories.

set -euo pipefail

REMOTE=${REMOTE:-minghao@172.26.191.30}
REMOTE_DIR=${REMOTE_DIR:-/data2/minghao/code/FedPLoRA-main}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "[sync] LOCAL_DIR=$LOCAL_DIR"
echo "[sync] REMOTE=$REMOTE"
echo "[sync] REMOTE_DIR=$REMOTE_DIR"

rsync_common=(
  -av
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
)

ssh "$REMOTE" "mkdir -p '$REMOTE_DIR/methods' '$REMOTE_DIR/tasks' '$REMOTE_DIR/utilities' '$REMOTE_DIR/scripts' '$REMOTE_DIR/configs'"

rsync "${rsync_common[@]}" "$LOCAL_DIR/methods/" "$REMOTE:$REMOTE_DIR/methods/"
rsync "${rsync_common[@]}" "$LOCAL_DIR/tasks/" "$REMOTE:$REMOTE_DIR/tasks/"
rsync "${rsync_common[@]}" "$LOCAL_DIR/utilities/" "$REMOTE:$REMOTE_DIR/utilities/"
rsync "${rsync_common[@]}" "$LOCAL_DIR/scripts/" "$REMOTE:$REMOTE_DIR/scripts/"
rsync "${rsync_common[@]}" "$LOCAL_DIR/configs/" "$REMOTE:$REMOTE_DIR/configs/"

ssh "$REMOTE" "cd '$REMOTE_DIR' && python - <<'PY'
import importlib.util
mods = [
    'methods.v8',
    'methods.v9',
    'methods.v10',
    'methods.v11',
    'methods.v12',
    'methods.v13',
    'methods.lora_expert_baselines',
    'utilities.utils',
    'utilities.train_eval',
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print('[sync][error] missing modules after sync:', missing)
    raise SystemExit(2)
print('[sync][ok] required module paths exist')
PY"

echo "[sync][ok] code sync complete"
