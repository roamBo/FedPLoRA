#!/usr/bin/env bash
# Prepare or verify the repo-local HuggingFace cache used by external lm-eval.
#
# Default online source is https://hf-mirror.com so collaborators on networks
# where huggingface.co is slow/blocked can still build the cache once. Formal
# evaluation should then run offline via --hf_cache_dir.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PY="${PY:-python}"
TASKS="${TASKS:-mmlu,pubmedqa,mbpp}"
CACHE_ROOT="${HF_CACHE_ROOT:-${CACHE_ROOT:-${CODE_DIR}/data/external_lm_eval_hf_cache}}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PURGE="${PURGE:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/RunScripts/prepare_external_lm_eval_cache.sh probe
  bash scripts/RunScripts/prepare_external_lm_eval_cache.sh verify [tasks]
  bash scripts/RunScripts/prepare_external_lm_eval_cache.sh prepare [tasks]
  bash scripts/RunScripts/prepare_external_lm_eval_cache.sh prepare-official [tasks]

Defaults:
  tasks      = mmlu,pubmedqa,mbpp
  cache root = $CODE_DIR/data/external_lm_eval_hf_cache
  mirror     = https://hf-mirror.com

Useful overrides:
  export CODE_DIR=/abs/path/to/FedPLoRA
  export PY=/abs/path/to/conda/env/bin/python
  export HF_ENDPOINT=https://hf-mirror.com
  export TASKS=mmlu,pubmedqa,mbpp
  export PURGE=1   # rebuild task cache if a partial/corrupt cache exists

Recommended flow on a server that cannot reliably reach huggingface.co:
  bash scripts/RunScripts/prepare_external_lm_eval_cache.sh probe
  bash scripts/RunScripts/prepare_external_lm_eval_cache.sh prepare
  bash scripts/RunScripts/prepare_external_lm_eval_cache.sh verify

If neither huggingface.co nor hf-mirror.com is reachable, prepare this same
cache on any online machine and rsync it to:
  $CODE_DIR/data/external_lm_eval_hf_cache
EOF
}

die() {
  echo "[hf-cache-launcher][error] $*" >&2
  exit 1
}

resolve_python() {
  if "$PY" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[0] >= 3 else 1)
PY
  then
    echo "$PY"
    return
  fi
  for candidate in python3 /data/yaominghao/miniconda3/envs/fedplora/bin/python /data2/minghao/miniconda3/envs/FedRepo2/bin/python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[0] >= 3 else 1)
PY
    then
      echo "$candidate"
      return
    fi
  done
  die "cannot find Python 3. Set PY=/abs/path/to/python from the conda env."
}

probe_endpoints() {
  echo "[hf-cache-launcher] endpoint probe; HTTP 200/3xx means reachable enough for metadata."
  for url in https://huggingface.co https://hf-mirror.com https://www.modelscope.cn; do
    echo "[probe] ${url}"
    if command -v curl >/dev/null 2>&1; then
      if command -v timeout >/dev/null 2>&1; then
        timeout 15 curl -I -L --connect-timeout 8 --max-time 15 "$url" 2>&1 | sed -n '1,8p' || true
      else
        curl -I -L --connect-timeout 8 --max-time 15 "$url" 2>&1 | sed -n '1,8p' || true
      fi
    else
      echo "[probe][skip] curl not found"
    fi
  done
}

set_cache_env() {
  export HF_HOME="$CACHE_ROOT"
  export HF_HUB_CACHE="$CACHE_ROOT/hub"
  export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hub"
  export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
}

verify_cache() {
  local py="$1"
  local tasks="$2"
  set_cache_env
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  unset HF_ENDPOINT
  "$py" "$CODE_DIR/scripts/Analysis/prepare_external_lm_eval_hf_cache.py" \
    --cache_root "$CACHE_ROOT" --tasks "$tasks" --verify_only
}

prepare_cache() {
  local py="$1"
  local tasks="$2"
  local endpoint="$3"
  set_cache_env
  unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
  local endpoint_args=()
  if [[ -n "$endpoint" ]]; then
    export HF_ENDPOINT="$endpoint"
    endpoint_args=(--hf_endpoint "$endpoint")
  else
    unset HF_ENDPOINT
  fi
  local purge_args=()
  if [[ "$PURGE" == "1" || "$PURGE" == "true" || "$PURGE" == "TRUE" ]]; then
    purge_args=(--purge)
  fi
  mkdir -p "$CACHE_ROOT"
  echo "[hf-cache-launcher] prepare tasks=${tasks} cache=${CACHE_ROOT} endpoint=${endpoint:-official/default} purge=${PURGE}"
  "$py" "$CODE_DIR/scripts/Analysis/prepare_external_lm_eval_hf_cache.py" \
    --cache_root "$CACHE_ROOT" --tasks "$tasks" "${purge_args[@]}" "${endpoint_args[@]}"
  verify_cache "$py" "$tasks"
}

cmd="${1:-help}"
if [[ $# -ge 2 ]]; then
  TASKS="$2"
fi

cd "$CODE_DIR" || die "missing CODE_DIR=$CODE_DIR"
test -f "$CODE_DIR/scripts/Analysis/prepare_external_lm_eval_hf_cache.py" \
  || die "missing prepare script under $CODE_DIR/scripts/Analysis"
PY_RESOLVED="$(resolve_python)"
echo "[hf-cache-launcher] CODE_DIR=$CODE_DIR"
echo "[hf-cache-launcher] PY=$PY_RESOLVED"
echo "[hf-cache-launcher] CACHE_ROOT=$CACHE_ROOT"

case "$cmd" in
  probe)
    probe_endpoints
    ;;
  verify)
    verify_cache "$PY_RESOLVED" "$TASKS"
    ;;
  prepare)
    probe_endpoints
    prepare_cache "$PY_RESOLVED" "$TASKS" "$HF_ENDPOINT"
    ;;
  prepare-official)
    probe_endpoints
    prepare_cache "$PY_RESOLVED" "$TASKS" ""
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
