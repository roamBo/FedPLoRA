#!/usr/bin/env bash
# gb external lm-eval launcher: correct cwd, cache, batch size, absolute paths.
set -eo pipefail

CODE_DIR="${CODE_DIR:-/data/yaominghao/gb/FedPLoRA}"
RESULT_ROOT="${RESULT_ROOT:-/data/yaominghao/gb/result/FedPLoRA/order_0723}"
GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PY="${PY:-/data/yaominghao/miniconda3/envs/fedplora/bin/python}"
SCRIPT="${CODE_DIR}/scripts/Analysis/run_external_lm_eval.py"
CACHE_ROOT="${CODE_DIR}/data/external_lm_eval_hf_cache"
LOG_DIR="${RESULT_ROOT}/launcher_logs"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/RunScripts/run_external_eval_gb.sh e1-normal
  bash scripts/RunScripts/run_external_eval_gb.sh e2-v13a
  bash scripts/RunScripts/run_external_eval_gb.sh verify-cache
  bash scripts/RunScripts/run_external_eval_gb.sh prepare-cache [mmlu,pubmedqa,mbpp]

Env overrides: CODE_DIR RESULT_ROOT GPU_ID BATCH_SIZE
EOF
}

die() { echo "[external-eval-gb][error] $*" >&2; exit 1; }

cd "$CODE_DIR" || die "missing CODE_DIR=$CODE_DIR"
export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}"
mkdir -p "$LOG_DIR"

test -f "$SCRIPT" || die "missing script: $SCRIPT"
test -x "$PY" || test -f "$PY" || die "missing python: $PY"

verify_cache() {
  export HF_HOME="$CACHE_ROOT"
  export HF_HUB_CACHE="$CACHE_ROOT/hub"
  export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hub"
  export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  unset HF_ENDPOINT
  "$PY" "${CODE_DIR}/scripts/Analysis/prepare_external_lm_eval_hf_cache.py" \
    --cache_root "$CACHE_ROOT" --tasks mmlu,pubmedqa,mbpp --verify_only
}

prepare_cache() {
  local tasks="${1:-mmlu,pubmedqa,mbpp}"
  export HF_HOME="$CACHE_ROOT"
  export HF_HUB_CACHE="$CACHE_ROOT/hub"
  export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hub"
  export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
  "$PY" "${CODE_DIR}/scripts/Analysis/prepare_external_lm_eval_hf_cache.py" \
    --cache_root "$CACHE_ROOT" --tasks "$tasks" --purge --hf_endpoint "$HF_ENDPOINT"
  verify_cache
}

run_eval() {
  local tag="$1"
  local manifest="$2"
  local tasks="$3"
  local mode="$4"
  local outdir="$5"
  local logfile="$6"
  local extra="${7:-}"

  test -f "$manifest" || die "missing manifest: $manifest"
  verify_cache

  echo "[external-eval-gb] tag=$tag GPU_ID=$GPU_ID batch=$BATCH_SIZE out=$outdir"
  CUDA_VISIBLE_DEVICES="$GPU_ID" nohup /usr/bin/time -v \
    "$PY" "$SCRIPT" \
    --adapter_manifest "$manifest" \
    --tasks "$tasks" \
    --mode "$mode" \
    --device cuda:0 \
    --batch_size "$BATCH_SIZE" \
    --hf_cache_dir "$CACHE_ROOT" \
    --confirm_run_unsafe_code \
    --output_dir "$outdir" \
    $extra \
    > "$logfile" 2>&1 &
  echo "[external-eval-gb] pid=$! log=$logfile"
}

cmd="${1:-}"
case "$cmd" in
  verify-cache) verify_cache ;;
  prepare-cache) prepare_cache "${2:-mmlu,pubmedqa,mbpp}" ;;
  e1-normal)
    run_eval e1-normal \
      "$RESULT_ROOT/external_adapters/normal_seed42/adapter_export_manifest.json" \
      "mmlu:general,pubmedqa:medical,mbpp:code" \
      global \
      "$RESULT_ROOT/external_eval/normal_seed42" \
      "$LOG_DIR/test20260723_external_normal_seed42.log"
    ;;
  e2-v13a)
    run_eval e2-v13a \
      "$RESULT_ROOT/external_adapters/v13a_seed42/adapter_export_manifest.json" \
      "mmlu:general,pubmedqa:medical,mbpp:code" \
      both \
      "$RESULT_ROOT/external_eval/v13a_seed42" \
      "$LOG_DIR/test20260723_external_v13a_seed42.log"
    ;;
  *) usage; exit 2 ;;
esac
