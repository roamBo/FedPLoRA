# shellcheck shell=bash
# Shared helpers for LW v4 RunScripts (7 clients, Qwen-0.5B, LW-tagged artifacts).

_lwv4_repo_root() {
  local script_dir="${1:?}"
  cd "${script_dir}/../../.." && pwd
}

_lwv4_source_env() {
  local repo_root="${1:?}"
  if [[ -f "${repo_root}/configs/lwv4_baseline.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${repo_root}/configs/lwv4_baseline.env"
    set +a
  else
    echo "[lwv4][error] missing configs/lwv4_baseline.env" >&2
    return 1
  fi
}

_lwv4_resolve_gpu() {
  local repo_root="${1:?}"
  local gpu_arg="${2:-}"
  if [[ -n "${gpu_arg}" ]]; then
    export CUDA_DEVICES="${gpu_arg}"
    echo "[lwv4][cuda] CUDA_VISIBLE_DEVICES <- ${CUDA_DEVICES} (script arg)" >&2
    return 0
  fi
  # shellcheck disable=SC1091
  source "${repo_root}/configs/cuda_resolve.inc.sh"
  cuda_resolve_devices ""
}

_lwv4_gc_flag() {
  if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
    echo "--gradient_checkpointing"
  else
    echo "--no-gradient_checkpointing"
  fi
}

_lwv4_trust_flag() {
  if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    echo "--trust_remote_code"
  fi
}

# Usage: lwv4_train <agg_type> <client_state_suffix> [extra python args...]
lwv4_train() {
  local agg_type="${1:?}"
  local state_suffix="${2:?}"
  shift 2
  local gc_flag
  gc_flag="$(_lwv4_gc_flag)"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  python tasks/fed_train_sft_v4.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --agg_type "$agg_type" \
    --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
    ${gc_flag} \
    $(_lwv4_trust_flag) \
    --client_state_dir "${CLIENT_STATE_DIR}_${state_suffix}" --save_client_state_to_disk \
    --metrics_output_dir "$METRICS_OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --log_filename_prefix "$LOG_FILENAME_PREFIX" \
    --trained_models_root "$TRAINED_MODELS_ROOT" \
    --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
    --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA" \
    "$@"
}
