# shellcheck shell=bash
# Shared helpers for LW v4 RunScripts (7 clients, SmolLM2-135M base default, LW-tagged artifacts).

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

# Classic SFT baselines via tasks/fed_train_sft.py (normal, yoco, fedalt, ffa, …).
# Requires _LWv4_RUNSCRIPTS_DIR = scripts/RunScripts (set by run_lwv4_baseline.sh).
lwv4_train_baseline() {
  local agg_type="${1:?}"
  local rs_dir="${_LWv4_RUNSCRIPTS_DIR:?missing _LWv4_RUNSCRIPTS_DIR}"
  local metrics_dir="${METRICS_BASELINE_OUTPUT_DIR:-artifacts_LW7c/sft_metrics}"
  local seed="${SEED:-42}"

  # shellcheck disable=SC1091
  source "${rs_dir}/_fed_train_speed.inc.sh"
  # shellcheck disable=SC1091
  source "${rs_dir}/_run_domain_sft_batch.inc.sh"

  local -a cmd=(
    python tasks/fed_train_sft.py
    --model "${MODEL_PATH}"
    --benchmark_dir "${BENCHMARK_DIR}"
    --agg_type "${agg_type}"
    --rounds "${ROUNDS}"
    --local_epochs "${LOCAL_EPOCHS}"
    --lr "${LR}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --batch_size "${BATCH_SIZE}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --torch_dtype "${TORCH_DTYPE}"
    --target_modules "${TARGET_MODULES}"
    --client_state_dir "${CLIENT_STATE_DIR}"
    --metrics_output_dir "${metrics_dir}"
    --seed "${seed}"
  )

  if [[ -n "${TRAINED_MODELS_ROOT:-}" ]]; then
    cmd+=(--trained_models_root "${TRAINED_MODELS_ROOT}")
  fi
  if [[ "${GRADIENT_CHECKPOINTING:-1}" != "0" ]]; then
    cmd+=(--gradient_checkpointing)
  fi
  if [[ -n "${EVAL_MAX_BATCHES:-}" && "${EVAL_MAX_BATCHES}" != "0" ]]; then
    cmd+=(--eval_max_batches "${EVAL_MAX_BATCHES}")
  fi
  if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    cmd+=(--trust_remote_code)
  fi

  case "${agg_type}" in
    fedplora|fedplora-oneshot|fedplora_v3_lite|fedplora_v3_cluster|fedplora_v3_rpca|v3_lite|v3_cluster|v3_rpca|fedalt|fedsa_lora|fedsa|yoco)
      cmd+=(--save_client_state_to_disk)
      ;;
  esac

  case "${agg_type}" in
    fedplora-oneshot|fedplora_v3_lite|fedplora_v3_cluster|fedplora_v3_rpca|v3_lite|v3_cluster|v3_rpca)
      domain_sft_append_oneshot_v3_flags cmd "${agg_type}"
      ;;
  esac

  if [[ "${agg_type}" == "yoco" ]]; then
    cmd+=(
      --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA:-1e-4}"
      --yoco_pcwa_components "${YOCO_PCWA_COMPONENTS:-3}"
    )
  fi

  if [[ "${agg_type}" == "feddat" ]]; then
    cmd+=(--feddat_teacher_lambda "${FEDDAT_TEACHER_LAMBDA:-0.01}")
  fi

  fed_train_append_speed_flags cmd

  echo "[lwv4][baseline] agg_type=${agg_type} metrics=${metrics_dir}" >&2
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}"
}

# Standard Alpaca non-IID baselines via tasks/fed_train_standard_sft.py (LW env).
# Source configs/lw_standard_noniid.env before calling.
lwv4_train_standard_baseline() {
  local agg_type="${1:?}"
  local rs_dir="${_LWv4_RUNSCRIPTS_DIR:?missing _LWv4_RUNSCRIPTS_DIR}"
  local metrics_dir="${METRICS_OUTPUT_DIR:-artifacts_LW_standard/sft_metrics}"
  local seed="${SEED:-42}"
  local gc_flag
  gc_flag="$(_lwv4_gc_flag)"

  # shellcheck disable=SC1091
  source "${rs_dir}/_fed_train_speed.inc.sh"

  local -a cmd=(
    python tasks/fed_train_standard_sft.py
    --model "${MODEL_PATH}"
    --benchmark_dir "${BENCHMARK_DIR}"
    --agg_type "${agg_type}"
    --rounds "${ROUNDS}"
    --local_epochs "${LOCAL_EPOCHS}"
    --lr "${LR}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --batch_size "${BATCH_SIZE}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --torch_dtype "${TORCH_DTYPE}"
    --target_modules "${TARGET_MODULES}"
    --client_state_dir "${CLIENT_STATE_DIR}"
    --metrics_output_dir "${metrics_dir}"
    --seed "${seed}"
  )

  if [[ -n "${TRAINED_MODELS_ROOT:-}" ]]; then
    cmd+=("--trained_models_root" "${TRAINED_MODELS_ROOT}")
  fi
  # shellcheck disable=SC2086
  cmd+=(${gc_flag})
  if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    cmd+=(--trust_remote_code)
  fi
  if [[ -n "${EVAL_MAX_BATCHES:-}" && "${EVAL_MAX_BATCHES}" != "0" ]]; then
    cmd+=(--eval_max_batches "${EVAL_MAX_BATCHES}")
  fi

  case "${agg_type}" in
    fedalt|fedsa_lora|fedsa|yoco)
      cmd+=(--save_client_state_to_disk)
      ;;
  esac

  if [[ "${agg_type}" == "yoco" ]]; then
    cmd+=(
      --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA:-1e-4}"
      --yoco_pcwa_components "${YOCO_PCWA_COMPONENTS:-3}"
    )
  fi

  if [[ "${agg_type}" == "feddat" ]]; then
    cmd+=(--feddat_teacher_lambda "${FEDDAT_TEACHER_LAMBDA:-0.01}")
  fi

  fed_train_append_speed_flags cmd

  echo "[lw][standard][noniid] agg_type=${agg_type} benchmark=${BENCHMARK_DIR} metrics=${metrics_dir}" >&2
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}"
}
