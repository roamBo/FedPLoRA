# shellcheck shell=bash
# Shared loop for standard (non-cross-domain) SFT baselines.
# Source after: cd repo root, source configs/standard_sft.env, cuda_resolve.inc.sh

standard_sft_run_batch() {
  local batch_tag="$1"
  shift
  local -a _methods=( "$@" )

  local MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
  local BENCHMARK_DIR="${BENCHMARK_DIR:-data/standard_benchmark_alpaca/seed_42}"
  local ROUNDS="${ROUNDS:-1}"
  local LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
  local LR="${LR:-2e-4}"
  local LORA_R="${LORA_R:-8}"
  local LORA_ALPHA="${LORA_ALPHA:-16}"
  local LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
  local BATCH_SIZE="${BATCH_SIZE:-2}"
  local MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
  local TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
  local TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
  local CLIENT_STATE_DIR="${CLIENT_STATE_DIR:-artifacts_standard/client_states}"
  local METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-artifacts_standard/sft_metrics}"
  local TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
  local YOCO_SPARSE_LAMBDA="${YOCO_SPARSE_LAMBDA:-1e-4}"
  local YOCO_PCWA_COMPONENTS="${YOCO_PCWA_COMPONENTS:-3}"
  local FEDDAT_TEACHER_LAMBDA="${FEDDAT_TEACHER_LAMBDA:-0.01}"
  local EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"
  local SEED="${SEED:-42}"

  echo "[${batch_tag}] benchmark_dir=${BENCHMARK_DIR} rounds=${ROUNDS} seed=${SEED}"

  for AGG_TYPE in "${_methods[@]}"; do
    echo "[run] agg_type=${AGG_TYPE}"

    local -a cmd=(
      python tasks/fed_train_standard_sft.py
      --model "${MODEL_PATH}"
      --benchmark_dir "${BENCHMARK_DIR}"
      --agg_type "${AGG_TYPE}"
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
      --metrics_output_dir "${METRICS_OUTPUT_DIR}"
      --seed "${SEED}"
    )

    if [[ -n "${TRAINED_MODELS_ROOT:-}" ]]; then
      cmd+=(--trained_models_root "${TRAINED_MODELS_ROOT}")
    fi
    if [[ "${GRADIENT_CHECKPOINTING:-1}" != "0" ]]; then
      cmd+=(--gradient_checkpointing)
    fi
    if [[ -n "${EVAL_MAX_BATCHES}" && "${EVAL_MAX_BATCHES}" != "0" ]]; then
      cmd+=(--eval_max_batches "${EVAL_MAX_BATCHES}")
    fi
    if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
      cmd+=(--trust_remote_code)
    fi

    case "${AGG_TYPE}" in
      fedalt|fedsa_lora|fedsa|yoco)
        cmd+=(--save_client_state_to_disk)
        ;;
    esac

    if [[ "${AGG_TYPE}" == "yoco" ]]; then
      cmd+=(
        --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA}"
        --yoco_pcwa_components "${YOCO_PCWA_COMPONENTS}"
      )
    fi

    if [[ "${AGG_TYPE}" == "feddat" ]]; then
      cmd+=(--feddat_teacher_lambda "${FEDDAT_TEACHER_LAMBDA}")
    fi

    # shellcheck disable=SC1091
    source "${_SCRIPT_DIR}/_fed_train_speed.inc.sh"
    fed_train_append_speed_flags cmd

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}"
  done
}
