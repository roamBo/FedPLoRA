# shellcheck shell=bash
# Shared loop for domain SFT batch scripts. Source after:
#   cd repo root, source configs/domain_sft.env, cuda_resolve.inc.sh, _fed_train_speed.inc.sh
#
# Requires: NC, BENCHMARK_DIR, CUDA_DEVICES; cwd = repo root.
# Uses: SEED (default 42), MODEL_PATH, ROUNDS, LR, LoRA/hyper vars from env.
# Run checkpoints default inside Python to <repo>/../trained_models/<stem>/ (no --save_run_checkpoint_dir).
# Optional: export TRAINED_MODELS_ROOT=/abs/path to override the parent directory.

domain_sft_run_batch() {
  local batch_tag="$1"
  shift
  local -a _methods=( "$@" )

  local MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
  local ROUNDS="${ROUNDS:-10}"
  local LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
  local LR="${LR:-2e-4}"
  local LORA_R="${LORA_R:-8}"
  local LORA_ALPHA="${LORA_ALPHA:-16}"
  local LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
  local BATCH_SIZE="${BATCH_SIZE:-2}"
  local MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
  local TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
  local TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
  local CLIENT_STATE_DIR="${CLIENT_STATE_DIR:-artifacts/domain_client_states}"
  local TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
  local YOCO_SPARSE_LAMBDA="${YOCO_SPARSE_LAMBDA:-1e-4}"
  local YOCO_PCWA_COMPONENTS="${YOCO_PCWA_COMPONENTS:-3}"
  local ONESHOT_ANCHOR_LAMBDA="${ONESHOT_ANCHOR_LAMBDA:-1e-4}"
  local ONESHOT_PROX_LAMBDA="${ONESHOT_PROX_LAMBDA:-0.0}"
  local ONESHOT_CONSENSUS_POWER="${ONESHOT_CONSENSUS_POWER:-2.0}"
  local ONESHOT_IMPORTANCE_POWER="${ONESHOT_IMPORTANCE_POWER:-1.0}"
  local ONESHOT_IMPORTANCE_CLIP="${ONESHOT_IMPORTANCE_CLIP:-5.0}"
  local ONESHOT_CONFLICT_THRESHOLD="${ONESHOT_CONFLICT_THRESHOLD:-0.35}"
  local ONESHOT_CONFLICT_BLEND="${ONESHOT_CONFLICT_BLEND:-1.0}"
  local ONESHOT_SCALE_CLIP_RATIO="${ONESHOT_SCALE_CLIP_RATIO:-0.0}"
  local ONESHOT_NO_KEEP_INIT_ON_CONFLICT="${ONESHOT_NO_KEEP_INIT_ON_CONFLICT:-0}"
  local ONESHOT_ORTHOGONALIZE="${ONESHOT_ORTHOGONALIZE:-0}"
  local EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"
  local SEED="${SEED:-42}"

  echo "[${batch_tag}] clients=${NC} benchmark_dir=${BENCHMARK_DIR} trained_models_root=${TRAINED_MODELS_ROOT:-<default ../trained_models>} seed=${SEED}"

  for AGG_TYPE in "${_methods[@]}"; do
    echo "[run] agg_type=${AGG_TYPE} (auto checkpoint path in Python under trained_models/)"

    local CMD=(
      python tasks/fed_train_sft.py
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
      --seed "${SEED}"
    )
    if [[ -n "${TRAINED_MODELS_ROOT:-}" ]]; then
      CMD+=(--trained_models_root "${TRAINED_MODELS_ROOT}")
    fi
    # 默认开启以省显存；显存充足时 export GRADIENT_CHECKPOINTING=0 可明显加速单 step。
    if [[ "${GRADIENT_CHECKPOINTING:-1}" != "0" ]]; then
      CMD+=(--gradient_checkpointing)
    fi
    if [[ -n "${EVAL_MAX_BATCHES}" && "${EVAL_MAX_BATCHES}" != "0" ]]; then
      CMD+=(--eval_max_batches "${EVAL_MAX_BATCHES}")
    fi

    case "${AGG_TYPE}" in
      fedplora|fedplora-oneshot|fedalt|fedsa_lora|fedsa|yoco)
        CMD+=(--save_client_state_to_disk)
        ;;
    esac

    if [[ "${AGG_TYPE}" == "fedplora-oneshot" ]]; then
      CMD+=(
        --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA}"
        --oneshot_anchor_lambda "${ONESHOT_ANCHOR_LAMBDA}"
        --oneshot_prox_lambda "${ONESHOT_PROX_LAMBDA}"
        --oneshot_consensus_power "${ONESHOT_CONSENSUS_POWER}"
        --oneshot_importance_power "${ONESHOT_IMPORTANCE_POWER}"
        --oneshot_importance_clip "${ONESHOT_IMPORTANCE_CLIP}"
        --oneshot_conflict_threshold "${ONESHOT_CONFLICT_THRESHOLD}"
        --oneshot_conflict_blend "${ONESHOT_CONFLICT_BLEND}"
        --oneshot_scale_clip_ratio "${ONESHOT_SCALE_CLIP_RATIO}"
      )
      if [[ "${ONESHOT_NO_KEEP_INIT_ON_CONFLICT}" == "1" ]]; then
        CMD+=(--oneshot_no_keep_init_on_conflict)
      fi
      if [[ "${ONESHOT_ORTHOGONALIZE}" == "1" ]]; then
        CMD+=(--oneshot_orthogonalize)
      fi
    fi

    if [[ "${AGG_TYPE}" == "yoco" ]]; then
      CMD+=(
        --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA}"
        --yoco_pcwa_components "${YOCO_PCWA_COMPONENTS}"
      )
    fi

    if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
      CMD+=(--trust_remote_code)
    fi

    fed_train_append_speed_flags CMD

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CMD[@]}"
  done
}
