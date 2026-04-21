#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/domain_benchmark/seed_42}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
ROUNDS="${ROUNDS:-10}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
CLIENT_STATE_DIR="${CLIENT_STATE_DIR:-artifacts/domain_client_states}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

METHODS=("gp_lora" "normal" "ffa" "fedex")

for AGG_TYPE in "${METHODS[@]}"; do
  echo "[run] agg_type=${AGG_TYPE}"
  CMD=(
    python fed_train_sft.py
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
    --gradient_checkpointing
  )

  if [[ "${AGG_TYPE}" == "gp_lora" ]]; then
    CMD+=(--save_client_state_to_disk)
  fi

  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    CMD+=(--trust_remote_code)
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CMD[@]}"
done
