#!/usr/bin/env bash
# Baseline batch 1: FedP-LoRA-oneshot, FedALT, Flora, FedAvg-Normal
# Maps to agg_type: fedplora-oneshot, fedalt, flora, normal
#
# Usage (from repo root):
#   source configs/domain_sft_baselines.env
#   bash scripts/RunScripts/run_domain_sft_baselines1.sh [7|14|21|35]
#
# First argument selects data/domain_benchmark_<N>c/seed_42 (default: 35).

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

if [[ -f "${_REPO_ROOT}/configs/domain_sft_baselines.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/domain_sft_baselines.env"
  set +a
fi

usage() {
  echo "Usage: $0 [7|14|21|35]" >&2
  echo "  Uses data/domain_benchmark_<N>c/seed_42 (default: 35)." >&2
  exit 1
}

NC="${1:-35}"
case "${NC}" in
  7|14|21|35) BENCHMARK_DIR="data/domain_benchmark_${NC}c/seed_42" ;;
  *) usage ;;
esac

MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
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

METHODS=("fedplora-oneshot" "fedalt" "flora" "normal")

echo "[run_domain_sft_baselines1] clients=${NC} benchmark_dir=${BENCHMARK_DIR}"
echo "[run_domain_sft_baselines1] methods: FedP-LoRA-oneshot | FedALT | Flora | FedAvg-Normal"

for AGG_TYPE in "${METHODS[@]}"; do
  echo "[run] agg_type=${AGG_TYPE}"
  CMD=(
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
    --gradient_checkpointing
  )

  case "${AGG_TYPE}" in
    fedplora|fedplora-oneshot|fedalt|fedsa_lora|fedsa|yoco)
      CMD+=(--save_client_state_to_disk)
      ;;
  esac

  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    CMD+=(--trust_remote_code)
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CMD[@]}"
done
