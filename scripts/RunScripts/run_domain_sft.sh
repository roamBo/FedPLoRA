#!/usr/bin/env bash
# Optional first argument: GPU index or list. If omitted, auto-pick or fallback; see configs/cuda_resolve.inc.sh.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"
if [[ -f "${_REPO_ROOT}/configs/domain_sft_pilot.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/domain_sft_pilot.env"
  set +a
fi

GPU_CLI="${1:-}"
# shellcheck disable=SC1091
source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
cuda_resolve_devices "${GPU_CLI}"

MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/domain_benchmark/seed_42}"
AGG_TYPE="${AGG_TYPE:-fedplora}"
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
SAVE_CLIENT_STATE_TO_DISK="${SAVE_CLIENT_STATE_TO_DISK:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
CLIENT_STATE_DIR="${CLIENT_STATE_DIR:-artifacts/domain_client_states}"

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

if [[ "${SAVE_CLIENT_STATE_TO_DISK}" == "1" ]]; then
  CMD+=(--save_client_state_to_disk)
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CMD[@]}"
