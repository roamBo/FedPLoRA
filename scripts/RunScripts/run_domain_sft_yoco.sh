#!/usr/bin/env bash
# Single method: yoco (FedMLLM conflict one-shot). Speed flags via _fed_train_speed.inc.sh / env.
#
# Usage (from repo root):
#   bash scripts/RunScripts/run_domain_sft_yoco.sh [7|14|21|35] [gpu]

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

if [[ -f "${_REPO_ROOT}/configs/domain_sft.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/domain_sft.env"
  set +a
fi

usage() {
  echo "Usage: $0 [7|14|21|35] [gpu]" >&2
  exit 1
}

NC="${1:-35}"
GPU_CLI="${2:-}"
case "${NC}" in
  7|14|21|35) BENCHMARK_DIR="data/domain_benchmark_${NC}c/seed_42" ;;
  *) usage ;;
esac

# shellcheck disable=SC1091
source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
cuda_resolve_devices "${GPU_CLI}"

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_fed_train_speed.inc.sh"

MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
ROUNDS="${ROUNDS:-1}"
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
YOCO_SPARSE_LAMBDA="${YOCO_SPARSE_LAMBDA:-1e-4}"
YOCO_AGGREGATE_MODE="${YOCO_AGGREGATE_MODE:-conflict}"
YOCO_CONFLICT_METHOD="${YOCO_CONFLICT_METHOD:-avgm}"
YOCO_SIGN_LAMBDA="${YOCO_SIGN_LAMBDA:-0.01}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"

AGG_TYPE="yoco"
echo "[run_domain_sft_yoco] clients=${NC} benchmark_dir=${BENCHMARK_DIR}"

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
  --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA}"
  --yoco_aggregate_mode "${YOCO_AGGREGATE_MODE}"
  --yoco_conflict_method "${YOCO_CONFLICT_METHOD}"
  --yoco_sign_lambda "${YOCO_SIGN_LAMBDA}"
)
if [[ -n "${EVAL_MAX_BATCHES}" && "${EVAL_MAX_BATCHES}" != "0" ]]; then
  CMD+=(--eval_max_batches "${EVAL_MAX_BATCHES}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi

fed_train_append_speed_flags CMD

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CMD[@]}"
