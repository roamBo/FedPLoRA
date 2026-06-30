#!/usr/bin/env bash
# FedPLoRA v6 / DCR on LW7c (SmolLM2-135M by default).
#
# Usage:
#   bash scripts/RunScripts/run_v6_dcr_lw7c.sh [gpu] [agg_type ...]
#
# Examples:
#   bash scripts/RunScripts/run_v6_dcr_lw7c.sh 0 v6_dcr_global
#   bash scripts/RunScripts/run_v6_dcr_lw7c.sh 1 v6_dcr_domain
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

_USER_METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-}"
_USER_CLIENT_STATE_DIR="${CLIENT_STATE_DIR:-}"

if [[ -f "${_REPO_ROOT}/configs/lwv4_baseline.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/lwv4_baseline.env"
  set +a
fi

GPU_CLI="${1:-}"
if [[ -f "${_REPO_ROOT}/configs/cuda_resolve.inc.sh" && -z "${GPU_CLI}" ]]; then
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
  cuda_resolve_devices ""
else
  export CUDA_DEVICES="${GPU_CLI:-0}"
fi
shift || true

MODEL_ROOT="${MODEL_ROOT:-/data2/minghao/model}"
CODE_ROOT="${CODE_ROOT:-/home/minghao/code/FedPLoRA-main}"
DATA_ROOT="${DATA_ROOT:-$CODE_ROOT/data}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/SmolLM2-135M}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$DATA_ROOT/domain_benchmark_LW7c/seed_42}"
ROUNDS="${ROUNDS:-1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
SEED="${SEED:-42}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-10}"
TRAINED_MODELS_ROOT="${TRAINED_MODELS_ROOT:-$MODEL_ROOT/trained_models_LW}"
METRICS_OUTPUT_DIR="${_USER_METRICS_OUTPUT_DIR:-artifacts_LW7c/sft_metrics_v6}"
CLIENT_STATE_DIR="${_USER_CLIENT_STATE_DIR:-artifacts_LW7c/v6_client_states}"
YOCO_SPARSE_LAMBDA="${YOCO_SPARSE_LAMBDA:-1e-4}"
ONESHOT_ANCHOR_LAMBDA="${ONESHOT_ANCHOR_LAMBDA:-1e-4}"

METHODS=("$@")
if [[ ${#METHODS[@]} -eq 0 ]]; then
  METHODS=(v6_dcr_global v6_dcr_domain)
fi

mkdir -p log_LWv6 "${METRICS_OUTPUT_DIR}"

for AGG in "${METHODS[@]}"; do
  echo "[v6-dcr][LW7c] agg_type=${AGG} gpu=${CUDA_DEVICES}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  python -u tasks/fed_train_sft.py \
    --model "${MODEL_PATH}" \
    --benchmark_dir "${BENCHMARK_DIR}" \
    --agg_type "${AGG}" \
    --rounds "${ROUNDS}" --local_epochs "${LOCAL_EPOCHS}" --lr "${LR}" \
    --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
    --batch_size "${BATCH_SIZE}" --max_seq_length "${MAX_SEQ_LENGTH}" \
    --torch_dtype "${TORCH_DTYPE}" --target_modules "${TARGET_MODULES}" \
    --client_state_dir "${CLIENT_STATE_DIR}_${AGG}" --save_client_state_to_disk \
    --metrics_output_dir "${METRICS_OUTPUT_DIR}" \
    --trained_models_root "${TRAINED_MODELS_ROOT}" \
    --eval_max_batches "${EVAL_MAX_BATCHES}" --seed "${SEED}" \
    --gradient_checkpointing \
    --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA}" \
    --oneshot_anchor_lambda "${ONESHOT_ANCHOR_LAMBDA}" \
    --v6_dcr_rc_policy "${V6_DCR_RC_POLICY:-auto}" \
    --v6_dcr_energy_tau "${V6_DCR_ENERGY_TAU:-0.80}" \
    --v6_dcr_conflict_strength "${V6_DCR_CONFLICT_STRENGTH:-1.0}" \
    --v6_dcr_importance_power "${V6_DCR_IMPORTANCE_POWER:-0.0}" \
    --v6_dcr_importance_clip "${V6_DCR_IMPORTANCE_CLIP:-5.0}"
done

echo "[v6-dcr][LW7c] done. metrics -> ${METRICS_OUTPUT_DIR}/"
