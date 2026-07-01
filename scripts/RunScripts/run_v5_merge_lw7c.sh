#!/usr/bin/env bash
# FedPLoRA-v5 merge family on the LW7c lightweight benchmark (SmolLM2-135M).
# Fast mechanism screen before the 35c main run. Goes through tasks/fed_train_sft.py.
#
# Usage (from repo root):
#   bash scripts/RunScripts/run_v5_merge_lw7c.sh [gpu] [agg_type ...]
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

if [[ -f "${_REPO_ROOT}/configs/lwv4_baseline.env" ]]; then
  set -a; source "${_REPO_ROOT}/configs/lwv4_baseline.env"; set +a
fi

GPU_CLI="${1:-}"
if [[ -f "${_REPO_ROOT}/configs/cuda_resolve.inc.sh" && -z "${GPU_CLI}" ]]; then
  source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
  cuda_resolve_devices ""
else
  export CUDA_DEVICES="${GPU_CLI:-0}"
fi

MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/SmolLM2-135M}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/domain_benchmark_LW7c/seed_42}"
LORA_R="${LORA_R:-8}"; LORA_ALPHA="${LORA_ALPHA:-16}"; LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-2}"; MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
LR="${LR:-2e-4}"; ROUNDS="${ROUNDS:-1}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"; SEED="${SEED:-42}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-10}"
METRICS_OUTPUT_DIR="${METRICS_BASELINE_OUTPUT_DIR:-artifacts_LW7c/sft_metrics}"
CLIENT_STATE_DIR="${CLIENT_STATE_DIR:-artifacts_LW7c/v5_client_states}"
TRAINED_MODELS_ROOT="${TRAINED_MODELS_ROOT:-../trained_models_LW}"
KEEP_RATIO="${V5M_KEEP_RATIO:-0.2}"
DARE_P="${V5M_DARE_P:-0.3}"
RANK_POLICY="${V5M_RANK_POLICY:-fixed}"
RANK_CAP="${V5M_RANK_CAP:-32}"
ENERGY_TAU="${V5M_ENERGY_TAU:-0.95}"

GC_FLAG="--gradient_checkpointing"
[[ "${GRADIENT_CHECKPOINTING:-1}" == "0" ]] && GC_FLAG="--no-gradient_checkpointing"

shift || true
METHODS=("$@")
if [[ ${#METHODS[@]} -eq 0 ]]; then
  METHODS=(v5m_mean v5m_ties v5m_dare_ties v5m_knots_ties)
fi

echo "[v5m][LW7c] model=${MODEL_PATH} methods=${METHODS[*]} rank_policy=${RANK_POLICY}"

for AGG in "${METHODS[@]}"; do
  echo "[v5m][LW7c] === agg_type=${AGG} ==="
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  python tasks/fed_train_sft.py \
    --model "${MODEL_PATH}" \
    --benchmark_dir "${BENCHMARK_DIR}" \
    --agg_type "${AGG}" \
    --rounds "${ROUNDS}" --local_epochs "${LOCAL_EPOCHS}" --lr "${LR}" \
    --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
    --batch_size "${BATCH_SIZE}" --max_seq_length "${MAX_SEQ_LENGTH}" \
    --torch_dtype "${TORCH_DTYPE}" --target_modules "${TARGET_MODULES}" \
    --client_state_dir "${CLIENT_STATE_DIR}" \
    --metrics_output_dir "${METRICS_OUTPUT_DIR}" \
    --trained_models_root "${TRAINED_MODELS_ROOT}" \
    --eval_max_batches "${EVAL_MAX_BATCHES}" --seed "${SEED}" \
    ${GC_FLAG} \
    --v5m_keep_ratio "${KEEP_RATIO}" \
    --v5m_dare_p "${DARE_P}" \
    --v5m_rank_policy "${RANK_POLICY}" \
    --v5m_rank_cap "${RANK_CAP}" \
    --v5m_energy_tau "${ENERGY_TAU}"
done

echo "[v5m][LW7c] done. metrics -> ${METRICS_OUTPUT_DIR}/"
