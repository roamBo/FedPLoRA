#!/usr/bin/env bash
# FedPLoRA-v5 merge family on the 35c main benchmark (Llama-3.1-8B).
# Goes through tasks/fed_train_sft.py (v5m_* is a memory-global-agg method like flora).
#
# Usage (from repo root):
#   bash scripts/RunScripts/run_v5_merge_35c.sh [7|14|21|35] [gpu]
#
# Methods (comm-matched to flora/flexlora at fixed rank):
#   v5m_mean        weighted ΔW mean + rank-r SVD (sanity == flexlora protocol)
#   v5m_ties        interference-aware TIES on ΔW
#   v5m_dare_ties   DARE drop+rescale then TIES
#   v5m_knots_ties  subspace-aligned TIES (flagship; fully factored)
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

if [[ -f "${_REPO_ROOT}/configs/domain_sft.env" ]]; then
  set -a; source "${_REPO_ROOT}/configs/domain_sft.env"; set +a
fi

NC="${1:-35}"
GPU_CLI="${2:-}"
case "${NC}" in
  7|14|21|35) BENCHMARK_DIR="data/domain_benchmark_${NC}c/seed_42" ;;
  *) echo "Usage: $0 [7|14|21|35] [gpu]" >&2; exit 1 ;;
esac

if [[ -f "${_REPO_ROOT}/configs/cuda_resolve.inc.sh" ]]; then
  source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
  cuda_resolve_devices "${GPU_CLI}"
else
  export CUDA_DEVICES="${GPU_CLI:-0}"
fi

MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
LORA_R="${LORA_R:-8}"; LORA_ALPHA="${LORA_ALPHA:-16}"; LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-2}"; MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
LR="${LR:-2e-4}"; ROUNDS="${ROUNDS:-1}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"; SEED="${SEED:-42}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"
METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-artifacts_${NC}c/sft_metrics}"
CLIENT_STATE_DIR="${CLIENT_STATE_DIR:-artifacts_${NC}c/domain_client_states}"
KEEP_RATIO="${V5M_KEEP_RATIO:-0.2}"
DARE_P="${V5M_DARE_P:-0.3}"
RANK_POLICY="${V5M_RANK_POLICY:-fixed}"   # fixed = comm-matched; energy = adaptive-rank ablation
RANK_CAP="${V5M_RANK_CAP:-32}"
ENERGY_TAU="${V5M_ENERGY_TAU:-0.95}"

METHODS=("${@:3}")
if [[ ${#METHODS[@]} -eq 0 ]]; then
  METHODS=(v5m_mean v5m_ties v5m_dare_ties v5m_knots_ties)
fi

echo "[v5m][35c] clients=${NC} model=${MODEL_PATH} methods=${METHODS[*]} rank_policy=${RANK_POLICY}"

for AGG in "${METHODS[@]}"; do
  echo "[v5m] === agg_type=${AGG} ==="
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
    --eval_max_batches "${EVAL_MAX_BATCHES}" --seed "${SEED}" \
    --gradient_checkpointing \
    --v5m_keep_ratio "${KEEP_RATIO}" \
    --v5m_dare_p "${DARE_P}" \
    --v5m_rank_policy "${RANK_POLICY}" \
    --v5m_rank_cap "${RANK_CAP}" \
    --v5m_energy_tau "${ENERGY_TAU}"
done

echo "[v5m][35c] done. metrics -> ${METRICS_OUTPUT_DIR}/"
