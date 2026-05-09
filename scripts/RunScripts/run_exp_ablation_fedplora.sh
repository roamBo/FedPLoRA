#!/usr/bin/env bash
# 【机制消融】多轮 fedplora：逐项关闭正则或服务端共识 / 动量
# 产物：artifacts_{N}c/sft_metrics/* 与各 run 日志（N=客户端数）
#
# Usage:
#   source configs/domain_sft_baselines.env
#   bash scripts/RunScripts/run_exp_ablation_fedplora.sh [7|14|21|35]
#
# 环境变量 ABLATION_MODE 可选：full | no_align | no_prox | no_orth | no_consensus | no_momentum
# 若未设置 ABLATION_MODE，则按顺序跑全部配置。

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

NC="${1:-35}"
case "${NC}" in
  7|14|21|35) ;;
  *) echo "Usage: $0 [7|14|21|35]" >&2; exit 1 ;;
esac

BENCHMARK_DIR="${BENCHMARK_DIR:-data/domain_benchmark_${NC}c/seed_42}"
MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
ROUNDS="${ROUNDS:-10}"

AL="${GP_ALIGN_LAMBDA:-0.01}"
PL="${GP_PROX_LAMBDA:-0.001}"
OL="${GP_ORTH_LAMBDA:-0.0001}"
CP="${GP_CONSENSUS_POWER:-2.0}"
AM="${GP_AGG_MOMENTUM:-0.5}"

BASE=(
  python tasks/fed_train_sft.py
  --model "${MODEL_PATH}"
  --benchmark_dir "${BENCHMARK_DIR}"
  --agg_type fedplora
  --rounds "${ROUNDS}"
  --local_epochs "${LOCAL_EPOCHS:-1}"
  --lr "${LR:-2e-4}"
  --lora_r "${LORA_R:-8}"
  --lora_alpha "${LORA_ALPHA:-16}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --batch_size "${BATCH_SIZE:-2}"
  --max_seq_length "${MAX_SEQ_LENGTH:-2048}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --target_modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
  --client_state_dir "${CLIENT_STATE_DIR:-artifacts/domain_client_states}"
  --gradient_checkpointing
  --save_client_state_to_disk
)

run_one() {
  local tag="$1"
  shift
  echo "[ablation] ${tag}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${BASE[@]}" "$@"
}

if [[ -n "${ABLATION_MODE:-}" ]]; then
  MODES="${ABLATION_MODE}"
else
  MODES="full no_align no_prox no_orth no_consensus no_momentum"
fi

echo "[exp_ablation_fedplora] benchmark_dir=${BENCHMARK_DIR}"

for mode in ${MODES}; do
  case "${mode}" in
    full)
      run_one "full" \
        --gp_align_lambda "${AL}" --gp_prox_lambda "${PL}" --gp_orth_lambda "${OL}" \
        --gp_consensus_power "${CP}" --gp_agg_momentum "${AM}" ;;
    no_align)
      run_one "no_align" \
        --gp_align_lambda 0 --gp_prox_lambda "${PL}" --gp_orth_lambda "${OL}" \
        --gp_consensus_power "${CP}" --gp_agg_momentum "${AM}" ;;
    no_prox)
      run_one "no_prox" \
        --gp_align_lambda "${AL}" --gp_prox_lambda 0 --gp_orth_lambda "${OL}" \
        --gp_consensus_power "${CP}" --gp_agg_momentum "${AM}" ;;
    no_orth)
      run_one "no_orth" \
        --gp_align_lambda "${AL}" --gp_prox_lambda "${PL}" --gp_orth_lambda 0 \
        --gp_consensus_power "${CP}" --gp_agg_momentum "${AM}" ;;
    no_consensus)
      run_one "no_consensus" \
        --gp_align_lambda "${AL}" --gp_prox_lambda "${PL}" --gp_orth_lambda "${OL}" \
        --gp_consensus_power "${CP}" --gp_agg_momentum "${AM}" \
        --fedplora_ablation_no_consensus ;;
    no_momentum)
      run_one "no_momentum" \
        --gp_align_lambda "${AL}" --gp_prox_lambda "${PL}" --gp_orth_lambda "${OL}" \
        --gp_consensus_power "${CP}" --gp_agg_momentum "${AM}" \
        --fedplora_ablation_no_momentum ;;
    *)
      echo "Unknown ABLATION_MODE=${mode}" >&2
      exit 1
      ;;
  esac
done
