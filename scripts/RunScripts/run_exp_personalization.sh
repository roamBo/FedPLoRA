#!/usr/bin/env bash
# 【个性化收益分析】跑 fedplora 与 normal，开启 --eval_personalization_metrics
# 读 artifacts_{N}c/sft_metrics/*.json 中的 client_local_macro_loss vs off_domain_macro_loss（N=客户端数）
#
# Usage (repo root):
#   source configs/domain_sft_baselines.env   # 可选
#   bash scripts/RunScripts/run_exp_personalization.sh [7|14|21|35]
#
# 环境变量可覆盖：MODEL_PATH, CUDA_DEVICES, ROUNDS, BENCHMARK_ROOT（默认 data/domain_benchmark_${N}c）

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

COMMON=(
  python tasks/fed_train_sft.py
  --model "${MODEL_PATH}"
  --benchmark_dir "${BENCHMARK_DIR}"
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
  --eval_personalization_metrics
)

echo "[exp_personalization] benchmark_dir=${BENCHMARK_DIR}"

echo "[run] fedplora + personalization metrics"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${COMMON[@]}" \
  --agg_type fedplora \
  --save_client_state_to_disk

echo "[run] normal + personalization metrics"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${COMMON[@]}" \
  --agg_type normal
