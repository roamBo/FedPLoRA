#!/usr/bin/env bash
# 【通信-性能实验】仅打印各 agg_type 的单轮上下行字节估计（不启动训练）
#
# Usage:
#   source configs/domain_sft_baselines.env
#   bash scripts/RunScripts/run_exp_comm_profile.sh
#
# 可选：AGG_LIST=normal,fedplora,ffa  bash scripts/RunScripts/run_exp_comm_profile.sh

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

MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}"
AGG_LIST="${AGG_LIST:-normal,fedex,ffa,fedplora,fedplora-oneshot,yoco,fedsa_lora,fedalt,flora}"

CMD=(
  python scripts/RunScripts/print_sft_comm_profile.py
  --model "${MODEL_PATH}"
  --lora_r "${LORA_R:-8}"
  --lora_alpha "${LORA_ALPHA:-16}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --target_modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj}"
  --agg_types "${AGG_LIST}"
)
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi

echo "[exp_comm_profile] model=${MODEL_PATH}"
"${CMD[@]}"
