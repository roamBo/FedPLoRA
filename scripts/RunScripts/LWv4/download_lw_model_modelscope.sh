#!/usr/bin/env bash
# Download lightweight **base** (non-Instruct) backbone via ModelScope.
# LW SFT uses domain prompt/response JSONL; no chat-tuned checkpoint needed.
#
# Usage: bash scripts/RunScripts/LWv4/download_lw_model_modelscope.sh
# Override: MODELSCOPE_ID=... MODEL_PATH=... bash ...
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
cd "${_REPO_ROOT}"
# shellcheck disable=SC1091
source "${_REPO_ROOT}/configs/lwv4_baseline.env"

# ModelScope: https://www.modelscope.cn/models/HuggingFaceTB/SmolLM2-135M
MODELSCOPE_ID="${MODELSCOPE_ID:-HuggingFaceTB/SmolLM2-135M}"
MODEL_ROOT="${MODEL_ROOT:-/data2/minghao/model}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/SmolLM2-135M}"
MODEL_PARENT="$(dirname "${MODEL_PATH}")"
mkdir -p "${MODEL_PARENT}"

if [[ -f "${MODEL_PATH}/config.json" ]] && compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null; then
  echo "[lwv4][model] already exists: ${MODEL_PATH}"
  exit 0
fi
if [[ -f "${MODEL_PATH}/config.json" ]] && compgen -G "${MODEL_PATH}/pytorch_model*.bin" >/dev/null; then
  echo "[lwv4][model] already exists: ${MODEL_PATH}"
  exit 0
fi

if ! command -v modelscope >/dev/null 2>&1; then
  echo "[lwv4][model] installing modelscope CLI ..."
  pip install -U modelscope
fi

echo "[lwv4][model] downloading ${MODELSCOPE_ID} -> ${MODEL_PATH}"
modelscope download --model "${MODELSCOPE_ID}" --local_dir "${MODEL_PATH}"
echo "[lwv4][model] done: ${MODEL_PATH}"
