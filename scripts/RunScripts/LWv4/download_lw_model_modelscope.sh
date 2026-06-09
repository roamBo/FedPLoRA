#!/usr/bin/env bash
# Download lightweight backbone via ModelScope (same parent dir as Llama-8B).
#
# Usage: bash scripts/RunScripts/LWv4/download_lw_model_modelscope.sh
# Override: MODELSCOPE_ID=... MODEL_PATH=... bash ...
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
cd "${_REPO_ROOT}"
# shellcheck disable=SC1091
source "${_REPO_ROOT}/configs/lwv4_baseline.env"

MODELSCOPE_ID="${MODELSCOPE_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/data/yaominghao/gb/models/Qwen2.5-0.5B-Instruct}"
MODEL_PARENT="$(dirname "${MODEL_PATH}")"
mkdir -p "${MODEL_PARENT}"

if [[ -f "${MODEL_PATH}/config.json" ]]; then
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
