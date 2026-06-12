#!/usr/bin/env bash
# Standard (non-cross-domain) SFT baselines on Alpaca + Llama.
# Runs all comparison methods EXCEPT FedPLoRA (fedplora / fedplora-oneshot / v3 / v4).
#
# Methods: normal, flora, flexlora, feddat, fedalt, yoco, fedsa_lora, ffa
#
# Usage (repo root):
#   bash scripts/RunScripts/run_standard_sft_baselines.sh [gpu]
#
# Prerequisite:
#   bash scripts/DataProcessScripts/build_alpaca_standard_benchmark_noniid.sh 10 0.5
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

if [[ -f "${_REPO_ROOT}/configs/standard_sft.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/standard_sft.env"
  set +a
fi

GPU_CLI="${1:-}"
# shellcheck disable=SC1091
source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
cuda_resolve_devices "${GPU_CLI}"

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_run_standard_sft_batch.inc.sh"

METHODS=(
  normal
  flora
  flexlora
  feddat
  fedalt
  yoco
  fedsa_lora
  ffa
)

echo "[standard_sft] model=${MODEL_PATH:-} benchmark=${BENCHMARK_DIR:-} methods=${#METHODS[@]}"

standard_sft_run_batch "standard_baselines" "${METHODS[@]}"

echo "[standard_sft] done. Metrics: ${METRICS_OUTPUT_DIR:-artifacts_standard/sft_metrics}/"
