#!/usr/bin/env bash
# LW standard Alpaca Dirichlet non-IID (alpha=0.5) baselines — all methods except FedPLoRA.
# Uses tasks/fed_train_standard_sft.py + SmolLM2-135M (configs/lw_standard_noniid.env).
#
# Methods: normal, flora, flexlora, feddat, fedalt, yoco, fedsa_lora, ffa
#
# Prerequisite:
#   bash scripts/DataProcessScripts/build_alpaca_lw_standard_noniid_benchmark.sh
#   bash scripts/RunScripts/LWv4/download_lw_model_modelscope.sh
#
# Usage: bash scripts/RunScripts/LWv4/run_lw_standard_noniid_baseline.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
_LWv4_RUNSCRIPTS_DIR="$(cd "${_SCRIPT_DIR}/.." && pwd)"
export _LWv4_RUNSCRIPTS_DIR
cd "${_REPO_ROOT}"

if [[ -f "${_REPO_ROOT}/configs/lw_standard_noniid.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/lw_standard_noniid.env"
  set +a
else
  echo "[lw][standard][error] missing configs/lw_standard_noniid.env" >&2
  exit 1
fi

_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

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

echo "[lw][standard][noniid] alpha=${DIRICHLET_ALPHA:-0.5} benchmark=${BENCHMARK_DIR} model=${MODEL_PATH} methods=${#METHODS[@]}"

for agg_type in "${METHODS[@]}"; do
  echo "[lw][standard][noniid] run agg_type=${agg_type}"
  lwv4_train_standard_baseline "${agg_type}"
done

echo "[lw][standard][noniid] done. Metrics: ${METRICS_OUTPUT_DIR:-artifacts_LW_standard/sft_metrics}/"
