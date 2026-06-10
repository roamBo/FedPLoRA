#!/usr/bin/env bash
# LW classic SFT baselines — same comparison set as run_domain_sft_baselines1/2.
# Uses tasks/fed_train_sft.py (not fed_train_sft_v4.py).
#
# Methods: normal, flora, flexlora, feddat, fedplora-oneshot, fedalt, yoco, fedsa_lora, ffa
#
# Usage: bash scripts/RunScripts/LWv4/run_lwv4_baseline.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
_LWv4_RUNSCRIPTS_DIR="$(cd "${_SCRIPT_DIR}/.." && pwd)"
export _LWv4_RUNSCRIPTS_DIR
cd "${_REPO_ROOT}"
_lwv4_source_env "${_REPO_ROOT}"
_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

METHODS=(
  normal
  flora
  flexlora
  feddat
  fedplora-oneshot
  fedalt
  yoco
  fedsa_lora
  ffa
)

echo "[lwv4][baseline] benchmark=${BENCHMARK_DIR} model=${MODEL_PATH} methods=${#METHODS[@]}"

for agg_type in "${METHODS[@]}"; do
  echo "[lwv4][baseline] run agg_type=${agg_type}"
  lwv4_train_baseline "${agg_type}"
done

echo "[lwv4][baseline] done. Metrics: ${METRICS_BASELINE_OUTPUT_DIR:-artifacts_LW7c/sft_metrics}/"
