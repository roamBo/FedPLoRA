#!/usr/bin/env bash
# Run full LW v4 screening matrix: baseline + branches A–F (18 configs).
# Goal: pick best v4 direction at minimal cost before 35c × Llama-8B.
#
# Usage: bash scripts/RunScripts/LWv4/run_lwv4_all.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-}"

bash "${_SCRIPT_DIR}/run_lwv4_baseline.sh" "${GPU}"
bash "${_SCRIPT_DIR}/run_lwv4_branch_a.sh" "${GPU}"
bash "${_SCRIPT_DIR}/run_lwv4_branch_b.sh" "${GPU}"
bash "${_SCRIPT_DIR}/run_lwv4_branch_c.sh" "${GPU}"
bash "${_SCRIPT_DIR}/run_lwv4_branch_d.sh" "${GPU}"
bash "${_SCRIPT_DIR}/run_lwv4_branch_e.sh" "${GPU}"
bash "${_SCRIPT_DIR}/run_lwv4_branch_f.sh" "${GPU}"

echo "[lwv4] all branches done. Metrics: artifacts_LW7c/v4_sft_metrics/  Logs: log_LWv4/"
