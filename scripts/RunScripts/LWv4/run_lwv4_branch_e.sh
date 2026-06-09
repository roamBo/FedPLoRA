#!/usr/bin/env bash
# LW Branch E — Anchor (E1/E2, stub aggregator)
# Usage: bash scripts/RunScripts/LWv4/run_lwv4_branch_e.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
cd "${_REPO_ROOT}"
_lwv4_source_env "${_REPO_ROOT}"
_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

lwv4_train v4_anchor_gate LW_e1 \
  --v4_use_anchor 1 --v4_anchor_gate_threshold 0.30 \
  --v4_gate_kappa 1.0 --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.3 --v4_lambda_max 0.9

lwv4_train v4_anchor_lambda LW_e2 \
  --v4_use_anchor 1 --v4_anchor_cluster_lambda 0.6 \
  --v4_gate_kappa 1.0 --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.2 --v4_lambda_max 1.0
