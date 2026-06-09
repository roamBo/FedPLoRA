#!/usr/bin/env bash
# LW Branch A — Hier++ (A1/A2/A3)
# Usage: bash scripts/RunScripts/LWv4/run_lwv4_branch_a.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
cd "${_REPO_ROOT}"
_lwv4_source_env "${_REPO_ROOT}"
_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

lwv4_train v4_hier_soft_prior LW_a1 \
  --v4_gate_kappa 1.0 --v4_gate_power 1.0 \
  --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.3 --v4_lambda_max 0.9 \
  --v4_personalized_eval 1 --v4_default_uniform 1

lwv4_train v4_hier_soft_spectral LW_a2 \
  --v4_gate_kappa 1.0 --v4_gate_power 1.0 \
  --v4_cluster_mode spectral --v4_cluster_k 5 \
  --v4_lambda_min 0.3 --v4_lambda_max 0.9 \
  --v4_personalized_eval 1 --v4_default_uniform 1

lwv4_train v4_hier_soft_pfl_eval LW_a3 \
  --v4_gate_kappa 1.0 --v4_gate_power 2.0 \
  --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.2 --v4_lambda_max 1.0 \
  --v4_personalized_eval 1 --v4_default_uniform 1
