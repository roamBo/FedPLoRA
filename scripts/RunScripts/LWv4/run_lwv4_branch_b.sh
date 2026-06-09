#!/usr/bin/env bash
# LW Branch B — SVD (B1/B2)
# Usage: bash scripts/RunScripts/LWv4/run_lwv4_branch_b.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
cd "${_REPO_ROOT}"
_lwv4_source_env "${_REPO_ROOT}"
_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

lwv4_train v4_svd_orth_only LW_b1 \
  --v4_svd_orth_init 1 --v4_svd_refactor 0 --v4_svd_procrustes 0

lwv4_train v4_svd_full LW_b2 \
  --v4_svd_orth_init 1 --v4_svd_refactor 1 --v4_svd_procrustes 1
