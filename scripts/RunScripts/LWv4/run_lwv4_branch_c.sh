#!/usr/bin/env bash
# LW Branch C — Sign (C1/C2)
# Usage: bash scripts/RunScripts/LWv4/run_lwv4_branch_c.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
cd "${_REPO_ROOT}"
_lwv4_source_env "${_REPO_ROOT}"
_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

lwv4_train v4_sign_v2agg LW_c1 \
  --v4_bsign_lambda 1e-3 --v4_bsign_gamma 5.0 --v4_bsign_anchor_steps 1 \
  --v4_asparse_lambda 0

lwv4_train v4_sign_full LW_c2 \
  --v4_bsign_lambda 1e-3 --v4_bsign_gamma 5.0 --v4_bsign_anchor_steps 1 \
  --v4_asparse_lambda 1e-4
