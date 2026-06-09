#!/usr/bin/env bash
# LW Branch F — AdaRank (F1/F2, stub aggregator)
# Usage: bash scripts/RunScripts/LWv4/run_lwv4_branch_f.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
cd "${_REPO_ROOT}"
_lwv4_source_env "${_REPO_ROOT}"
_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

lwv4_train v4_adarank_risk16 LW_f1 --v4_adarank_mode risk16

lwv4_train v4_adarank_full LW_f2 --v4_adarank_mode full
