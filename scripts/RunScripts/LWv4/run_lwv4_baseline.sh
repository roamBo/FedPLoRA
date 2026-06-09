#!/usr/bin/env bash
# LW v4 baselines: v2 oneshot + fedalt
# Usage: bash scripts/RunScripts/LWv4/run_lwv4_baseline.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
_lwv4_source_env "${_REPO_ROOT}"
_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

lwv4_train fedplora_oneshot LW_v2base

lwv4_train fedalt LW_fedalt
