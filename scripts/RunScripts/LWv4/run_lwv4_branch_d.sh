#!/usr/bin/env bash
# LW Branch D — Mix (D1/D2/D3)
# Usage: bash scripts/RunScripts/LWv4/run_lwv4_branch_d.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
cd "${_REPO_ROOT}"
_lwv4_source_env "${_REPO_ROOT}"
_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

lwv4_train v4_mix_fixed05 LW_d1 \
  --v4_mix_mode fixed --v4_mix_eta 0.5 \
  --v4_mix_save_dir "${V4_MIX_SAVE_ROOT}_d1"

lwv4_train v4_mix_per_domain LW_d2 \
  --v4_mix_mode per_domain --v4_mix_eta 0.5 \
  --v4_mix_save_dir "${V4_MIX_SAVE_ROOT}_d2"

lwv4_train v4_mix_moe LW_d3 \
  --v4_mix_mode moe --v4_mix_eta 0.5 \
  --v4_mix_gate_hidden 64 --v4_mix_gate_epochs 3 \
  --v4_mix_save_dir "${V4_MIX_SAVE_ROOT}_d3"
