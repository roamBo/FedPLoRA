#!/usr/bin/env bash
# Legacy entry: runs group3 (normal only) then group4 (flora, ffa) — same three methods as before.
# Prefer calling run_domain_sft_batch_group3_normal.sh and run_domain_sft_batch_group4_flora_ffa.sh separately.
#
# Usage:
#   bash scripts/RunScripts/run_domain_sft_batch_group3_normal_flora_ffa.sh [7|14|21|35] [gpu]

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${_SCRIPT_DIR}/run_domain_sft_batch_group3_normal.sh" "$@"
bash "${_SCRIPT_DIR}/run_domain_sft_batch_group4_flora_ffa.sh" "$@"
