#!/usr/bin/env bash
# Sequential diag 1→2→3 for OPT-1.3B (35c benchmark).
# Requires torch>=2.6 if weights are pytorch_model.bin (see 诊断命令.md).
#
# Usage:
#   export MODEL_ROOT=/data/yaominghao/gb/models
#   export BENCHMARK_DIR=data/domain_benchmark_35c/seed_42
#   bash scripts/Analysis/run_diag_opt13b.sh 1
#   nohup bash scripts/Analysis/run_diag_opt13b.sh 1 > log_diag/OPT-1.3B_run.log 2>&1 &
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_diag_run_common.inc.sh"

MODEL_ROOT="${MODEL_ROOT:-/data/yaominghao/gb/models}"
DIAG_MODEL_TAG="OPT-1.3B"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/OPT-1.3B}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,out_proj,fc1,fc2}"

_diag_run_all "${1:-}"
