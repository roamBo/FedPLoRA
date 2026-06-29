#!/usr/bin/env bash
# Sequential diag 1→2→3 for Pythia-1.4B (35c benchmark).
# Requires torch>=2.6 if weights are pytorch_model.bin (see 诊断命令.md).
#
# Usage:
#   export MODEL_ROOT=/data/yaominghao/gb/models
#   export BENCHMARK_DIR=data/domain_benchmark_35c/seed_42
#   bash scripts/Analysis/run_diag_pythia14b.sh 1
#   nohup bash scripts/Analysis/run_diag_pythia14b.sh 1 > log_diag/Pythia-1.4B_run.log 2>&1 &
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_diag_run_common.inc.sh"

MODEL_ROOT="${MODEL_ROOT:-/data/yaominghao/gb/models}"
DIAG_MODEL_TAG="Pythia-1.4B"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Pythia-1.4B}"
TARGET_MODULES="${TARGET_MODULES:-query_key_value,dense,dense_h_to_4h,dense_4h_to_h}"

_diag_run_all "${1:-}"
