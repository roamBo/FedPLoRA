#!/usr/bin/env bash
# Sequential diag 1→2→3 for Qwen2.5-1.5B (35c benchmark).
#
# Prereq: conda activate fedplora; cd repo root.
# You export shared vars (BENCHMARK_DIR, SEED, MAX_STEPS, ...); MODEL_PATH / TARGET_MODULES set below.
#
# Usage:
#   export MODEL_ROOT=/data/yaominghao/gb/models
#   export BENCHMARK_DIR=data/domain_benchmark_35c/seed_42
#   bash scripts/Analysis/run_diag_qwen15b.sh          # GPU 1
#   bash scripts/Analysis/run_diag_qwen15b.sh 2        # GPU 2
#   MAX_STEPS=60 bash scripts/Analysis/run_diag_qwen15b.sh   # smoke
#
# Background:
#   nohup bash scripts/Analysis/run_diag_qwen15b.sh 1 > log_diag/Qwen2.5-1.5B_run.log 2>&1 &
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_diag_run_common.inc.sh"

MODEL_ROOT="${MODEL_ROOT:-/data/yaominghao/gb/models}"
DIAG_MODEL_TAG="Qwen2.5-1.5B"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-1.5B}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

_diag_run_all "${1:-}"
