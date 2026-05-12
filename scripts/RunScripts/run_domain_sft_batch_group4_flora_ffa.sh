#!/usr/bin/env bash
# Group 4 — Flora, FFA (full training; checkpoints saved).
# Split from the former group3 bundle; FedAvg-Normal is in run_domain_sft_batch_group3_normal.sh.
#
# Usage:
#   bash scripts/RunScripts/run_domain_sft_batch_group4_flora_ffa.sh [7|14|21|35] [gpu]

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

if [[ -f "${_REPO_ROOT}/configs/domain_sft.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/domain_sft.env"
  set +a
fi

usage() {
  echo "Usage: $0 [7|14|21|35] [gpu]" >&2
  exit 1
}

NC="${1:-35}"
GPU_CLI="${2:-}"
case "${NC}" in
  7|14|21|35) BENCHMARK_DIR="data/domain_benchmark_${NC}c/seed_42" ;;
  *) usage ;;
esac

# shellcheck disable=SC1091
source "${_REPO_ROOT}/configs/cuda_resolve.inc.sh"
cuda_resolve_devices "${GPU_CLI}"

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_fed_train_speed.inc.sh"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_run_domain_sft_batch.inc.sh"

domain_sft_run_batch "batch-group4-flora-ffa" flora ffa
