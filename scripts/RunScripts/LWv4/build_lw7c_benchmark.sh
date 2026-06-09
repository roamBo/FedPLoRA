#!/usr/bin/env bash
# Build LW7c: 7 clients (1/domain), ~1/5 data per client (one 35c shard, not merged).
#
# Usage: bash scripts/RunScripts/LWv4/build_lw7c_benchmark.sh
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
cd "${_REPO_ROOT}"
python scripts/DataProcessScripts/build_lw7c_benchmark.py \
  --src_35c data/domain_benchmark_35c \
  --output_dir data/domain_benchmark_LW7c \
  --seed 42
