#!/usr/bin/env bash
# Alpaca -> Dirichlet non-IID federated benchmark (default alpha=0.5, 10 clients).
#
# Usage (repo root):
#   bash scripts/DataProcessScripts/build_alpaca_standard_benchmark_noniid.sh [num_clients] [alpha]
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

NUM_CLIENTS="${1:-10}"
ALPHA="${2:-0.5}"
RAW_JSONL="data/standard_sources/alpaca/alpaca.jsonl"
BENCHMARK_ROOT="data/standard_benchmark_alpaca_noniid_a${ALPHA}"

echo "[alpaca][noniid] step 1/2: export HF dataset -> ${RAW_JSONL}"
python scripts/DataProcessScripts/prepare_alpaca_standard_data.py \
  --dataset tatsu-lab/alpaca \
  --domain alpaca \
  --output "${RAW_JSONL}" \
  --prompt_template "{instruction}\n\n{input}" \
  --response_template "{output}" \
  --shuffle --seed 42

echo "[alpaca][noniid] step 2/2: Dirichlet alpha=${ALPHA}, clients=${NUM_CLIENTS} -> ${BENCHMARK_ROOT}/seed_42"
python scripts/DataProcessScripts/build_standard_sft_benchmark.py \
  --input_jsonl "${RAW_JSONL}" \
  --output_dir "${BENCHMARK_ROOT}" \
  --num_clients "${NUM_CLIENTS}" \
  --domain_label alpaca \
  --partition dirichlet \
  --dirichlet_alpha "${ALPHA}" \
  --seed 42

echo "[alpaca][noniid] done. BENCHMARK_DIR=${BENCHMARK_ROOT}/seed_42"
