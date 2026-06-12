#!/usr/bin/env bash
# One-shot: download Alpaca -> IID federated benchmark (default 10 clients).
#
# Usage (repo root):
#   bash scripts/DataProcessScripts/build_alpaca_standard_benchmark.sh [num_clients]
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

NUM_CLIENTS="${1:-10}"
RAW_JSONL="data/standard_sources/alpaca/alpaca.jsonl"
BENCHMARK_ROOT="data/standard_benchmark_alpaca"

echo "[alpaca] step 1/2: export HF dataset -> ${RAW_JSONL}"
python scripts/DataProcessScripts/prepare_alpaca_standard_data.py \
  --dataset tatsu-lab/alpaca \
  --domain alpaca \
  --output "${RAW_JSONL}" \
  --prompt_template "{instruction}\n\n{input}" \
  --response_template "{output}" \
  --shuffle --seed 42

echo "[alpaca] step 2/2: IID benchmark (${NUM_CLIENTS} clients) -> ${BENCHMARK_ROOT}/seed_42"
python scripts/DataProcessScripts/build_standard_sft_benchmark.py \
  --input_jsonl "${RAW_JSONL}" \
  --output_dir "${BENCHMARK_ROOT}" \
  --num_clients "${NUM_CLIENTS}" \
  --domain_label alpaca \
  --seed 42

echo "[alpaca] done. BENCHMARK_DIR=${BENCHMARK_ROOT}/seed_42"
