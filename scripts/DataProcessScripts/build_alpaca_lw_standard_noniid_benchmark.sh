#!/usr/bin/env bash
# LW pilot: subsampled Alpaca + Dirichlet non-IID (alpha=0.5, 7 clients).
#
# Usage (repo root):
#   bash scripts/DataProcessScripts/build_alpaca_lw_standard_noniid_benchmark.sh [max_samples] [alpha]
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

MAX_SAMPLES="${1:-5200}"
ALPHA="${2:-0.5}"
NUM_CLIENTS=7
RAW_JSONL="data/standard_sources/alpaca/alpaca_LW.jsonl"
BENCHMARK_ROOT="data/standard_benchmark_alpaca_LW_noniid_a${ALPHA}"

echo "[alpaca][LW][noniid] step 1/2: export HF dataset (cap=${MAX_SAMPLES}) -> ${RAW_JSONL}"
python scripts/DataProcessScripts/prepare_alpaca_standard_data.py \
  --dataset tatsu-lab/alpaca \
  --domain alpaca \
  --output "${RAW_JSONL}" \
  --prompt_template "{instruction}\n\n{input}" \
  --response_template "{output}" \
  --max_samples "${MAX_SAMPLES}" \
  --shuffle --seed 42

echo "[alpaca][LW][noniid] step 2/2: Dirichlet alpha=${ALPHA}, clients=${NUM_CLIENTS} -> ${BENCHMARK_ROOT}/seed_42"
python scripts/DataProcessScripts/build_standard_sft_benchmark.py \
  --input_jsonl "${RAW_JSONL}" \
  --output_dir "${BENCHMARK_ROOT}" \
  --num_clients "${NUM_CLIENTS}" \
  --domain_label alpaca \
  --partition dirichlet \
  --dirichlet_alpha "${ALPHA}" \
  --max_samples "${MAX_SAMPLES}" \
  --min_samples_per_client 30 \
  --seed 42

echo "[alpaca][LW][noniid] done. BENCHMARK_DIR=${BENCHMARK_ROOT}/seed_42"
