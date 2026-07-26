#!/usr/bin/env bash
# Re-evaluate finalized run bundles without training and report matched-domain
# macro/per-domain/Worst In-Domain metrics.
#
# Usage (GPU server, repository root):
#   CUDA_VISIBLE_DEVICES=0 \
#   MATCHED_DOMAIN_OUTPUT_ROOT=/path/to/result/matched_domain_eval \
#   bash scripts/RunScripts/run_eval_only_matched_domain.sh \
#     /path/to/formal/result_logs/method_a.json \
#     /path/to/formal/result_logs/method_b.json
#
# Each input JSON must be an existing formal metrics file containing the exact
# model, benchmark, agg_type, seed, and save_run_checkpoint_dir used by the run.
# The script never trains and never overwrites the source metrics JSON.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <formal-result.json> [more-result.json ...]" >&2
  exit 2
fi

MATCHED_DOMAIN_OUTPUT_ROOT="${MATCHED_DOMAIN_OUTPUT_ROOT:-${REPO_ROOT}/artifacts_matched_domain_eval}"
MATCHED_DOMAIN_SKIP_MISSING="${MATCHED_DOMAIN_SKIP_MISSING:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
EVAL_MAX_SEQ_LENGTH="${EVAL_MAX_SEQ_LENGTH:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_TORCH_DTYPE="${EVAL_TORCH_DTYPE:-bfloat16}"
mkdir -p "${MATCHED_DOMAIN_OUTPUT_ROOT}"

FAILED=0
DONE=0
SKIPPED=0

for result_json in "$@"; do
  if [[ ! -f "${result_json}" ]]; then
    echo "[error] missing formal result JSON: ${result_json}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi

  mapfile -t cfg < <(python - "${result_json}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
args = payload.get("args") or {}
effective = payload.get("effective_hparams") or {}

fields = [
    args.get("agg_type") or effective.get("agg_type") or "",
    args.get("model") or "",
    payload.get("benchmark_dir") or args.get("benchmark_dir") or "",
    args.get("save_run_checkpoint_dir") or "",
    str(args.get("seed", effective.get("seed", ""))),
    "1" if args.get("trust_remote_code", False) else "0",
    "1" if args.get("memory_agg_eval_use_local_clients", False) else "0",
    "1" if args.get("yoco_eval_use_local_clients", False) else "0",
]
for value in fields:
    print(str(value))
PY
  )

  if [[ "${#cfg[@]}" -ne 8 ]]; then
    echo "[error] incomplete run metadata: ${result_json}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi

  agg="${cfg[0]}"
  model="${cfg[1]}"
  benchmark="${cfg[2]}"
  checkpoint="${cfg[3]}"
  seed="${cfg[4]}"
  trust_remote_code="${cfg[5]}"
  use_local_clients="${cfg[6]}"
  yoco_use_local_clients="${cfg[7]}"

  missing=""
  [[ -n "${agg}" ]] || missing="${missing} agg_type"
  [[ -e "${model}" ]] || missing="${missing} model:${model}"
  [[ -d "${benchmark}" ]] || missing="${missing} benchmark:${benchmark}"
  [[ -f "${checkpoint}/run_checkpoint_meta.json" ]] || missing="${missing} checkpoint:${checkpoint}"
  if [[ -n "${missing}" ]]; then
    if [[ "${MATCHED_DOMAIN_SKIP_MISSING}" == "1" ]]; then
      echo "[skip]${missing} source=${result_json}" >&2
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    echo "[error]${missing} source=${result_json}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi

  dataset_tag="$(basename "$(dirname "${benchmark}")")_$(basename "${benchmark}")"
  run_tag="$(basename "${checkpoint}")"
  output_dir="${MATCHED_DOMAIN_OUTPUT_ROOT}/${dataset_tag}/${agg}_${seed}"
  mkdir -p "${output_dir}"

  cmd=(
    python tasks/fed_train_sft.py
    --model "${model}"
    --benchmark_dir "${benchmark}"
    --agg_type "${agg}"
    --seed "${seed}"
    --metrics_output_dir "${output_dir}"
    --eval_only_from_checkpoint "${checkpoint}"
    --eval_only_matched_domain
    --eval_max_batches "${EVAL_MAX_BATCHES}"
    --max_seq_length "${EVAL_MAX_SEQ_LENGTH}"
    --batch_size "${EVAL_BATCH_SIZE}"
    --torch_dtype "${EVAL_TORCH_DTYPE}"
  )
  [[ "${trust_remote_code}" == "1" ]] && cmd+=(--trust_remote_code)
  [[ "${use_local_clients}" == "1" ]] && cmd+=(--memory_agg_eval_use_local_clients)
  [[ "${yoco_use_local_clients}" == "1" ]] && cmd+=(--yoco_eval_use_local_clients)

  echo "[run] agg=${agg} seed=${seed} dataset=${dataset_tag} max_seq_length=${EVAL_MAX_SEQ_LENGTH}"
  echo "      checkpoint=${checkpoint}"
  echo "      output=${output_dir} run=${run_tag}"
  if "${cmd[@]}"; then
    DONE=$((DONE + 1))
  else
    echo "[error] eval-only failed: ${result_json}" >&2
    FAILED=$((FAILED + 1))
  fi
done

echo "[matched-domain-eval] done=${DONE} skipped=${SKIPPED} failed=${FAILED}"
[[ "${FAILED}" -eq 0 ]]
