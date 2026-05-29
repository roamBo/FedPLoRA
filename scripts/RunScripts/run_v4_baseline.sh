#!/usr/bin/env bash
# Replicate v2 baselines with v4 evaluation settings (eval_max_batches=200, 3 seeds).
# Used to give Branch A/C/D a stable reference point.
#
# Usage: bash scripts/RunScripts/run_v4_baseline.sh [gpu]
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"
if [[ -f "${_REPO_ROOT}/configs/v4_baseline.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/configs/v4_baseline.env"
  set +a
fi
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_v4_run_common.inc.sh"
v4_resolve_gpu "${1:-}"

# Baseline 1 — v2 fedplora-oneshot
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type fedplora_oneshot \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir "${CLIENT_STATE_DIR}_v2base" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"

# Baseline 2 — FedSA-LoRA / FedALT
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type fedalt \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir "${CLIENT_STATE_DIR}_fedalt" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS"
