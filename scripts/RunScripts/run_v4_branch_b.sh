#!/usr/bin/env bash
# Branch B — FedPLoRA-SVD runs
# B1: A_0 orthogonal init only (no SVD refactor at aggregation)
# B2: A_0 orthogonal + stacked-A SVD refactor
#
# Usage: bash scripts/RunScripts/run_v4_branch_b.sh [gpu]
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

# B1 — orth init only
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v4_svd_orth_only \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir "${CLIENT_STATE_DIR}_b1" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --v4_svd_orth_init 1 --v4_svd_refactor 0 --v4_svd_procrustes 0 \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"

# B2 — orth init + SVD refactor
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v4_svd_full \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir "${CLIENT_STATE_DIR}_b2" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --v4_svd_orth_init 1 --v4_svd_refactor 1 --v4_svd_procrustes 1 \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"
