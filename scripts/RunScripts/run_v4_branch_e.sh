#!/usr/bin/env bash
# Branch E — FedPLoRA-Anchor runs (Stage 5; aggregator stub → Hier++ prior until anchor wired)
# E1: tune gate_threshold via anchor (placeholder: v4_anchor_gate)
# E2: tune cluster_lambda via anchor (placeholder: v4_anchor_lambda)
#
# Usage: bash scripts/RunScripts/run_v4_branch_e.sh [gpu]
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

# E1 — anchor gate threshold sweep placeholder
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v4_anchor_gate \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir "${CLIENT_STATE_DIR}_e1" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --v4_use_anchor 1 --v4_anchor_gate_threshold 0.30 \
  --v4_gate_kappa 1.0 --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.3 --v4_lambda_max 0.9 \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"

# E2 — anchor cluster lambda placeholder
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v4_anchor_lambda \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir "${CLIENT_STATE_DIR}_e2" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --v4_use_anchor 1 --v4_anchor_cluster_lambda 0.6 \
  --v4_gate_kappa 1.0 --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.2 --v4_lambda_max 1.0 \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"
