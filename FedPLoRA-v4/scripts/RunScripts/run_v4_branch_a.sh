#!/usr/bin/env bash
# Branch A — FedPLoRA-Hier++ runs
# A1: soft gate, prior cluster (K=3)
# A2: soft gate, spectral cluster (K=5)
# A3: soft gate, prior cluster (K=3) + personalized eval focus
#
# Usage (FedPLoRA repo root): bash FedPLoRA-v4/scripts/RunScripts/run_v4_branch_a.sh [gpu]
# Example: .../run_v4_branch_a.sh 0
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
V4_ROOT="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$V4_ROOT/configs/v4_baseline.env"
set -a
. "$ENV_FILE"
set +a
# shellcheck disable=SC1091
source "${HERE}/_v4_run_common.inc.sh"
v4_resolve_gpu "${1:-}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python "$V4_ROOT/tasks/fed_train_sft_v4.py" \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v4_hier_soft_prior \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir "${CLIENT_STATE_DIR}_a1" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --v4_gate_kappa 1.0 --v4_gate_power 1.0 \
  --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.3 --v4_lambda_max 0.9 \
  --v4_personalized_eval 1 --v4_default_uniform 1 \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python "$V4_ROOT/tasks/fed_train_sft_v4.py" \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v4_hier_soft_spectral \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir "${CLIENT_STATE_DIR}_a2" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --v4_gate_kappa 1.0 --v4_gate_power 1.0 \
  --v4_cluster_mode spectral --v4_cluster_k 5 \
  --v4_lambda_min 0.3 --v4_lambda_max 0.9 \
  --v4_personalized_eval 1 --v4_default_uniform 1 \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python "$V4_ROOT/tasks/fed_train_sft_v4.py" \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v4_hier_soft_pfl_eval \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir "${CLIENT_STATE_DIR}_a3" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --v4_gate_kappa 1.0 --v4_gate_power 2.0 \
  --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.2 --v4_lambda_max 1.0 \
  --v4_personalized_eval 1 --v4_default_uniform 1 \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"
