#!/usr/bin/env bash
# Branch D — FedPLoRA-Mix runs
# D1: fixed eta=0.5
# D2: per-domain eta grid search on validation rows
# D3: per-input MoE placeholder (currently falls back to fixed/static mixer)
#
# Usage: bash scripts/RunScripts/run_v4_branch_d.sh [gpu]
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

# D1 — fixed eta = 0.5
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v4_mix_fixed05 \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir "${CLIENT_STATE_DIR}_d1" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --v4_mix_mode fixed --v4_mix_eta 0.5 \
  --v4_mix_save_dir "artifacts/v4_mix_a_local_d1" \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"

# D2 — per-domain eta grid search on val
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v4_mix_per_domain \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir "${CLIENT_STATE_DIR}_d2" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --v4_mix_mode per_domain --v4_mix_eta 0.5 \
  --v4_mix_val_scope local \
  --v4_mix_search_grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
  --v4_mix_search_max_batches 4 \
  --v4_mix_save_dir "artifacts/v4_mix_a_local_d2" \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"

# D3 — per-input MoE gate (stub)
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v4_mix_moe \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --gradient_checkpointing \
  --client_state_dir "${CLIENT_STATE_DIR}_d3" --save_client_state_to_disk \
  --metrics_output_dir "$METRICS_OUTPUT_DIR" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
  --v4_mix_mode moe --v4_mix_eta 0.5 \
  --v4_mix_gate_hidden 64 --v4_mix_gate_epochs 3 \
  --v4_mix_save_dir "artifacts/v4_mix_a_local_d3" \
  --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"
