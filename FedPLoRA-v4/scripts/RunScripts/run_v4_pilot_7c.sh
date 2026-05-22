#!/usr/bin/env bash
# 7c pilot: tiny benchmark with 1 client/domain, used for fast mechanism verification.
# Goal: each run < 30 minutes so we can iterate on branch designs before 35c main.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
V4_ROOT="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$V4_ROOT/configs/v4_baseline.env"
set -a
. "$ENV_FILE"
set +a

BENCHMARK_DIR=data/domain_benchmark_7c/seed_42
EVAL_MAX_BATCHES=100
EVAL_SEEDS=42

for AGG in fedplora_oneshot fedalt \
           v4_hier_soft_prior v4_hier_soft_spectral v4_hier_soft_pfl_eval \
           v4_sign_v2agg v4_sign_full \
           v4_mix_fixed05; do
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  python "$V4_ROOT/tasks/fed_train_sft_v4.py" \
    --model "$MODEL_PATH" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --agg_type "$AGG" \
    --rounds 1 --local_epochs 1 --lr "$LR" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
    --client_state_dir "artifacts_7c/v4_client_states_${AGG}" --save_client_state_to_disk \
    --metrics_output_dir artifacts_7c/v4_sft_metrics \
    --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
    --v4_bsign_lambda 1e-3 --v4_bsign_gamma 5.0 \
    --v4_asparse_lambda $( [[ "$AGG" == "v4_sign_full" ]] && echo 1e-4 || echo 0 ) \
    --v4_mix_mode fixed --v4_mix_eta 0.5 \
    --v4_mix_save_dir "artifacts_7c/v4_mix_a_local_${AGG}" \
    --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA"
done
