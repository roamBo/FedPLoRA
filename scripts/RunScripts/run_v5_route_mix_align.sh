#!/usr/bin/env bash
# FedPLoRA-Oneshot v5 — validation-routed A mixing + local B alignment.
#
# V5-L: deployable route. Each client uses its own private val rows to choose
#       one eta, then reuses that eta for all downstream domains.
# V5-LA: V5-L + B-only local alignment after routed A_eff is installed.
# V5-D: public-domain-anchor upper bound. It searches eta per evaluated domain.
# V5-RPCA: server common+sparse residual aggregation + V5-LA local safe route.
#
# Usage:
#   bash scripts/RunScripts/run_v5_route_mix_align.sh [gpu]
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

V5_METRICS_OUTPUT_DIR="${V5_METRICS_OUTPUT_DIR:-${METRICS_OUTPUT_DIR}}"
V5_ROUTE_GRID="${V5_ROUTE_GRID:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
V5_ROUTE_SEARCH_MAX_BATCHES="${V5_ROUTE_SEARCH_MAX_BATCHES:-4}"
V5_ROUTE_TIE_MARGIN="${V5_ROUTE_TIE_MARGIN:-0.0}"
V5_ROUTE_TIE_BREAKER="${V5_ROUTE_TIE_BREAKER:-best}"
V5_ROUTE_POST_ALIGN_STEPS="${V5_ROUTE_POST_ALIGN_STEPS:-5}"
V5_ROUTE_POST_ALIGN_LR="${V5_ROUTE_POST_ALIGN_LR:-0.0001}"
V5_ROUTE_POST_ALIGN_PROX="${V5_ROUTE_POST_ALIGN_PROX:-0.0}"
V5_RPCA_RANK="${V5_RPCA_RANK:-1}"
V5_RPCA_SPARSE_QUANTILE="${V5_RPCA_SPARSE_QUANTILE:-0.80}"

run_v5() {
  local tag="$1"
  local scope="$2"
  local align_steps="$3"
  local agg_type="${4:-v5_route_mix_align}"
  local save_dir="artifacts_35c/v5_route_a_local_${tag}"
  local state_dir="${CLIENT_STATE_DIR}_v5_${tag}"
  local metrics_dir="${V5_METRICS_OUTPUT_DIR}/${tag}"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  python tasks/fed_train_sft_v4.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --agg_type "$agg_type" \
    --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
    --gradient_checkpointing \
    --client_state_dir "$state_dir" --save_client_state_to_disk \
    --metrics_output_dir "$metrics_dir" \
    --eval_max_batches "$EVAL_MAX_BATCHES" --eval_seeds "$EVAL_SEEDS" \
    --v4_mix_save_dir "$save_dir" \
    --v5_route_val_scope "$scope" \
    --v5_route_search_grid "$V5_ROUTE_GRID" \
    --v5_route_search_max_batches "$V5_ROUTE_SEARCH_MAX_BATCHES" \
    --v5_route_tie_margin "$V5_ROUTE_TIE_MARGIN" \
    --v5_route_tie_breaker "$V5_ROUTE_TIE_BREAKER" \
    --v5_route_post_align_steps "$align_steps" \
    --v5_route_post_align_lr "$V5_ROUTE_POST_ALIGN_LR" \
    --v5_route_post_align_prox_lambda "$V5_ROUTE_POST_ALIGN_PROX" \
    --v3_rpca_rank "$V5_RPCA_RANK" \
    --v3_sparse_quantile "$V5_RPCA_SPARSE_QUANTILE" \
    --oneshot_anchor_lambda "$ONESHOT_ANCHOR_LAMBDA" \
    --oneshot_prox_lambda "$ONESHOT_PROX_LAMBDA" \
    --oneshot_consensus_power "$ONESHOT_CONSENSUS_POWER" \
    --oneshot_importance_power "$ONESHOT_IMPORTANCE_POWER" \
    --oneshot_importance_clip "$ONESHOT_IMPORTANCE_CLIP" \
    --oneshot_conflict_threshold "$ONESHOT_CONFLICT_THRESHOLD" \
    --oneshot_conflict_blend "$ONESHOT_CONFLICT_BLEND" \
    --yoco_sparse_lambda "$YOCO_SPARSE_LAMBDA"
}

run_v5 "local_route" "local" "0" "v5_route_mix_align"
run_v5 "local_route_align" "local" "$V5_ROUTE_POST_ALIGN_STEPS" "v5_route_mix_align"
run_v5 "domain_anchor_align" "domain" "$V5_ROUTE_POST_ALIGN_STEPS" "v5_route_mix_align"
run_v5 "rpca_local_route_align" "local" "$V5_ROUTE_POST_ALIGN_STEPS" "v5_rpca_route_mix_align"
