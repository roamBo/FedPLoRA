#!/usr/bin/env bash
# LW FedPLoRA-Oneshot v5 screening: route / route+align / domain-anchor / rpca-route.
#
# Usage:
#   bash scripts/RunScripts/LWv4/run_lwv5_route_mix_align.sh [gpu]
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
cd "${_REPO_ROOT}"
_lwv4_source_env "${_REPO_ROOT}"
_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

V5_ROUTE_GRID="${V5_ROUTE_GRID:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
V5_ROUTE_SEARCH_MAX_BATCHES="${V5_ROUTE_SEARCH_MAX_BATCHES:-2}"
V5_ROUTE_TIE_MARGIN="${V5_ROUTE_TIE_MARGIN:-0.0}"
V5_ROUTE_TIE_BREAKER="${V5_ROUTE_TIE_BREAKER:-best}"
V5_ROUTE_POST_ALIGN_STEPS="${V5_ROUTE_POST_ALIGN_STEPS:-3}"
V5_ROUTE_POST_ALIGN_LR="${V5_ROUTE_POST_ALIGN_LR:-0.0001}"
V5_ROUTE_POST_ALIGN_PROX="${V5_ROUTE_POST_ALIGN_PROX:-0.0}"
V5_RPCA_RANK="${V5_RPCA_RANK:-1}"
V5_RPCA_SPARSE_QUANTILE="${V5_RPCA_SPARSE_QUANTILE:-0.80}"

lwv5_train() {
  local tag="$1"
  local scope="$2"
  local align_steps="$3"
  local agg_type="${4:-v5_route_mix_align}"
  local metrics_root="${METRICS_OUTPUT_DIR}_v5"

  LWV4_METRICS_OUTPUT_DIR_OVERRIDE="${metrics_root}/${tag}" \
  lwv4_train "$agg_type" "LWv5_${tag}" \
    --v4_mix_save_dir "${V4_MIX_SAVE_ROOT}_v5_${tag}" \
    --v5_route_val_scope "$scope" \
    --v5_route_search_grid "$V5_ROUTE_GRID" \
    --v5_route_search_max_batches "$V5_ROUTE_SEARCH_MAX_BATCHES" \
    --v5_route_tie_margin "$V5_ROUTE_TIE_MARGIN" \
    --v5_route_tie_breaker "$V5_ROUTE_TIE_BREAKER" \
    --v5_route_post_align_steps "$align_steps" \
    --v5_route_post_align_lr "$V5_ROUTE_POST_ALIGN_LR" \
    --v5_route_post_align_prox_lambda "$V5_ROUTE_POST_ALIGN_PROX" \
    --v3_rpca_rank "$V5_RPCA_RANK" \
    --v3_sparse_quantile "$V5_RPCA_SPARSE_QUANTILE"
}

lwv5_train "local_route" "local" "0" "v5_route_mix_align"
lwv5_train "local_route_align" "local" "$V5_ROUTE_POST_ALIGN_STEPS" "v5_route_mix_align"
lwv5_train "domain_anchor_align" "domain" "$V5_ROUTE_POST_ALIGN_STEPS" "v5_route_mix_align"
lwv5_train "rpca_local_route_align" "local" "$V5_ROUTE_POST_ALIGN_STEPS" "v5_rpca_route_mix_align"
