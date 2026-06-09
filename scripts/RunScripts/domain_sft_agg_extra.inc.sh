# shellcheck shell=bash
# Per-agg_type CLI extras for eval-only / post-hoc eval (must match §11.1 training flags).

domain_sft_append_agg_cli_extras() {
  local -n _cmd=$1
  local _agg="$2"
  case "${_agg}" in
    yoco)
      _cmd+=(
        --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA:-1e-4}"
        --yoco_aggregate_mode "${YOCO_AGGREGATE_MODE:-conflict}"
        --yoco_conflict_method "${YOCO_CONFLICT_METHOD:-avgm}"
        --yoco_sign_lambda "${YOCO_SIGN_LAMBDA:-0.01}"
      )
      ;;
    fedplora-oneshot)
      _cmd+=(
        --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA:-1e-4}"
        --oneshot_anchor_lambda "${ONESHOT_ANCHOR_LAMBDA:-1e-4}"
        --oneshot_prox_lambda "${ONESHOT_PROX_LAMBDA:-0.0}"
        --oneshot_consensus_power "${ONESHOT_CONSENSUS_POWER:-2.0}"
        --oneshot_importance_power "${ONESHOT_IMPORTANCE_POWER:-1.0}"
        --oneshot_importance_clip "${ONESHOT_IMPORTANCE_CLIP:-5.0}"
        --oneshot_conflict_threshold "${ONESHOT_CONFLICT_THRESHOLD:-0.35}"
        --oneshot_conflict_blend "${ONESHOT_CONFLICT_BLEND:-1.0}"
        --oneshot_scale_clip_ratio "${ONESHOT_SCALE_CLIP_RATIO:-0.0}"
      )
      if [[ "${ONESHOT_NO_KEEP_INIT_ON_CONFLICT:-0}" == "1" ]]; then
        _cmd+=(--oneshot_no_keep_init_on_conflict)
      fi
      if [[ "${ONESHOT_ORTHOGONALIZE:-0}" == "1" ]]; then
        _cmd+=(--oneshot_orthogonalize)
      fi
      ;;
    fedplora_v3_lite|fedplora_v3_cluster|fedplora_v3_rpca|v3_lite|v3_cluster|v3_rpca)
      _cmd+=(
        --yoco_sparse_lambda "${YOCO_SPARSE_LAMBDA:-1e-4}"
        --oneshot_anchor_lambda "${ONESHOT_ANCHOR_LAMBDA:-1e-4}"
        --oneshot_prox_lambda "${ONESHOT_PROX_LAMBDA:-0.0}"
        --oneshot_consensus_power "${ONESHOT_CONSENSUS_POWER:-2.0}"
        --oneshot_importance_power "${ONESHOT_IMPORTANCE_POWER:-1.0}"
        --oneshot_importance_clip "${ONESHOT_IMPORTANCE_CLIP:-5.0}"
        --oneshot_conflict_threshold "${ONESHOT_CONFLICT_THRESHOLD:-0.35}"
        --oneshot_conflict_blend "${ONESHOT_CONFLICT_BLEND:-1.0}"
        --oneshot_scale_clip_ratio "${ONESHOT_SCALE_CLIP_RATIO:-0.0}"
        --v3_conflict_quantile "${V3_CONFLICT_QUANTILE:-0.8}"
        --v3_gate_temperature "${V3_GATE_TEMPERATURE:-0.05}"
        --v3_conflict_blend "${V3_CONFLICT_BLEND:-1.0}"
        --v3_cluster_mode "${V3_CLUSTER_MODE:-domain_prior}"
        --v3_cluster_lambda_min "${V3_CLUSTER_LAMBDA_MIN:-0.2}"
        --v3_cluster_lambda_max "${V3_CLUSTER_LAMBDA_MAX:-1.0}"
        --v3_rpca_rank "${V3_RPCA_RANK:-1}"
        --v3_sparse_quantile "${V3_SPARSE_QUANTILE:-0.8}"
      )
      ;;
    feddat)
      _cmd+=(--feddat_teacher_lambda "${FEDDAT_TEACHER_LAMBDA:-0.01}")
      ;;
    flora|flexlora)
      _cmd+=(--flora_svd_device "${FLORA_SVD_DEVICE:-auto}")
      ;;
  esac
}
