#!/usr/bin/env bash
# LW Stanford Alpaca (Dirichlet non-IID α=0.5) — FedPLoRA v2 + v4 全矩阵
# Model: SmolLM2-135M | Benchmark: standard_benchmark_alpaca_LW_noniid_a0.5
#
# Runs:
#   v2:  fedplora_oneshot
#   v4:  branches A–F (16 configs, same as LW7c run_lwv4_branch_*)
#
# Prerequisite:
#   bash scripts/DataProcessScripts/build_alpaca_lw_standard_noniid_benchmark.sh
#   bash scripts/RunScripts/LWv4/download_lw_model_modelscope.sh
#
# Usage: bash scripts/RunScripts/LWv4/run_lw_standard_fedplora_all.sh [gpu]
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/_lwv4_run_common.inc.sh"
_REPO_ROOT="$(_lwv4_repo_root "${_SCRIPT_DIR}")"
cd "${_REPO_ROOT}"
_lw_standard_source_env "${_REPO_ROOT}"
_lwv4_resolve_gpu "${_REPO_ROOT}" "${1:-}"

echo "[lw][standard][fedplora] model=${MODEL_PATH} benchmark=${BENCHMARK_DIR}"
echo "[lw][standard][fedplora] metrics=${METRICS_OUTPUT_DIR} logs=${LOG_DIR}"

# --- v2 baseline ---
echo "[lw][standard][fedplora] v2: fedplora_oneshot"
lw_standard_train fedplora_oneshot std_v2

# --- Branch A: Hier++ ---
echo "[lw][standard][fedplora] Branch A"
lw_standard_train v4_hier_soft_prior std_a1 \
  --v4_gate_kappa 1.0 --v4_gate_power 1.0 \
  --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.3 --v4_lambda_max 0.9 \
  --v4_personalized_eval 1 --v4_default_uniform 1

lw_standard_train v4_hier_soft_spectral std_a2 \
  --v4_gate_kappa 1.0 --v4_gate_power 1.0 \
  --v4_cluster_mode spectral --v4_cluster_k 5 \
  --v4_lambda_min 0.3 --v4_lambda_max 0.9 \
  --v4_personalized_eval 1 --v4_default_uniform 1

lw_standard_train v4_hier_soft_pfl_eval std_a3 \
  --v4_gate_kappa 1.0 --v4_gate_power 2.0 \
  --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.2 --v4_lambda_max 1.0 \
  --v4_personalized_eval 1 --v4_default_uniform 1

# --- Branch B: SVD ---
echo "[lw][standard][fedplora] Branch B"
lw_standard_train v4_svd_orth_only std_b1 \
  --v4_svd_orth_init 1 --v4_svd_refactor 0 --v4_svd_procrustes 0

lw_standard_train v4_svd_full std_b2 \
  --v4_svd_orth_init 1 --v4_svd_refactor 1 --v4_svd_procrustes 1

# --- Branch C: Sign ---
echo "[lw][standard][fedplora] Branch C"
lw_standard_train v4_sign_v2agg std_c1 \
  --v4_bsign_lambda 1e-3 --v4_bsign_gamma 5.0 --v4_bsign_anchor_steps 1 \
  --v4_asparse_lambda 0

lw_standard_train v4_sign_full std_c2 \
  --v4_bsign_lambda 1e-3 --v4_bsign_gamma 5.0 --v4_bsign_anchor_steps 1 \
  --v4_asparse_lambda 1e-4

# --- Branch D: Mix ---
echo "[lw][standard][fedplora] Branch D"
lw_standard_train v4_mix_fixed05 std_d1 \
  --v4_mix_mode fixed --v4_mix_eta 0.5 \
  --v4_mix_save_dir "${V4_MIX_SAVE_ROOT}_d1"

lw_standard_train v4_mix_per_domain std_d2 \
  --v4_mix_mode per_domain --v4_mix_eta 0.5 \
  --v4_mix_save_dir "${V4_MIX_SAVE_ROOT}_d2"

lw_standard_train v4_mix_moe std_d3 \
  --v4_mix_mode moe --v4_mix_eta 0.5 \
  --v4_mix_gate_hidden 64 --v4_mix_gate_epochs 3 \
  --v4_mix_save_dir "${V4_MIX_SAVE_ROOT}_d3"

# --- Branch E: Anchor ---
echo "[lw][standard][fedplora] Branch E"
lw_standard_train v4_anchor_gate std_e1 \
  --v4_use_anchor 1 --v4_anchor_gate_threshold 0.30 \
  --v4_gate_kappa 1.0 --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.3 --v4_lambda_max 0.9

lw_standard_train v4_anchor_lambda std_e2 \
  --v4_use_anchor 1 --v4_anchor_cluster_lambda 0.6 \
  --v4_gate_kappa 1.0 --v4_cluster_mode prior --v4_cluster_k 3 \
  --v4_lambda_min 0.2 --v4_lambda_max 1.0

# --- Branch F: AdaRank ---
echo "[lw][standard][fedplora] Branch F"
lw_standard_train v4_adarank_risk16 std_f1 --v4_adarank_mode risk16

lw_standard_train v4_adarank_full std_f2 --v4_adarank_mode full

echo "[lw][standard][fedplora] all done (1 v2 + 16 v4 = 17 runs)."
echo "[lw][standard][fedplora] metrics: ${METRICS_OUTPUT_DIR}/"
echo "[lw][standard][fedplora] checkpoints: ${TRAINED_MODELS_ROOT}/"
