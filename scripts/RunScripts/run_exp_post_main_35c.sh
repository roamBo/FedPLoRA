#!/usr/bin/env bash
# 35 客户端主实验完成后：一次跑完【通信-性能】+【个性化收益分析】（均不训练）。
#
# Usage (repo root):
#   bash scripts/RunScripts/run_exp_post_main_35c.sh [gpu]

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

GPU_CLI="${1:-}"

echo "========== [post-main-35c] 1/2 communication profile (no training) =========="
bash "${_SCRIPT_DIR}/run_exp_comm_profile.sh" "${GPU_CLI}"

echo "========== [post-main-35c] 2/2 personalization eval-only (12 methods, 35 clients) =========="
bash "${_SCRIPT_DIR}/run_exp_personalization.sh" 35 "${GPU_CLI}"

echo "[post-main-35c] all done."
