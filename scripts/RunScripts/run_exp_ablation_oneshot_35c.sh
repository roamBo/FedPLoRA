#!/usr/bin/env bash
# 35c FedPLoRA-Oneshot v2 三模块消融一键入口（可指定 GPU）。
# 等价于：bash scripts/RunScripts/run_exp_ablation_fedplora.sh 35 [gpu]

set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${_SCRIPT_DIR}/run_exp_ablation_fedplora.sh" 35 "${1:-}"
