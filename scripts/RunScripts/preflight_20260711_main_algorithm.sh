#!/usr/bin/env bash
# Usage:
#   cd /data/yaominghao/gb/FedPLoRA
#   source scripts/RunScripts/preflight_20260711_main_algorithm.sh
#
# 20260711 v13 one-shot order 专用 preflight：
# - 使用 v13 角色（非 v12 RUN_ID_PREFIX）
# - 默认不调用 set_run_paths 42，避免子进程误写入 v12 目录

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[usage][error] 请先执行 exec bash，然后 source 本脚本。" >&2
  return 2 2>/dev/null || exit 2
fi

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "[usage][error] 请用 source 加载本脚本，否则函数和环境变量不会保留在当前 shell。" >&2
  echo "正确示例：source scripts/RunScripts/preflight_20260711_main_algorithm.sh" >&2
  exit 2
fi

export FEDPLORA_PREFLIGHT_ROLE=v13
export FEDPLORA_SKIP_DEFAULT_SET_RUN_PATHS=${FEDPLORA_SKIP_DEFAULT_SET_RUN_PATHS:-1}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/preflight_20260709_common.sh"
