#!/usr/bin/env bash
# Usage:
#   cd /data2/minghao/code/FedPLoRA-main
#   source scripts/RunScripts/preflight_20260709_baseline.sh

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[usage][error] 请先执行 exec bash，然后 source 本脚本。" >&2
  return 2 2>/dev/null || exit 2
fi

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "[usage][error] 请用 source 加载本脚本，否则函数和环境变量不会保留在当前 shell。" >&2
  echo "正确示例：source scripts/RunScripts/preflight_20260709_baseline.sh" >&2
  exit 2
fi

export FEDPLORA_PREFLIGHT_ROLE=baseline
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/preflight_20260709_common.sh"
