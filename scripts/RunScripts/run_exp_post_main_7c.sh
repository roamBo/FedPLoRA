#!/usr/bin/env bash
# 已改为 35 客户端默认；本脚本转发到 run_exp_post_main_35c.sh
echo "[warn] run_exp_post_main_7c.sh 已弃用，请使用 run_exp_post_main_35c.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_exp_post_main_35c.sh" "$@"
