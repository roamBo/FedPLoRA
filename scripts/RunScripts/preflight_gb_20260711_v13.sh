#!/usr/bin/env bash
# gb 服务器 v13 one-shot order 专用 preflight + 后台运行函数。
# Usage:
#   cd /data/yaominghao/gb/FedPLoRA
#   source scripts/RunScripts/preflight_gb_20260711_v13.sh
#   GPU=0 run_v13_sft NX1_v13a_os_split43_train43 fedplora_v13a_os 43 v13_20260711_nx1_35c_dir05_r1_finaleval --force_retrain

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[usage][error] 请先 exec bash，然后 source 本脚本。" >&2
  return 2 2>/dev/null || exit 2
fi

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "[usage][error] 请用 source 加载，不要 bash 直接执行。" >&2
  exit 2
fi

set -o pipefail

export CODE_DIR=${CODE_DIR:-/data/yaominghao/gb/FedPLoRA}
export CODE_ROOT="$CODE_DIR"
export CONDA_ENV_NAME=${CONDA_ENV_NAME:-fedplora}
export MODEL_ROOT=${MODEL_ROOT:-/data/yaominghao/gb/models/trained_models_LW}
export MODEL_PATH=${MODEL_PATH:-/data/yaominghao/gb/models/SmolLM2-135M}
export DATA_ROOT=${DATA_ROOT:-$CODE_DIR/data}
export RESULT_ROOT=${RESULT_ROOT:-/data/yaominghao/gb/result/FedPLoRA}
export BENCHMARK_DIR_MAIN=${BENCHMARK_DIR_MAIN:-$DATA_ROOT/domain_benchmark_35c_dir05/seed_42}
export TARGET_MODULES=${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
export GPU=${GPU:-1}

export NX1_RUN_ID_PREFIX=${NX1_RUN_ID_PREFIX:-v13_20260711_nx1_35c_dir05_r1_finaleval}
export NX4_RUN_ID_PREFIX=${NX4_RUN_ID_PREFIX:-v13_20260711_nx4_personalized_eval}
export NX3_RUN_ID_PREFIX=${NX3_RUN_ID_PREFIX:-v13_20260711_nx3_ablation_split42_r1_finaleval}
export SMOKE_RUN_ID_20260711=${SMOKE_RUN_ID_20260711:-v13_20260711_smoke_seed42}

export LAUNCH_DIR=${LAUNCH_DIR:-$RESULT_ROOT/manual_launch_logs_20260711}
export V13_ONE_EXP="$CODE_DIR/scripts/RunScripts/run_20260711_one_experiment.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "[preflight_gb_v13][error] conda 不可用。" >&2
  return 2
fi
CONDA_BASE="$(conda info --base)"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]; then
  conda activate "$CONDA_ENV_NAME"
fi

cd "$CODE_DIR" || return 2

export FEDPLORA_SKIP_DEFAULT_SET_RUN_PATHS=1
# shellcheck disable=SC1091
source "$CODE_DIR/scripts/RunScripts/preflight_20260711_main_algorithm.sh"

mkdir -p "$LAUNCH_DIR"

_v13_run_tag () {
  printf 'SmolLM2-135M_dir05_r1_e1_lr%s' "${LR:-0.0002}"
}

run_v13_smoke () {
  local method="$1"
  local agg="$2"
  shift 2
  local gpu="${GPU:-1}"
  local launch_log="$LAUNCH_DIR/${method}.launch.log"
  mkdir -p "$RESULT_ROOT/$SMOKE_RUN_ID_20260711/run_logs" "$RESULT_ROOT/$SMOKE_RUN_ID_20260711/pipeline_logs"
  nohup bash "$V13_ONE_EXP" \
    --kind smoke --method "$method" --agg "$agg" --gpu "$gpu" \
    -- "$@" \
    > "$launch_log" 2>&1 &
  local main_log="$RESULT_ROOT/$SMOKE_RUN_ID_20260711/run_logs/test20260711_main_smoke_${method}_seed42.log"
  echo "[run_v13_smoke] method=${method} agg=${agg} gpu=${gpu} pid=$! launch=${launch_log} main=${main_log}"
}

run_v13_sft () {
  local method="$1"
  local agg="$2"
  local seed="$3"
  local run_prefix="$4"
  shift 4
  local gpu="${GPU:-1}"
  local launch_log="$LAUNCH_DIR/${method}.launch.log"
  local run_tag
  run_tag="$(_v13_run_tag)"
  nohup bash "$V13_ONE_EXP" \
    --kind sft --method "$method" --agg "$agg" \
    --seed "$seed" --split-seed "$seed" \
    --run-id-prefix "$run_prefix" --gpu "$gpu" \
    -- "$@" \
    > "$launch_log" 2>&1 &
  local main_log="$RESULT_ROOT/${run_prefix}_seed${seed}/run_logs/test20260711_main_${method}_${run_tag}_seed${seed}.log"
  echo "[run_v13_sft] method=${method} agg=${agg} seed=${seed} gpu=${gpu} pid=$! launch=${launch_log} main=${main_log}"
}

run_v13_sft_split () {
  local method="$1"
  local agg="$2"
  local seed="$3"
  local split_seed="$4"
  local run_prefix="$5"
  shift 5
  local gpu="${GPU:-1}"
  local launch_log="$LAUNCH_DIR/${method}.launch.log"
  local run_tag
  run_tag="$(_v13_run_tag)"
  nohup bash "$V13_ONE_EXP" \
    --kind sft --method "$method" --agg "$agg" \
    --seed "$seed" --split-seed "$split_seed" \
    --run-id-prefix "$run_prefix" --gpu "$gpu" \
    -- "$@" \
    > "$launch_log" 2>&1 &
  local main_log="$RESULT_ROOT/${run_prefix}_seed${seed}/run_logs/test20260711_main_${method}_${run_tag}_seed${seed}.log"
  echo "[run_v13_sft_split] method=${method} agg=${agg} seed=${seed} split_seed=${split_seed} gpu=${gpu} pid=$! launch=${launch_log} main=${main_log}"
}

run_v13_eval () {
  local method="$1"
  local seed="$2"
  local run_prefix="$3"
  shift 3
  local gpu="${GPU:-1}"
  local launch_log="$LAUNCH_DIR/${method}.launch.log"
  nohup bash "$V13_ONE_EXP" \
    --kind personalized_eval --method "$method" \
    --seed "$seed" --split-seed "$seed" \
    --run-id-prefix "$run_prefix" --gpu "$gpu" \
    -- "$@" \
    > "$launch_log" 2>&1 &
  local main_log="$RESULT_ROOT/${run_prefix}_seed${seed}/run_logs/test20260711_main_${method}_seed${seed}.log"
  local out_json="$RESULT_ROOT/${run_prefix}_seed${seed}/result_logs/${method}_seed${seed}.json"
  echo "[run_v13_eval] method=${method} seed=${seed} gpu=${gpu} pid=$! launch=${launch_log} main=${main_log} out=${out_json}"
}

# 简写：默认使用 NX1/NX4/NX3 前缀
run_v13_nx1 () {
  run_v13_sft "$1" "$2" "$3" "$NX1_RUN_ID_PREFIX" "${@:4}"
}

run_v13_nx4 () {
  run_v13_eval "$1" "$2" "$NX4_RUN_ID_PREFIX" "${@:3}"
}

run_v13_nx3 () {
  run_v13_sft_split "$1" "$2" "$3" "$4" "$NX3_RUN_ID_PREFIX" "${@:5}"
}

echo "[preflight_gb_v13][ok] GPU default=${GPU} LAUNCH_DIR=${LAUNCH_DIR}"
echo "[preflight_gb_v13][ok] 用法: export GPU=1 && run_v13_nx1 <method> <agg> <seed> [--force_retrain ...]"
