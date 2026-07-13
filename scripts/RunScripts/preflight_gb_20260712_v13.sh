#!/usr/bin/env bash
# gb 服务器 20260712 order 专用 preflight + 后台运行函数。
# Usage:
#   cd /data/yaominghao/gb/FedPLoRA
#   source scripts/RunScripts/preflight_gb_20260712_v13.sh
#   export GPU=1
#   run_v13_smoke smoke_v13a_os fedplora_v13a_os --force_retrain

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
export MODEL_ROOT=${MODEL_ROOT:-/data/yaominghao/gb/models}
export MODEL_PATH=${MODEL_PATH:-$MODEL_ROOT/SmolLM2-135M}
export DATA_ROOT=${DATA_ROOT:-$CODE_DIR/data}
export RESULT_ROOT=${RESULT_ROOT:-/data/yaominghao/gb/result/FedPLoRA}
export BENCHMARK_DIR_MAIN=${BENCHMARK_DIR_MAIN:-$DATA_ROOT/domain_benchmark_35c_dir05/seed_42}
export TARGET_MODULES=${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
export GPU=${GPU:-1}

export SMOKE_RUN_ID_20260712=${SMOKE_RUN_ID_20260712:-v13_20260712_smoke_seed42}
export HELDOUT_SMOKE_RUN_ID_PREFIX=${HELDOUT_SMOKE_RUN_ID_PREFIX:-v13_20260712_smoke}
export NX0_RUN_ID_PREFIX=${NX0_RUN_ID_PREFIX:-v13_20260712_nx0_35c_dir05_r1_finaleval}
export HELDOUT_RUN_ID_PREFIX=${HELDOUT_RUN_ID_PREFIX:-v13_20260712_strict_heldout_split42}
export FINGERPRINT_RUN_ID=${FINGERPRINT_RUN_ID:-v13_20260712_fingerprint}

export LAUNCH_DIR=${LAUNCH_DIR:-$RESULT_ROOT/manual_launch_logs_20260712}
export V13_ONE_EXP="$CODE_DIR/scripts/RunScripts/run_20260712_one_experiment.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "[preflight_gb_v13_12][error] conda 不可用。" >&2
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
  mkdir -p "$RESULT_ROOT/$SMOKE_RUN_ID_20260712/run_logs" "$RESULT_ROOT/$SMOKE_RUN_ID_20260712/pipeline_logs"
  nohup bash "$V13_ONE_EXP" \
    --kind smoke --method "$method" --agg "$agg" --gpu "$gpu" \
    -- "$@" \
    > "$launch_log" 2>&1 &
  local main_log="$RESULT_ROOT/$SMOKE_RUN_ID_20260712/run_logs/test20260712_main_smoke_${method}_seed42.log"
  echo "[run_v13_smoke] method=${method} agg=${agg} gpu=${gpu} pid=$! launch=${launch_log} main=${main_log}"
}

run_v13_heldout_smoke () {
  local method="${1:-X2_strict_heldout_smoke_seed42}"
  shift "$#" 2>/dev/null || true
  local gpu="${GPU:-1}"
  local launch_log="$LAUNCH_DIR/${method}.launch.log"
  mkdir -p "$RESULT_ROOT/${HELDOUT_SMOKE_RUN_ID_PREFIX}_seed42/run_logs" "$RESULT_ROOT/$SMOKE_RUN_ID_20260712/pipeline_logs"
  PIPELINE_EVAL_MAX_BATCHES=1 PIPELINE_ROUNDS=1 \
  nohup bash "$V13_ONE_EXP" \
    --kind personalized_eval --method "$method" \
    --seed 42 --split-seed 42 \
    --run-id-prefix "$HELDOUT_SMOKE_RUN_ID_PREFIX" --gpu "$gpu" \
    -- --held_out_clients auto_one_per_domain \
       --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
       --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
       --few_shot_caps 5 \
       --held_out_route_probe_samples 5 \
       --eval_on_local \
       --cold_start \
       --max_steps 1 \
       --v11c_mu 0.4 \
       "$@" \
    > "$launch_log" 2>&1 &
  local main_log="$RESULT_ROOT/${HELDOUT_SMOKE_RUN_ID_PREFIX}_seed42/run_logs/test20260712_main_${method}_seed42.log"
  echo "[run_v13_heldout_smoke] method=${method} gpu=${gpu} pid=$! launch=${launch_log} main=${main_log}"
}

run_v13_fingerprint () {
  local launch_log="$RESULT_ROOT/$FINGERPRINT_RUN_ID/pipeline_logs/fingerprint_3splits.log"
  mkdir -p "$RESULT_ROOT/$FINGERPRINT_RUN_ID/pipeline_logs" "$RESULT_ROOT/$FINGERPRINT_RUN_ID/fingerprints"
  nohup bash -lc "
set -euo pipefail
cd '$CODE_DIR'
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
ROOT=\"\$(dirname \"\$BENCHMARK_DIR_MAIN\")\"
for SEED in 42 43 44; do
  SPLIT=\"\$ROOT/seed_\$SEED\"
  check_benchmark \"\$SPLIT\"
  python utilities/benchmark_fingerprint.py \"\$SPLIT\" \
    --output '$RESULT_ROOT/$FINGERPRINT_RUN_ID/fingerprints/seed_\${SEED}.json'
done
" > "$launch_log" 2>&1 &
  echo "[run_v13_fingerprint] pid=$! launch=${launch_log} out=$RESULT_ROOT/$FINGERPRINT_RUN_ID/fingerprints/seed_{42,43,44}.json"
}

run_v13_nx0 () {
  local method="$1"
  local agg="$2"
  shift 2
  local gpu="${GPU:-1}"
  local launch_log="$LAUNCH_DIR/${method}.launch.log"
  local run_tag
  run_tag="$(_v13_run_tag)"
  PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 \
  nohup bash "$V13_ONE_EXP" \
    --kind sft --method "$method" --agg "$agg" \
    --seed 42 --split-seed 42 \
    --run-id-prefix "$NX0_RUN_ID_PREFIX" --gpu "$gpu" \
    -- "$@" \
    > "$launch_log" 2>&1 &
  local main_log="$RESULT_ROOT/${NX0_RUN_ID_PREFIX}_seed42/run_logs/test20260712_main_${method}_${run_tag}_seed42.log"
  echo "[run_v13_nx0] method=${method} agg=${agg} gpu=${gpu} pid=$! launch=${launch_log} main=${main_log}"
}

run_v13_heldout () {
  local method="${1:-X2_strict_heldout_seed42}"
  shift "$#" 2>/dev/null || true
  local gpu="${GPU:-1}"
  local launch_log="$LAUNCH_DIR/${method}.launch.log"
  PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 \
  nohup bash "$V13_ONE_EXP" \
    --kind personalized_eval --method "$method" \
    --seed 42 --split-seed 42 \
    --run-id-prefix "$HELDOUT_RUN_ID_PREFIX" --gpu "$gpu" \
    -- --held_out_clients auto_one_per_domain \
       --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
       --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
       --few_shot_caps 5,10 \
       --held_out_route_probe_samples 10 \
       --eval_on_local \
       --cold_start \
       --v11c_mu 0.4 \
       "$@" \
    > "$launch_log" 2>&1 &
  local main_log="$RESULT_ROOT/${HELDOUT_RUN_ID_PREFIX}_seed42/run_logs/test20260712_main_${method}_seed42.log"
  local out_json="$RESULT_ROOT/${HELDOUT_RUN_ID_PREFIX}_seed42/result_logs/${method}_seed42.json"
  echo "[run_v13_heldout] method=${method} gpu=${gpu} pid=$! launch=${launch_log} main=${main_log} out=${out_json}"
}

echo "[preflight_gb_v13_12][ok] GPU default=${GPU} LAUNCH_DIR=${LAUNCH_DIR}"
echo "[preflight_gb_v13_12][ok] 用法: export GPU=1 && run_v13_nx0 <method> <agg> [--force_retrain ...]"
