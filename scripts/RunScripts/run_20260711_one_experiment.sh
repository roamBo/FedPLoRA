#!/usr/bin/env bash
# One foreground experiment runner for 20260711 orders.
#
# Design intent:
#   nohup bash run_20260711_one_experiment.sh ... &
# is exactly one experiment process.  This script writes its own log path and
# pid file, then execs python so the nohup PID becomes the python PID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

KIND=""
METHOD=""
AGG=""
SEED="42"
SPLIT_SEED="42"
RUN_ID_PREFIX=""
GPU=""
LOG_PREFIX="test20260711_main"
SMOKE_LOG_PREFIX="test20260711_main_smoke"
EXTRA_ARGS=()

usage () {
  cat >&2 <<'EOF'
Usage:
  bash scripts/RunScripts/run_20260711_one_experiment.sh \
    --kind sft|smoke|personalized_eval \
    --method METHOD \
    [--agg AGG] \
    [--seed SEED] \
    [--split-seed SPLIT_SEED] \
    [--run-id-prefix PREFIX] \
    [--gpu GPU_ID] \
    [-- extra python args...]
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --kind) KIND="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --agg) AGG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --split-seed) SPLIT_SEED="$2"; shift 2 ;;
    --run-id-prefix) RUN_ID_PREFIX="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --log-prefix) LOG_PREFIX="$2"; shift 2 ;;
    --smoke-log-prefix) SMOKE_LOG_PREFIX="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "[one-exp][error] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$KIND" ] || [ -z "$METHOD" ]; then
  echo "[one-exp][error] --kind and --method are required." >&2
  usage
  exit 2
fi
case "$KIND" in
  sft|smoke|personalized_eval) ;;
  *)
    echo "[one-exp][error] unsupported --kind=$KIND" >&2
    exit 2
    ;;
esac
if [ "$KIND" != "personalized_eval" ] && [ -z "$AGG" ]; then
  echo "[one-exp][error] --agg is required for kind=$KIND" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[one-exp][error] conda not found; run from a shell where conda is initialized." >&2
  exit 2
fi
CONDA_BASE="$(conda info --base)"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

_CLI_RUN_ID_PREFIX="$RUN_ID_PREFIX"
export FEDPLORA_SKIP_DEFAULT_SET_RUN_PATHS=${FEDPLORA_SKIP_DEFAULT_SET_RUN_PATHS:-1}
if [ -f "$SCRIPT_DIR/preflight_20260711_main_algorithm.sh" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/preflight_20260711_main_algorithm.sh"
else
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/preflight_20260709_main_algorithm.sh"
fi
if [ -n "$_CLI_RUN_ID_PREFIX" ]; then
  export RUN_ID_PREFIX="$_CLI_RUN_ID_PREFIX"
fi

export RUN_TAG_DATASET=${RUN_TAG_DATASET:-dir05}
export ROUNDS=${PIPELINE_ROUNDS:-1}
export LOCAL_EPOCHS=${PIPELINE_LOCAL_EPOCHS:-1}
export EVAL_MAX_BATCHES=${PIPELINE_EVAL_MAX_BATCHES:-0}
export LR=${LR:-0.0002}

_FEDPLORA_LOG_PREFIX="$LOG_PREFIX"
_FEDPLORA_SMOKE_LOG_PREFIX="$SMOKE_LOG_PREFIX"

DIR05_ROOT="$(dirname "$BENCHMARK_DIR_MAIN")"
export BENCHMARK_DIR="$DIR05_ROOT/seed_$SPLIT_SEED"
check_benchmark "$BENCHMARK_DIR"

if [ -n "$GPU" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

if [ "$KIND" = "smoke" ]; then
  export SMOKE_RUN_ID=${SMOKE_RUN_ID_20260711:-v13_20260711_smoke_seed42}
  export SMOKE_ROOT="$RESULT_ROOT/$SMOKE_RUN_ID"
  export SMOKE_TRAINED_MODELS_ROOT="$MODEL_ROOT/$SMOKE_RUN_ID"
  mkdir -p "$SMOKE_ROOT/run_logs" "$SMOKE_ROOT/result_logs" "$SMOKE_ROOT/result_files/client_states" "$SMOKE_TRAINED_MODELS_ROOT" "$SMOKE_ROOT/pids"
  LOG_FILE="$SMOKE_ROOT/run_logs/${SMOKE_LOG_PREFIX}_${METHOD}_seed42.log"
  PID_FILE="$SMOKE_ROOT/pids/${METHOD}.pid"

  exec > "$LOG_FILE" 2>&1
  echo "$$" > "$PID_FILE"
  echo "[one-exp] kind=$KIND method=$METHOD agg=$AGG pid=$$ gpu=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[one-exp] log=$LOG_FILE"

  exec python -u tasks/fed_train_sft.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$BENCHMARK_DIR_MAIN" \
    --num_clients 35 \
    --agg_type "$AGG" \
    --rounds 1 --local_epochs 1 --lr "$LR" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
    --client_state_dir "$SMOKE_ROOT/result_files/client_states/$METHOD" \
    --metrics_output_dir "$SMOKE_ROOT/result_logs/$METHOD" \
    --save_run_checkpoint_dir "$SMOKE_TRAINED_MODELS_ROOT/${METHOD}_smoke_seed42" \
    --trained_models_root "$SMOKE_TRAINED_MODELS_ROOT" \
    --eval_max_batches 1 --seed 42 \
    --train_max_steps_per_client 1 \
    --max_train_samples_per_client 10 \
    --save_client_state_to_disk \
    --gradient_checkpointing \
    --eval_personalization_metrics \
    --eval_final_only \
    --skip_post_agg_snapshots \
    "${EXTRA_ARGS[@]}"
fi

if [ -z "$RUN_ID_PREFIX" ]; then
  case "$KIND" in
    sft) RUN_ID_PREFIX="v13_20260711_manual_35c_dir05_r1_finaleval" ;;
    personalized_eval) RUN_ID_PREFIX="v13_20260711_manual_personalized_eval" ;;
  esac
fi
export RUN_ID_PREFIX
set_run_paths "$SEED"
mkdir -p "$RUN_ROOT/pids"
echo "[one-exp] RUN_ID_PREFIX=$RUN_ID_PREFIX RUN_ID=$RUN_ID RUN_ROOT=$RUN_ROOT"

if [ "$KIND" = "personalized_eval" ]; then
  case "$METHOD" in
    X2_*|N2_*) ;;
    *)
      echo "[one-exp][error] personalized_eval method must start with X2_ or N2_: $METHOD" >&2
      exit 2
      ;;
  esac
  LOG_FILE="$RUN_ROOT/run_logs/${LOG_PREFIX}_${METHOD}_seed${SEED}.log"
  PID_FILE="$RUN_ROOT/pids/${METHOD}.pid"
  exec > "$LOG_FILE" 2>&1
  echo "$$" > "$PID_FILE"
  echo "[one-exp] kind=$KIND method=$METHOD seed=$SEED split_seed=$SPLIT_SEED pid=$$ gpu=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[one-exp] log=$LOG_FILE"

  exec python -u scripts/Analysis/eval_personalized.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --target_modules "$TARGET_MODULES" \
    --torch_dtype "$TORCH_DTYPE" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --lr "$LR" --local_epochs "$LOCAL_EPOCHS" \
    --eval_max_batches "$EVAL_MAX_BATCHES" \
    --seed "$SEED" \
    --schemes local,global,v7,v11c,select \
    --select_candidates local,v7,global,v11c \
    --cold_start --eval_on_local \
    "${EXTRA_ARGS[@]}" \
    --out "$RUN_ROOT/result_logs/${METHOD}_seed${SEED}.json"
fi

assert_role_run "$METHOD" "$AGG"
LOG_FILE="$RUN_ROOT/run_logs/${LOG_PREFIX}_${METHOD}_${RUN_TAG}_seed${SEED}.log"
PID_FILE="$RUN_ROOT/pids/${METHOD}.pid"
exec > "$LOG_FILE" 2>&1
echo "$$" > "$PID_FILE"
echo "[one-exp] kind=$KIND method=$METHOD agg=$AGG seed=$SEED split_seed=$SPLIT_SEED pid=$$ gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[one-exp] log=$LOG_FILE"

exec python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --num_clients "$EXPECTED_NUM_CLIENTS" \
  --agg_type "$AGG" \
  --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir "$RUN_ROOT/result_files/client_states/$METHOD" \
  --metrics_output_dir "$RUN_ROOT/result_logs/$METHOD" \
  --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/${METHOD}_${RUN_TAG}_seed${SEED}" \
  --trained_models_root "$TRAINED_MODELS_ROOT" \
  --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
  --save_client_state_to_disk \
  --gradient_checkpointing \
  --eval_personalization_metrics \
  --eval_final_only \
  --skip_post_agg_snapshots \
  "${EXTRA_ARGS[@]}"
