#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-/data2/minghao/code/FedPLoRA-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/minghao/anaconda3/envs/FedRepo2/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data2/minghao/model/SmolLM2-135M}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42}"
RUN_ROOT="${RUN_ROOT:-/data2/minghao/result/FedPLoRA/method_ablation_20260728_d1_seed42_gpu0}"
GPU_ID="${GPU_ID:-0}"

cd "$CODE_DIR"
mkdir -p "$RUN_ROOT"/{logs,pids,rm1_route_metrics,m1_hard_vs_soft,m0_shared_init}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[launch][error] missing python: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$BENCHMARK_DIR/clients.json" ]]; then
  echo "[launch][error] missing benchmark clients.json: $BENCHMARK_DIR/clients.json" >&2
  exit 1
fi

common_args=(
  --model "$MODEL_PATH"
  --benchmark_dir "$BENCHMARK_DIR"
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --max_seq_length "${MAX_SEQ_LENGTH:-256}"
  --batch_size "${BATCH_SIZE:-2}"
  --eval_batch_size "${EVAL_BATCH_SIZE:-0}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --local_epochs "${LOCAL_EPOCHS:-1}"
  --lr "${LR:-0.0002}"
  --max_steps "${MAX_STEPS:-0}"
  --max_train_samples_per_client "${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}"
  --eval_max_batches "${EVAL_MAX_BATCHES:-0}"
  --seed "${SEED:-42}"
  --held_out_clients auto_one_per_domain
  --held_out_policy offset
  --held_out_offset "${HELD_OUT_OFFSET:-0}"
  --few_shot_caps "${FEW_SHOT_CAPS:-10}"
  --held_out_route_probe_samples "${HELD_OUT_ROUTE_PROBE_SAMPLES:-10}"
  --schemes base,global,coldstart
  --onboarding_accounting
)

launch_eval () {
  local name="$1"; shift
  local out_json="$1"; shift
  local log="$RUN_ROOT/logs/${name}.log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" nohup /usr/bin/time -v "$PYTHON_BIN" \
    scripts/Analysis/eval_personalized.py "${common_args[@]}" "$@" \
    --out "$out_json" > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_ROOT/pids/${name}.pid"
  echo "[launch] $name pid=$pid log=$log"
}

# RM1: principal-angle/subspace router against raw factor/delta/retrieval/random/oracle arms.
launch_eval \
  rm1_route_metrics \
  "$RUN_ROOT/rm1_route_metrics/rm1_route_metrics.json" \
  --client_init_mode shared \
  --held_out_route_metrics flat_b_cosine,subspace,relative_l2,delta_w_cosine,nearest_client_subspace,random,oracle

# M1: hard top-1 subspace routing vs soft top-2 subspace B-expert mixture.
launch_eval \
  m1_hard_vs_soft \
  "$RUN_ROOT/m1_hard_vs_soft/m1_hard_vs_soft.json" \
  --client_init_mode shared \
  --held_out_route_metrics subspace,oracle \
  --held_out_soft_route_metrics subspace \
  --held_out_soft_route_temperature "${SOFT_ROUTE_TEMPERATURE:-0.1}"

# M0: shared initialization vs independent per-client initialization in one PID/log.
M0_LOG="$RUN_ROOT/logs/m0_shared_vs_independent.log"
CUDA_VISIBLE_DEVICES="$GPU_ID" RUN_ROOT="$RUN_ROOT" OUT_DIR="$RUN_ROOT/m0_shared_init" \
  CODE_DIR="$CODE_DIR" PYTHON_BIN="$PYTHON_BIN" MODEL_PATH="$MODEL_PATH" BENCHMARK_DIR="$BENCHMARK_DIR" \
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}" BATCH_SIZE="${BATCH_SIZE:-2}" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-0}" \
  TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}" LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}" LR="${LR:-0.0002}" \
  MAX_STEPS="${MAX_STEPS:-0}" MAX_TRAIN_SAMPLES_PER_CLIENT="${MAX_TRAIN_SAMPLES_PER_CLIENT:-0}" \
  EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}" SEED="${SEED:-42}" HELD_OUT_OFFSET="${HELD_OUT_OFFSET:-0}" \
  FEW_SHOT_CAPS="${FEW_SHOT_CAPS:-10}" HELD_OUT_ROUTE_PROBE_SAMPLES="${HELD_OUT_ROUTE_PROBE_SAMPLES:-10}" \
  nohup bash scripts/RunScripts/run_m0_shared_vs_independent_20260728.sh \
  > "$M0_LOG" 2>&1 &
M0_PID=$!
echo "$M0_PID" > "$RUN_ROOT/pids/m0_shared_vs_independent.pid"
echo "[launch] m0_shared_vs_independent pid=$M0_PID log=$M0_LOG"

echo "[launch][ok] all jobs submitted on CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "[launch][root] $RUN_ROOT"
