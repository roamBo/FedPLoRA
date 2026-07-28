#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-/data2/minghao/code/FedPLoRA-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/minghao/anaconda3/envs/FedRepo2/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data2/minghao/model/SmolLM2-135M}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42}"
OUT_DIR="${OUT_DIR:-/data2/minghao/result/FedPLoRA/method_ablation_20260728_d1_seed42_gpu0/m0_shared_init}"

cd "$CODE_DIR"
mkdir -p "$OUT_DIR"

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
  --held_out_route_metrics subspace,random,oracle
  --schemes base,global,coldstart
  --onboarding_accounting
)

for mode in shared independent; do
  echo "[M0] start client_init_mode=$mode out=$OUT_DIR/m0_${mode}.json"
  /usr/bin/time -v "$PYTHON_BIN" scripts/Analysis/eval_personalized.py \
    "${common_args[@]}" \
    --client_init_mode "$mode" \
    --out "$OUT_DIR/m0_${mode}.json"
  echo "[M0] done client_init_mode=$mode"
done

echo "[M0] all_done out_dir=$OUT_DIR"
