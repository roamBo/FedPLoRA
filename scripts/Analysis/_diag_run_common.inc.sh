# shellcheck shell=bash
# Shared runner: sequential diag_subspace → diag_subspace_AB → diag_b_swap.
# Caller must set: DIAG_MODEL_TAG, MODEL_PATH, TARGET_MODULES
# Optional env: REPO_ROOT, BENCHMARK_DIR, SEED, MAX_STEPS, BATCH_SIZE, MAX_SEQ_LENGTH,
#               LORA_R, LORA_ALPHA, LORA_DROPOUT, TORCH_DTYPE, CUDA_VISIBLE_DEVICES
_diag_run_all() {
  local gpu="${1:-${CUDA_VISIBLE_DEVICES:-1}}"
  local tag="${DIAG_MODEL_TAG:?set DIAG_MODEL_TAG}"
  local model_path="${MODEL_PATH:?set MODEL_PATH}"
  local target_modules="${TARGET_MODULES:?set TARGET_MODULES}"

  local repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
  cd "${repo_root}"

  local benchmark_dir="${BENCHMARK_DIR:-data/domain_benchmark_35c/seed_42}"
  local seed="${SEED:-42}"
  local max_steps="${MAX_STEPS:-0}"
  local batch_size="${BATCH_SIZE:-1}"
  local max_seq="${MAX_SEQ_LENGTH:-512}"
  local lora_r="${LORA_R:-8}"
  local lora_alpha="${LORA_ALPHA:-16}"
  local lora_dropout="${LORA_DROPOUT:-0.05}"
  local torch_dtype="${TORCH_DTYPE:-bfloat16}"

  if [[ ! -f "${model_path}/config.json" ]]; then
    echo "[diag][${tag}] ERROR: model not found: ${model_path}" >&2
    echo "[diag][${tag}] hint: export MODEL_ROOT=/data/yaominghao/gb/models" >&2
    exit 1
  fi

  mkdir -p log_diag artifacts_35c/diag

  echo "[diag][${tag}] gpu=${gpu} model=${model_path} benchmark=${benchmark_dir} max_steps=${max_steps}"

  echo "[diag][${tag}] (1/3) diag_subspace.py ..."
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/Analysis/diag_subspace.py \
    --model "${model_path}" \
    --benchmark_dir "${benchmark_dir}" \
    --lora_r "${lora_r}" --lora_alpha "${lora_alpha}" --lora_dropout "${lora_dropout}" \
    --batch_size "${batch_size}" --max_seq_length "${max_seq}" \
    --torch_dtype "${torch_dtype}" --target_modules "${target_modules}" \
    --gradient_checkpointing \
    --max_steps "${max_steps}" --seed "${seed}" \
    --out "artifacts_35c/diag/${tag}_diag_subspace_A_seed${seed}.json" \
    --save_figs \
    2>&1 | tee "log_diag/${tag}_diag_subspace_A_seed${seed}.log"

  echo "[diag][${tag}] (2/3) diag_subspace_AB.py ..."
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/Analysis/diag_subspace_AB.py \
    --model "${model_path}" \
    --benchmark_dir "${benchmark_dir}" \
    --lora_r "${lora_r}" --lora_alpha "${lora_alpha}" --lora_dropout "${lora_dropout}" \
    --batch_size "${batch_size}" --max_seq_length "${max_seq}" \
    --torch_dtype "${torch_dtype}" --target_modules "${target_modules}" \
    --gradient_checkpointing \
    --max_steps "${max_steps}" --seed "${seed}" \
    --out "artifacts_35c/diag/${tag}_diag_AB_seed${seed}.json" \
    --save_figs \
    2>&1 | tee "log_diag/${tag}_diag_AB_seed${seed}.log"

  echo "[diag][${tag}] (3/3) diag_b_swap.py ..."
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/Analysis/diag_b_swap.py \
    --model "${model_path}" \
    --benchmark_dir "${benchmark_dir}" \
    --lora_r "${lora_r}" --lora_alpha "${lora_alpha}" --lora_dropout "${lora_dropout}" \
    --batch_size "${batch_size}" --max_seq_length "${max_seq}" \
    --torch_dtype "${torch_dtype}" --target_modules "${target_modules}" \
    --gradient_checkpointing \
    --max_steps "${max_steps}" --seed "${seed}" \
    --eval_max_batches 20 --n_peers 4 --n_cross 2 \
    --out "artifacts_35c/diag/${tag}_diag_b_swap_seed${seed}.json" \
    2>&1 | tee "log_diag/${tag}_diag_b_swap_seed${seed}.log"

  echo "[diag][${tag}] done."
  echo "[diag][${tag}] logs: log_diag/${tag}_diag_*_seed${seed}.log"
  echo "[diag][${tag}] json:  artifacts_35c/diag/${tag}_diag_*_seed${seed}.json"
}
