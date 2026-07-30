#!/usr/bin/env bash
# Launch v14 unlearning-dividend synthetic checkpoints as detached eval-only jobs.
#
# Run this launcher itself with nohup.  It writes one child log + one pid file
# per checkpoint and keeps a separate queue log from the parent nohup command.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${CODE_DIR}"

SYNTH_ROOT="${SYNTH_ROOT:-}"
RESULT_ROOT="${RESULT_ROOT:-/data2/minghao/result/FedPLoRA/unlearning_20260730}"
MODEL_PATH="${MODEL_PATH:-/data2/minghao/model/SmolLM2-135M}"
BENCHMARK_DIR="${BENCHMARK_DIR:-}"
GPU_IDS="${GPU_IDS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_MAX_SEQ_LENGTH="${EVAL_MAX_SEQ_LENGTH:-256}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
MATCHED_DOMAIN="${MATCHED_DOMAIN:-0}"
ONLY_TAG_GLOB="${ONLY_TAG_GLOB:-*}"

if [[ -z "${SYNTH_ROOT}" ]]; then
  echo "[unlearning-queue][error] set SYNTH_ROOT to the build_unlearning_dividend_phase0.py output directory" >&2
  exit 2
fi
if [[ -z "${BENCHMARK_DIR}" ]]; then
  echo "[unlearning-queue][error] set BENCHMARK_DIR to the benchmark seed directory" >&2
  exit 2
fi
if [[ ! -d "${SYNTH_ROOT}/checkpoints" ]]; then
  echo "[unlearning-queue][error] missing synthetic checkpoints: ${SYNTH_ROOT}/checkpoints" >&2
  exit 2
fi
SYNTH_NAME="$(basename "${SYNTH_ROOT}")"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "[unlearning-queue][error] GPU_IDS is empty" >&2
  exit 2
fi

RUN_LOG_DIR="${RUN_LOG_DIR:-${RESULT_ROOT}/run_logs/unlearning_phase0/${SYNTH_NAME}}"
METRICS_ROOT="${METRICS_ROOT:-${RESULT_ROOT}/result_logs/unlearning_phase0/${SYNTH_NAME}}"
PID_DIR="${PID_DIR:-${RESULT_ROOT}/pids/unlearning_phase0/${SYNTH_NAME}}"
STATUS_DIR="${STATUS_DIR:-${RESULT_ROOT}/status/unlearning_phase0/${SYNTH_NAME}}"
mkdir -p "${RUN_LOG_DIR}" "${METRICS_ROOT}" "${PID_DIR}" "${STATUS_DIR}"

mapfile -t CKPTS < <(find "${SYNTH_ROOT}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name "${ONLY_TAG_GLOB}" | sort)
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "[unlearning-queue][error] no checkpoint dirs matched ${SYNTH_ROOT}/checkpoints/${ONLY_TAG_GLOB}" >&2
  exit 2
fi

echo "[unlearning-queue] code=${CODE_DIR}"
echo "[unlearning-queue] synth=${SYNTH_ROOT}"
echo "[unlearning-queue] checkpoints=${#CKPTS[@]} max_parallel=${MAX_PARALLEL} gpu_ids=${GPU_IDS}"
echo "[unlearning-queue] result_root=${RESULT_ROOT} eval_max_batches=${EVAL_MAX_BATCHES}"

running_pids=()
running_tags=()
gpu_cursor=0
launched=0
failed=0

prune_finished() {
  local new_pids=()
  local new_tags=()
  local idx pid tag
  for idx in "${!running_pids[@]}"; do
    pid="${running_pids[$idx]}"
    tag="${running_tags[$idx]}"
    if kill -0 "${pid}" 2>/dev/null; then
      new_pids+=("${pid}")
      new_tags+=("${tag}")
    else
      if wait "${pid}" 2>/dev/null; then
        echo "[unlearning-queue][done] tag=${tag} pid=${pid}"
        echo "ok" > "${STATUS_DIR}/${tag}.status"
      else
        echo "[unlearning-queue][fail] tag=${tag} pid=${pid}" >&2
        echo "failed" > "${STATUS_DIR}/${tag}.status"
        failed=$((failed + 1))
      fi
    fi
  done
  running_pids=("${new_pids[@]}")
  running_tags=("${new_tags[@]}")
}

wait_for_slot() {
  while [[ "${#running_pids[@]}" -ge "${MAX_PARALLEL}" ]]; do
    sleep 20
    prune_finished
  done
}

for ckpt in "${CKPTS[@]}"; do
  tag="$(basename "${ckpt}")"
  metrics_dir="${METRICS_ROOT}/${tag}"
  log_file="${RUN_LOG_DIR}/test20260730_unlearning_${tag}.log"
  pid_file="${PID_DIR}/${tag}.pid"
  status_file="${STATUS_DIR}/${tag}.status"

  if find "${metrics_dir}" -maxdepth 1 -type f -name '*.json' -print -quit 2>/dev/null | grep -q .; then
    echo "[unlearning-queue][skip] tag=${tag} metrics already exists"
    echo "skipped_existing" > "${status_file}"
    continue
  fi
  if [[ -f "${pid_file}" ]]; then
    old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "[unlearning-queue][skip] tag=${tag} already running pid=${old_pid}"
      continue
    fi
  fi

  wait_for_slot
  gpu="${GPU_ARRAY[$((gpu_cursor % ${#GPU_ARRAY[@]}))]}"
  gpu_cursor=$((gpu_cursor + 1))
  mkdir -p "${metrics_dir}"

  cmd=(
    python -u tasks/fed_train_sft.py
    --model "${MODEL_PATH}"
    --benchmark_dir "${BENCHMARK_DIR}"
    --agg_type fedalt
    --eval_only_from_checkpoint "${ckpt}"
    --metrics_output_dir "${metrics_dir}"
    --batch_size "${EVAL_BATCH_SIZE}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --max_seq_length "${EVAL_MAX_SEQ_LENGTH}"
    --eval_max_batches "${EVAL_MAX_BATCHES}"
    --torch_dtype "${TORCH_DTYPE}"
    --target_modules "${TARGET_MODULES}"
    --eval_personalization_metrics
  )
  if [[ "${MATCHED_DOMAIN}" == "1" ]]; then
    cmd+=(--eval_only_matched_domain)
  fi
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    cmd+=(--trust_remote_code)
  fi

  echo "[unlearning-queue][launch] tag=${tag} gpu=${gpu} log=${log_file}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" /usr/bin/time -v "${cmd[@]}" > "${log_file}" 2>&1 &
  pid="$!"
  echo "${pid}" > "${pid_file}"
  echo "running" > "${status_file}"
  running_pids+=("${pid}")
  running_tags+=("${tag}")
  launched=$((launched + 1))
done

while [[ "${#running_pids[@]}" -gt 0 ]]; do
  sleep 20
  prune_finished
done

echo "[unlearning-queue][finished] launched=${launched} failed=${failed}"
if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
