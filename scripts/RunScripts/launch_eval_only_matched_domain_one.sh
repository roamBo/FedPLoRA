#!/usr/bin/env bash
# Launch one matched-domain eval-only job as a detached nohup process.
#
# Usage:
#   MD_ROOT=/path/to/eval_only_root \
#   bash scripts/RunScripts/launch_eval_only_matched_domain_one.sh \
#     <tag> <formal-result.json> <output-dir> <gpu-id>
#
# This script starts the job, writes a pid/log/meta file, and exits immediately.
# Use check_eval_only_matched_domain_jobs.sh to monitor progress later.

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <tag> <formal-result.json> <output-dir> <gpu-id>" >&2
  exit 2
fi

TAG="$1"
SOURCE_JSON="$2"
OUTPUT_DIR="$3"
GPU_ID="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MD_ROOT="${MD_ROOT:-${OUTPUT_DIR%/*}}"
MD_RUNNER="${MD_RUNNER:-${REPO_ROOT}/scripts/RunScripts/run_eval_only_matched_domain.sh}"
LOG_DIR="${MD_LOG_DIR:-${MD_ROOT}/logs}"
PID_DIR="${MD_PID_DIR:-${MD_ROOT}/pids}"
META_DIR="${MD_META_DIR:-${MD_ROOT}/meta}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${PID_DIR}" "${META_DIR}"

if [[ ! -f "${SOURCE_JSON}" ]]; then
  echo "[launch-md][error] missing source JSON: ${SOURCE_JSON}" >&2
  exit 2
fi
if [[ ! -f "${MD_RUNNER}" ]]; then
  echo "[launch-md][error] missing runner: ${MD_RUNNER}" >&2
  exit 2
fi

PID_FILE="${PID_DIR}/${TAG}.pid"
LOG_FILE="${LOG_DIR}/${TAG}.log"
META_FILE="${META_DIR}/${TAG}.meta"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[launch-md][skip] tag=${TAG} already running pid=${OLD_PID} log=${LOG_FILE}"
    exit 0
  fi
fi

{
  echo "tag=${TAG}"
  echo "source_json=${SOURCE_JSON}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "gpu=${GPU_ID}"
  echo "runner=${MD_RUNNER}"
  echo "log=${LOG_FILE}"
  date +"launched_at=%Y-%m-%dT%H:%M:%S%z"
} > "${META_FILE}"

nohup env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}" \
  EVAL_MAX_SEQ_LENGTH="${EVAL_MAX_SEQ_LENGTH:-256}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}" \
  EVAL_TORCH_DTYPE="${EVAL_TORCH_DTYPE:-bfloat16}" \
  MATCHED_DOMAIN_OUTPUT_ROOT="${OUTPUT_DIR}" \
  bash "${MD_RUNNER}" "${SOURCE_JSON}" \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"
disown "${PID}" 2>/dev/null || true

echo "[launch-md][ok] tag=${TAG} pid=${PID}"
echo "[launch-md][log] ${LOG_FILE}"
echo "[launch-md][pid] ${PID_FILE}"
