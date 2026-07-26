#!/usr/bin/env bash
# Report detached matched-domain eval-only job status from pid/log files.
#
# Usage:
#   bash scripts/RunScripts/check_eval_only_matched_domain_jobs.sh /path/to/MD_ROOT

set -euo pipefail

MD_ROOT="${1:-${MD_ROOT:-}}"
if [[ -z "${MD_ROOT}" ]]; then
  echo "Usage: $0 <matched-domain-root>" >&2
  exit 2
fi

LOG_DIR="${MD_LOG_DIR:-${MD_ROOT}/logs}"
PID_DIR="${MD_PID_DIR:-${MD_ROOT}/pids}"

if [[ ! -d "${PID_DIR}" ]]; then
  echo "[md-status][error] missing pid dir: ${PID_DIR}" >&2
  exit 2
fi

running=0
ok=0
failed=0
unknown=0

shopt -s nullglob
for pid_file in "${PID_DIR}"/*.pid; do
  tag="$(basename "${pid_file}" .pid)"
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  log_file="${LOG_DIR}/${tag}.log"

  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    running=$((running + 1))
    etime="$(ps -p "${pid}" -o etime= 2>/dev/null | xargs || true)"
    echo "[md-status][running] tag=${tag} pid=${pid} etime=${etime} log=${log_file}"
    continue
  fi

  if [[ ! -f "${log_file}" ]]; then
    unknown=$((unknown + 1))
    echo "[md-status][unknown] tag=${tag} pid=${pid} missing_log=${log_file}"
    continue
  fi

  if grep -Eq '\[matched-domain-eval\] done=[0-9]+ skipped=[0-9]+ failed=0' "${log_file}"; then
    ok=$((ok + 1))
    echo "[md-status][ok] tag=${tag} log=${log_file}"
  elif grep -Eiq 'Traceback|CUDA out of memory|\\[error\\]|failed=[1-9][0-9]*|Exit status: [1-9][0-9]*' "${log_file}"; then
    failed=$((failed + 1))
    echo "[md-status][failed] tag=${tag} log=${log_file}"
    tail -20 "${log_file}" | sed 's/^/  | /'
  else
    unknown=$((unknown + 1))
    echo "[md-status][exited_unknown] tag=${tag} pid=${pid} log=${log_file}"
    tail -12 "${log_file}" | sed 's/^/  | /'
  fi
done

json_count="$(find "${MD_ROOT}" -type f -name '*_matched_domain.json' 2>/dev/null | wc -l | xargs)"
echo "[md-status][summary] running=${running} ok=${ok} failed=${failed} unknown=${unknown} matched_json=${json_count}"

if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
