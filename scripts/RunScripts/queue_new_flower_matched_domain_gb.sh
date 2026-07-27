#!/usr/bin/env bash
# Queue new Flower baseline matched-domain evals on gb without blocking or
# launching 18 GPU jobs at once.
#
# Do not source this file. Run it with bash:
#   bash scripts/RunScripts/queue_new_flower_matched_domain_gb.sh launch
#   bash scripts/RunScripts/queue_new_flower_matched_domain_gb.sh status

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASELINE_RESULT_ROOT="${BASELINE_RESULT_ROOT:-${RESULT_ROOT:-/data/yaominghao/gb/result/FedPLoRA/order_0725/baseline}}"
NEW_MD_ROOT="${NEW_MD_ROOT:-${BASELINE_RESULT_ROOT}/matched_domain_new_flower}"
GPU_ID="${GPU_ID:-0}"
POLL_SECONDS="${POLL_SECONDS:-30}"

MD_LAUNCHER="${MD_LAUNCHER:-${CODE_DIR}/scripts/RunScripts/launch_eval_only_matched_domain_one.sh}"
MD_STATUS="${MD_STATUS:-${CODE_DIR}/scripts/RunScripts/check_eval_only_matched_domain_jobs.sh}"
MD_LOG_DIR="${MD_LOG_DIR:-${NEW_MD_ROOT}/logs}"
MD_PID_DIR="${MD_PID_DIR:-${NEW_MD_ROOT}/pids}"
MD_META_DIR="${MD_META_DIR:-${NEW_MD_ROOT}/meta}"
QUEUE_DIR="${QUEUE_DIR:-${NEW_MD_ROOT}/queue}"
QUEUE_LOG="${QUEUE_LOG:-${QUEUE_DIR}/new_flower_matched_domain_queue.log}"
QUEUE_PID="${QUEUE_PID:-${QUEUE_DIR}/new_flower_matched_domain_queue.pid}"

TAGS=(
  flower_yoco_seed42
  flower_yoco_seed43
  flower_yoco_seed44
  flower_ffa_seed42
  flower_ffa_seed43
  flower_ffa_seed44
  flower_flora_seed42
  flower_flora_seed43
  flower_flora_seed44
  flower_flexlora_seed42
  flower_flexlora_seed43
  flower_flexlora_seed44
  flower_feddat_seed42
  flower_feddat_seed43
  flower_feddat_seed44
  flower_hilora_seed42
  flower_hilora_seed43
  flower_hilora_seed44
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/RunScripts/queue_new_flower_matched_domain_gb.sh launch
  bash scripts/RunScripts/queue_new_flower_matched_domain_gb.sh worker
  bash scripts/RunScripts/queue_new_flower_matched_domain_gb.sh status
  bash scripts/RunScripts/queue_new_flower_matched_domain_gb.sh summarize

Env overrides:
  CODE_DIR=/data/yaominghao/gb/FedPLoRA
  BASELINE_RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0725/baseline
  NEW_MD_ROOT=$BASELINE_RESULT_ROOT/matched_domain_new_flower
  GPU_ID=0

The launch command starts one background queue process and returns immediately.
The queue itself runs the 18 matched-domain jobs serially on GPU_ID.
EOF
}

die() {
  echo "[new-flower-md-queue][error] $*" >&2
  exit 1
}

init_dirs() {
  cd "$CODE_DIR" || die "missing CODE_DIR=$CODE_DIR"
  mkdir -p "$NEW_MD_ROOT" "$MD_LOG_DIR" "$MD_PID_DIR" "$MD_META_DIR" "$QUEUE_DIR"
  test -f "$MD_LAUNCHER" || die "missing launcher: $MD_LAUNCHER"
  test -f "$MD_STATUS" || die "missing status script: $MD_STATUS"
}

find_source_json() {
  local tag="$1"
  local root="${BASELINE_RESULT_ROOT}/${tag}/result_logs"
  if [[ ! -d "$root" ]]; then
    die "missing result_logs for ${tag}: ${root}"
  fi
  mapfile -t hits < <(find "$root" -type f -name '*.json' | sort)
  if [[ "${#hits[@]}" -ne 1 ]]; then
    printf '[new-flower-md-queue][error] expected one JSON for %s, got %s under %s\n' \
      "$tag" "${#hits[@]}" "$root" >&2
    printf '%s\n' "${hits[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${hits[0]}"
}

wait_one_eval() {
  local tag="$1"
  local pid_file="${MD_PID_DIR}/md_${tag}.pid"
  local log_file="${MD_LOG_DIR}/md_${tag}.log"
  local pid=""

  for _ in {1..20}; do
    if [[ -s "$pid_file" ]]; then
      pid="$(cat "$pid_file")"
      break
    fi
    sleep 1
  done
  [[ -n "$pid" ]] || die "launcher did not write pid file: $pid_file"

  echo "[new-flower-md-queue][wait] tag=${tag} pid=${pid} log=${log_file}"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done

  if grep -Eq '\[matched-domain-eval\] done=[0-9]+ skipped=[0-9]+ failed=0' "$log_file"; then
    echo "[new-flower-md-queue][ok] tag=${tag}"
    return 0
  fi

  echo "[new-flower-md-queue][failed] tag=${tag} log=${log_file}" >&2
  tail -80 "$log_file" >&2 || true
  exit 1
}

worker() {
  init_dirs
  echo "[new-flower-md-queue][start] root=${NEW_MD_ROOT} gpu=${GPU_ID} tags=${#TAGS[@]}"
  date '+[new-flower-md-queue][time] %Y-%m-%dT%H:%M:%S%z'
  for tag in "${TAGS[@]}"; do
    local source_json
    source_json="$(find_source_json "$tag")"
    echo "[new-flower-md-queue][launch] tag=${tag} source=${source_json}"
    MD_ROOT="$NEW_MD_ROOT" MD_LOG_DIR="$MD_LOG_DIR" MD_PID_DIR="$MD_PID_DIR" MD_META_DIR="$MD_META_DIR" \
      bash "$MD_LAUNCHER" "md_${tag}" "$source_json" "$NEW_MD_ROOT" "$GPU_ID"
    wait_one_eval "$tag"
  done
  echo "[new-flower-md-queue][done] all ${#TAGS[@]} jobs finished"
  "$MD_STATUS" "$NEW_MD_ROOT"
}

launch() {
  init_dirs
  if [[ -s "$QUEUE_PID" ]]; then
    local old_pid
    old_pid="$(cat "$QUEUE_PID" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[new-flower-md-queue][skip] already running pid=${old_pid} log=${QUEUE_LOG}"
      exit 0
    fi
  fi
  nohup env \
    CODE_DIR="$CODE_DIR" \
    BASELINE_RESULT_ROOT="$BASELINE_RESULT_ROOT" \
    NEW_MD_ROOT="$NEW_MD_ROOT" \
    GPU_ID="$GPU_ID" \
    POLL_SECONDS="$POLL_SECONDS" \
    MD_LAUNCHER="$MD_LAUNCHER" \
    MD_STATUS="$MD_STATUS" \
    MD_LOG_DIR="$MD_LOG_DIR" \
    MD_PID_DIR="$MD_PID_DIR" \
    MD_META_DIR="$MD_META_DIR" \
    bash "$0" worker > "$QUEUE_LOG" 2>&1 &
  echo $! > "$QUEUE_PID"
  echo "[new-flower-md-queue][launched] pid=$(cat "$QUEUE_PID") log=${QUEUE_LOG}"
  echo "[new-flower-md-queue][status] bash scripts/RunScripts/queue_new_flower_matched_domain_gb.sh status"
}

status() {
  init_dirs
  if [[ -s "$QUEUE_PID" ]] && kill -0 "$(cat "$QUEUE_PID")" 2>/dev/null; then
    echo "[new-flower-md-queue][running] pid=$(cat "$QUEUE_PID") log=${QUEUE_LOG}"
  else
    echo "[new-flower-md-queue][not-running] log=${QUEUE_LOG}"
  fi
  "$MD_STATUS" "$NEW_MD_ROOT" || true
  count="$(find "$NEW_MD_ROOT" -name '*_matched_domain.json' 2>/dev/null | wc -l | xargs)"
  echo "[new-flower-md-queue][json] matched_domain_json=${count}/18"
}

summarize() {
  init_dirs
  python scripts/Analysis/summarize_matched_domain_eval.py "$NEW_MD_ROOT" \
    | tee "$NEW_MD_ROOT/new_flower_baselines_summary.tsv"
  grep -R 'max_seq_length=256' "$MD_LOG_DIR" || true
}

cmd="${1:-help}"
case "$cmd" in
  launch) launch ;;
  worker) worker ;;
  status) status ;;
  summarize) summarize ;;
  help|-h|--help) usage ;;
  *) usage; exit 2 ;;
esac
