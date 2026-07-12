#!/usr/bin/env bash
# 2026-07-11 one-shot staged pipeline controller.
#
# This controller does NOT run smoke by default.  Smoke is a separate set of
# one-experiment nohup commands in order_20260711.md.  The controller:
#   1) checks smoke logs unless REQUIRE_SMOKE_OK=0;
#   2) runs NX2 md5 audit;
#   3) launches same-stage NX1 experiments in parallel, one nohup PID per run;
#   4) parses metrics and gates the next stage by threshold;
#   5) optionally launches NX4 / NX3 after the gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONE_EXP="$SCRIPT_DIR/run_20260711_one_experiment.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "[pipeline][error] conda not found; run from a shell where conda is initialized." >&2
  exit 2
fi
CONDA_BASE="$(conda info --base)"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/preflight_20260709_main_algorithm.sh"

export RUN_TAG_DATASET=${RUN_TAG_DATASET:-dir05}
export PIPELINE_ROUNDS=${PIPELINE_ROUNDS:-1}
export PIPELINE_LOCAL_EPOCHS=${PIPELINE_LOCAL_EPOCHS:-1}
export PIPELINE_EVAL_MAX_BATCHES=${PIPELINE_EVAL_MAX_BATCHES:-0}
export FULL_EVAL_MAX_BATCHES=${FULL_EVAL_MAX_BATCHES:-$PIPELINE_EVAL_MAX_BATCHES}
export LR=${LR:-0.0002}

export NX1_RUN_ID_PREFIX=${NX1_RUN_ID_PREFIX:-v13_20260711_nx1_35c_dir05_r1_finaleval}
export NX4_RUN_ID_PREFIX=${NX4_RUN_ID_PREFIX:-v13_20260711_nx4_personalized_eval}
export NX3_RUN_ID_PREFIX=${NX3_RUN_ID_PREFIX:-v13_20260711_nx3_ablation_split42_r1_finaleval}

export PIPELINE_RUN_ID=${PIPELINE_RUN_ID:-v13_20260711_pipeline_control_$(date +%Y%m%d_%H%M%S)}
export PIPELINE_ROOT="$RESULT_ROOT/$PIPELINE_RUN_ID"
mkdir -p "$PIPELINE_ROOT/pipeline_logs" "$PIPELINE_ROOT/pids" "$PIPELINE_ROOT/gates"
echo "$$" > "$PIPELINE_ROOT/pids/pipeline.pid"

read -r -a _FEDPLORA_GPUS <<< "${GPU_LIST:-0 1}"
if [ "${#_FEDPLORA_GPUS[@]}" -lt 1 ]; then
  _FEDPLORA_GPUS=(0)
fi
export MAX_PARALLEL=${MAX_PARALLEL:-${#_FEDPLORA_GPUS[@]}}
_FEDPLORA_GPU_CURSOR=0

timestamp () {
  date "+%Y-%m-%d %H:%M:%S"
}

log () {
  echo "[$(timestamp)] $*"
}

wait_for_slot () {
  local running
  while true; do
    running="$(jobs -rp | wc -l | tr -d ' ')"
    if [ "${running:-0}" -lt "$MAX_PARALLEL" ]; then
      return 0
    fi
    sleep 20
  done
}

next_gpu () {
  local gpu="${_FEDPLORA_GPUS[$_FEDPLORA_GPU_CURSOR]}"
  _FEDPLORA_GPU_CURSOR=$(( (_FEDPLORA_GPU_CURSOR + 1) % ${#_FEDPLORA_GPUS[@]} ))
  printf '%s\n' "$gpu"
}

check_logs_clean () {
  local label="$1"
  shift
  local rc=0
  local pattern
  shopt -s nullglob
  for pattern in "$@"; do
    local files=( $pattern )
    if [ "${#files[@]}" -eq 0 ]; then
      echo "[pipeline][$label][error] no logs matched pattern: $pattern" >&2
      rc=2
      continue
    fi
    if grep -HEn "Traceback|ModuleNotFoundError|CUDA out of memory|RuntimeError|\\[[^]]+\\]\\[error\\]" "${files[@]}"; then
      echo "[pipeline][$label][error] failure markers found in logs above." >&2
      rc=2
    fi
  done
  shopt -u nullglob
  return "$rc"
}

launch_one () {
  local stage="$1"
  local kind="$2"
  local method="$3"
  local agg="$4"
  local seed="$5"
  local split_seed="$6"
  local run_prefix="$7"
  shift 7

  wait_for_slot
  local gpu
  gpu="$(next_gpu)"
  log "[launch][$stage] gpu=$gpu kind=$kind method=$method agg=${agg:-na} seed=$seed split_seed=$split_seed"

  local args=(
    "$ONE_EXP"
    --kind "$kind"
    --method "$method"
    --seed "$seed"
    --split-seed "$split_seed"
    --run-id-prefix "$run_prefix"
    --gpu "$gpu"
  )
  if [ -n "$agg" ]; then
    args+=(--agg "$agg")
  fi
  args+=(-- "$@")

  local launch_log="$PIPELINE_ROOT/pipeline_logs/${stage}_${method}_seed${seed}.launch.log"
  nohup bash "${args[@]}" > "$launch_log" 2>&1 &
  local pid=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$pid" "$kind" "$method" "${agg:-na}" "$seed" "$split_seed" "$gpu" \
    >> "$PIPELINE_ROOT/pids/${stage}.tsv"
  echo "$pid" > "$PIPELINE_ROOT/pids/${stage}_${method}_seed${seed}.pid"
  echo "$launch_log" > "$PIPELINE_ROOT/pipeline_logs/${stage}_${method}_seed${seed}.launch_log_path.txt"
}

wait_stage () {
  local stage="$1"
  local pid kind method agg seed split_seed gpu
  local rc=0
  if [ ! -f "$PIPELINE_ROOT/pids/${stage}.tsv" ]; then
    echo "[pipeline][$stage][error] no pid manifest found." >&2
    return 2
  fi
  while IFS=$'\t' read -r pid kind method agg seed split_seed gpu; do
    log "[wait][$stage] pid=$pid method=$method gpu=$gpu"
    if ! wait "$pid"; then
      echo "[pipeline][$stage][error] pid=$pid method=$method failed." >&2
      rc=2
    fi
  done < "$PIPELINE_ROOT/pids/${stage}.tsv"
  return "$rc"
}

set_dir05_split_seed () {
  local split_seed="$1"
  local dir05_root
  dir05_root="$(dirname "$BENCHMARK_DIR_MAIN")"
  export BENCHMARK_DIR="$dir05_root/seed_$split_seed"
  check_benchmark "$BENCHMARK_DIR"
}

check_smoke_prereq () {
  if [ "${REQUIRE_SMOKE_OK:-1}" != "1" ]; then
    log "[smoke-check] skipped because REQUIRE_SMOKE_OK=${REQUIRE_SMOKE_OK:-0}"
    return 0
  fi
  local smoke_run_id="${SMOKE_RUN_ID_20260711:-v13_20260711_smoke_seed42}"
  local smoke_root="$RESULT_ROOT/$smoke_run_id"
  log "[smoke-check] checking previous smoke logs under $smoke_root"
  check_logs_clean smoke_prereq "$smoke_root/run_logs/test20260711_main_smoke_smoke_v13*.log"
  python - "$smoke_root" <<'PY'
import glob
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = ["smoke_v13a_os", "smoke_v13b_os_bonly"]
missing = []
for method in expected:
    files = list((root / "result_logs" / method).glob("*.json"))
    if not files:
        missing.append(method)
        continue
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rounds = data.get("rounds") or []
        if not rounds:
            missing.append(method)
if missing:
    print(f"[smoke-check][error] missing/empty smoke metrics: {missing}", file=sys.stderr)
    raise SystemExit(2)
print("[smoke-check] smoke metrics exist for", ",".join(expected))
PY
}

write_nx2_md5_audit () {
  local out_dir="$RESULT_ROOT/v13_20260711_nx2_md5_audit"
  mkdir -p "$out_dir"
  local baseline_dir="${BASELINE_BENCHMARK_DIR_SEED42:-$BENCHMARK_DIR_MAIN}"
  local ours_dir="${OURS_BENCHMARK_DIR_SEED42:-$BENCHMARK_DIR_MAIN}"
  python - "$baseline_dir" "$ours_dir" "$out_dir/md5_benchmark_seed42.json" <<'PY'
import hashlib
import json
import pathlib
import sys

baseline = pathlib.Path(sys.argv[1])
ours = pathlib.Path(sys.argv[2])
out = pathlib.Path(sys.argv[3])
files = ["clients.json", "train.jsonl"]

def md5(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {"baseline_dir": str(baseline), "ours_dir": str(ours), "files": {}}
ok = True
for name in files:
    bp = baseline / name
    op = ours / name
    item = {"baseline_exists": bp.is_file(), "ours_exists": op.is_file()}
    if bp.is_file():
        item["baseline_md5"] = md5(bp)
    if op.is_file():
        item["ours_md5"] = md5(op)
    item["match"] = item.get("baseline_md5") == item.get("ours_md5")
    ok = ok and bool(item["match"])
    payload["files"][name] = item
payload["all_match"] = ok
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[NX2] wrote {out} all_match={ok}")
if not ok:
    raise SystemExit(2)
PY
}

gate_nx1 () {
  export NX1_MIN_V13A_LOCAL=${NX1_MIN_V13A_LOCAL:-0.604}
  export NX1_MIN_COMPLETED_RUNS=${NX1_MIN_COMPLETED_RUNS:-4}
  python - <<'PY'
import glob
import json
import os
import pathlib
import statistics
import sys

root = pathlib.Path(os.environ["RESULT_ROOT"])
prefix = os.environ["NX1_RUN_ID_PREFIX"]
out = pathlib.Path(os.environ["PIPELINE_ROOT"]) / "gates" / "nx1_gate.json"
min_local = float(os.environ.get("NX1_MIN_V13A_LOCAL", "0.604"))
min_runs = int(os.environ.get("NX1_MIN_COMPLETED_RUNS", "4"))

rows = []
for path_s in glob.glob(str(root / f"{prefix}_seed*" / "result_logs" / "NX1_*" / "*.json")):
    path = pathlib.Path(path_s)
    method = path.parent.name
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    final = {}
    for item in reversed(data.get("rounds") or []):
        if not item.get("eval_skipped"):
            final = item
            break
    rows.append({
        "method": method,
        "path": str(path),
        "macro": final.get("domain_macro_token_accuracy"),
        "local": final.get("client_local_macro_token_accuracy"),
        "off": final.get("off_domain_macro_token_accuracy"),
    })

v13a = [r for r in rows if "v13a" in r["method"]]
missing_local = [r["method"] for r in v13a if r.get("local") is None]
v13a_local = [float(r["local"]) for r in v13a if r.get("local") is not None]
payload = {
    "prefix": prefix,
    "num_completed_runs": len(rows),
    "min_completed_runs": min_runs,
    "min_v13a_local": min_local,
    "rows": rows,
    "v13a_local_values": v13a_local,
    "v13a_local_mean": statistics.mean(v13a_local) if v13a_local else None,
}
ok = True
reasons = []
if len(rows) < min_runs:
    ok = False
    reasons.append(f"completed_runs {len(rows)} < {min_runs}")
if len(v13a) < 2:
    ok = False
    reasons.append(f"v13a_runs {len(v13a)} < 2")
if missing_local:
    ok = False
    reasons.append(f"missing local metric: {missing_local}")
if v13a_local and statistics.mean(v13a_local) < min_local:
    ok = False
    reasons.append(f"v13a_local_mean {statistics.mean(v13a_local):.6f} < {min_local:.6f}")
payload["ok"] = ok
payload["reasons"] = reasons
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not ok:
    raise SystemExit(2)
PY
}

gate_json_count () {
  local stage="$1"
  local pattern="$2"
  local min_count="$3"
  python - "$stage" "$pattern" "$min_count" "$PIPELINE_ROOT/gates/${stage}_gate.json" <<'PY'
import glob
import json
import pathlib
import sys

stage, pattern, min_count, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), pathlib.Path(sys.argv[4])
files = sorted(glob.glob(pattern))
valid = []
bad = []
for path in files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        valid.append(path)
    except Exception as e:
        bad.append({"path": path, "error": repr(e)})
payload = {
    "stage": stage,
    "pattern": pattern,
    "min_count": min_count,
    "valid_count": len(valid),
    "valid_files": valid,
    "bad_files": bad,
    "ok": len(valid) >= min_count and not bad,
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not payload["ok"]:
    raise SystemExit(2)
PY
}

log "[pipeline] root=$PIPELINE_ROOT"
log "[pipeline] GPU_LIST=${GPU_LIST:-0 1} MAX_PARALLEL=$MAX_PARALLEL"
log "[pipeline] thresholds: NX1_MIN_V13A_LOCAL=${NX1_MIN_V13A_LOCAL:-0.604}"

check_smoke_prereq

log "===== NX2: benchmark md5 audit ====="
write_nx2_md5_audit

log "===== NX1: protocol-aligned v13 runs ====="
for seed in 43 44; do
  set_dir05_split_seed "$seed"
  launch_one nx1 sft "NX1_v13a_os_split${seed}_train${seed}" fedplora_v13a_os "$seed" "$seed" "$NX1_RUN_ID_PREFIX" --force_retrain
  launch_one nx1 sft "NX1_v13b_bonly_split${seed}_train${seed}" fedplora_v13b_os_bonly "$seed" "$seed" "$NX1_RUN_ID_PREFIX" --force_retrain
done
wait_stage nx1
check_logs_clean nx1 "$RESULT_ROOT/${NX1_RUN_ID_PREFIX}_seed43/run_logs/*.log" "$RESULT_ROOT/${NX1_RUN_ID_PREFIX}_seed44/run_logs/*.log"
gate_nx1
log "[gate][nx1] passed; entering next stage."

if [ "${RUN_NX4:-1}" = "1" ]; then
  log "===== NX4: cold-start/select eval ====="
  for seed in 42 43 44; do
    set_dir05_split_seed "$seed"
    launch_one nx4 personalized_eval "X2_v13_coldstart_select_seed${seed}" "" "$seed" "$seed" "$NX4_RUN_ID_PREFIX" --v11c_mu 0.4
  done
  wait_stage nx4
  check_logs_clean nx4 "$RESULT_ROOT/${NX4_RUN_ID_PREFIX}_seed42/run_logs/*.log" "$RESULT_ROOT/${NX4_RUN_ID_PREFIX}_seed43/run_logs/*.log" "$RESULT_ROOT/${NX4_RUN_ID_PREFIX}_seed44/run_logs/*.log"
  gate_json_count nx4 "$RESULT_ROOT/${NX4_RUN_ID_PREFIX}_seed*/result_logs/X2_v13_coldstart_select_seed*.json" 3
  log "[gate][nx4] passed."
fi

if [ "${RUN_NX3:-0}" = "1" ]; then
  log "===== NX3: optional ablation after NX1 gate ====="
  set_dir05_split_seed 42
  launch_one nx3 sft NX3_v11a_alpha100_split42_train43 fedplora_v11a_relaxed_a 43 42 "$NX3_RUN_ID_PREFIX" \
    --v10_a_correction_alpha 1.0 \
    --v10_a_anchor_lambda 0.0 \
    --v10_a_prox_lambda 0.0 \
    --v10_b_prox_lambda 0.0 \
    --v10_a_norm_clip_ratio 0.0 \
    --force_retrain
  launch_one nx3 sft NX3_v11c_mu020_split42_train42 fedplora_v11c_gmix 42 42 "$NX3_RUN_ID_PREFIX" \
    --v10_a_correction_alpha 1.0 \
    --v10_a_anchor_lambda 0.0 \
    --v10_a_prox_lambda 0.0 \
    --v10_b_prox_lambda 0.0 \
    --v10_a_norm_clip_ratio 0.0 \
    --v11_global_b_mix_mu 0.2 \
    --force_retrain
  launch_one nx3 sft NX3_v11c_mu020_split42_train44 fedplora_v11c_gmix 44 42 "$NX3_RUN_ID_PREFIX" \
    --v10_a_correction_alpha 1.0 \
    --v10_a_anchor_lambda 0.0 \
    --v10_a_prox_lambda 0.0 \
    --v10_b_prox_lambda 0.0 \
    --v10_a_norm_clip_ratio 0.0 \
    --v11_global_b_mix_mu 0.2 \
    --force_retrain
  wait_stage nx3
  check_logs_clean nx3 "$RESULT_ROOT/${NX3_RUN_ID_PREFIX}_seed*/run_logs/*.log"
  gate_json_count nx3 "$RESULT_ROOT/${NX3_RUN_ID_PREFIX}_seed*/result_logs/NX3_*/*.json" 3
  log "[gate][nx3] passed."
fi

log "[pipeline][done] complete. root=$PIPELINE_ROOT"
