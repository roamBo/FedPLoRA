#!/usr/bin/env bash
# 20260713 staged controller:
#   smoke is external -> fingerprint all splits -> strict held-out split43/44
#   in parallel -> gate -> held-out offset1 split42 -> manifest/reliability.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONE_EXP="$SCRIPT_DIR/run_20260713_one_experiment.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "[pipeline][error] conda not found." >&2
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
export LR=${LR:-0.0002}

export HELDOUT_RUN_ID_PREFIX=${HELDOUT_RUN_ID_PREFIX:-v13_20260713_strict_heldout}
export OFFSET_RUN_ID_PREFIX=${OFFSET_RUN_ID_PREFIX:-v13_20260713_strict_heldout_offset1}
export PIPELINE_RUN_ID=${PIPELINE_RUN_ID:-v13_20260713_pipeline_$(date +%Y%m%d_%H%M%S)}
export PIPELINE_ROOT="$RESULT_ROOT/$PIPELINE_RUN_ID"
mkdir -p "$PIPELINE_ROOT/pipeline_logs" "$PIPELINE_ROOT/pids" "$PIPELINE_ROOT/gates" "$PIPELINE_ROOT/fingerprints" "$PIPELINE_ROOT/analysis"
echo "$$" > "$PIPELINE_ROOT/pids/pipeline.pid"

read -r -a _FEDPLORA_GPUS <<< "${GPU_LIST:-0 1}"
if [ "${#_FEDPLORA_GPUS[@]}" -lt 1 ]; then
  _FEDPLORA_GPUS=(0)
fi
export MAX_PARALLEL=${MAX_PARALLEL:-${#_FEDPLORA_GPUS[@]}}
_FEDPLORA_GPU_CURSOR=0

timestamp () { date "+%Y-%m-%d %H:%M:%S"; }
log () { echo "[$(timestamp)] $*"; }

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
  local label="$1"; shift
  local rc=0 pattern
  shopt -s nullglob
  for pattern in "$@"; do
    local files=( $pattern )
    if [ "${#files[@]}" -eq 0 ]; then
      echo "[pipeline][$label][error] no logs matched: $pattern" >&2
      rc=2
      continue
    fi
    if grep -HEn "Traceback|ModuleNotFoundError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|RuntimeError|\\[[^]]+\\]\\[error\\]" "${files[@]}"; then
      echo "[pipeline][$label][error] failure markers found." >&2
      rc=2
    fi
  done
  shopt -u nullglob
  return "$rc"
}

launch_one () {
  local stage="$1" kind="$2" method="$3" agg="$4" seed="$5" split_seed="$6" run_prefix="$7"
  shift 7
  wait_for_slot
  local gpu launch_log pid
  gpu="$(next_gpu)"
  launch_log="$PIPELINE_ROOT/pipeline_logs/${stage}_${method}_seed${seed}.launch.log"
  log "[launch][$stage] gpu=$gpu kind=$kind method=$method agg=${agg:-na} seed=$seed split=$split_seed"
  local args=("$ONE_EXP" --kind "$kind" --method "$method" --seed "$seed" --split-seed "$split_seed" --run-id-prefix "$run_prefix" --gpu "$gpu")
  if [ -n "$agg" ]; then
    args+=(--agg "$agg")
  fi
  args+=(-- "$@")
  nohup bash "${args[@]}" > "$launch_log" 2>&1 &
  pid=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$pid" "$kind" "$method" "${agg:-na}" "$seed" "$split_seed" "$gpu" >> "$PIPELINE_ROOT/pids/${stage}.tsv"
  echo "$pid" > "$PIPELINE_ROOT/pids/${stage}_${method}_seed${seed}.pid"
}

wait_stage () {
  local stage="$1" pid kind method agg seed split_seed gpu rc=0
  if [ ! -f "$PIPELINE_ROOT/pids/${stage}.tsv" ]; then
    echo "[pipeline][$stage][error] missing pid manifest." >&2
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

dir05_root () { dirname "$BENCHMARK_DIR_MAIN"; }

fingerprint_splits () {
  local root seed split out
  root="$(dir05_root)"
  for seed in 42 43 44; do
    split="$root/seed_$seed"
    out="$PIPELINE_ROOT/fingerprints/seed_${seed}.json"
    check_benchmark "$split"
    python utilities/benchmark_fingerprint.py "$split" --output "$out"
  done
  python - "$PIPELINE_ROOT/fingerprints" "$PIPELINE_ROOT/gates/fingerprint_gate.json" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
payload = {"files": [], "ok": True, "reasons": []}
files = sorted(root.glob("seed_*.json"))
if len(files) != 3:
    payload["ok"] = False
    payload["reasons"].append(f"expected 3 fingerprint files, found {len(files)}")
for p in files:
    if "${" in p.name:
        payload["ok"] = False
        payload["reasons"].append(f"literal variable in filename: {p.name}")
    data = json.loads(p.read_text(encoding="utf-8"))
    item = {
        "file": str(p),
        "split_dir": data.get("split_dir"),
        "sha": data.get("combined_sha256"),
        "clients": (data.get("clients") or {}).get("num_clients"),
        "train_rows": (data.get("split_counts") or {}).get("train"),
    }
    if item["clients"] != 35:
        payload["ok"] = False
        payload["reasons"].append(f"{p.name}: clients={item['clients']}")
    if item["train_rows"] != 9012:
        payload["reasons"].append(f"{p.name}: train_rows={item['train_rows']}")
    payload["files"].append(item)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not payload["ok"]:
    raise SystemExit(2)
PY
}

check_smoke_prereq () {
  if [ "${REQUIRE_SMOKE_OK:-1}" != "1" ]; then
    log "[smoke-check] skipped"
    return 0
  fi
  local root="$RESULT_ROOT/${SMOKE_RUN_ID_20260713:-v13_20260713_smoke_seed42}"
  check_logs_clean smoke_prereq "$root/run_logs/test20260713_main_*heldout_smoke*.log"
}

gate_heldout_main () {
  export HELDOUT_MIN_COLDSTART_DELTA=${HELDOUT_MIN_COLDSTART_DELTA:-0.005}
  export HELDOUT_MIN_PASS_COUNT=${HELDOUT_MIN_PASS_COUNT:-1}
  python - <<'PY'
import glob, json, os, pathlib, sys
root = pathlib.Path(os.environ["RESULT_ROOT"])
prefix = os.environ["HELDOUT_RUN_ID_PREFIX"]
out = pathlib.Path(os.environ["PIPELINE_ROOT"]) / "gates" / "heldout_main_gate.json"
min_delta = float(os.environ.get("HELDOUT_MIN_COLDSTART_DELTA", "0.005"))
min_pass = int(os.environ.get("HELDOUT_MIN_PASS_COUNT", "1"))
payload = {"min_delta": min_delta, "min_pass_count": min_pass, "splits": {}, "ok": True, "reasons": []}
pass_count = 0
for seed in (43, 44):
    files = sorted(glob.glob(str(root / f"{prefix}_seed{seed}" / "result_logs" / f"X2_strict_heldout_seed{seed}_seed{seed}.json")))
    item = {"files": files}
    if not files:
        item["ok"] = False
        payload["ok"] = False
        payload["reasons"].append(f"missing heldout seed{seed}")
        payload["splits"][str(seed)] = item
        continue
    data = json.load(open(files[-1], "r", encoding="utf-8"))
    res = data.get("results") or {}
    global_macro = (res.get("global") or {}).get("macro_acc")
    deltas = {}
    for name in ["coldstart", "coldstart_geom", "v11c_coldstart", "select_without_local"]:
        val = (res.get(name) or {}).get("macro_acc")
        if val is not None and global_macro is not None:
            deltas[name] = float(val) - float(global_macro)
    best = max(deltas.values()) if deltas else None
    item.update({
        "global_macro": global_macro,
        "deltas_vs_global": deltas,
        "best_delta": best,
        "strict_held_out": data.get("strict_held_out"),
        "ok": bool(best is not None and best >= min_delta),
    })
    if item["ok"]:
        pass_count += 1
    payload["splits"][str(seed)] = item
payload["pass_count"] = pass_count
if pass_count < min_pass:
    payload["ok"] = False
    payload["reasons"].append(f"pass_count {pass_count} < {min_pass}")
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not payload["ok"]:
    raise SystemExit(2)
PY
}

gate_offset () {
  export OFFSET_MIN_COLDSTART_DELTA=${OFFSET_MIN_COLDSTART_DELTA:-0.0}
  python - <<'PY'
import glob, json, os, pathlib
root = pathlib.Path(os.environ["RESULT_ROOT"])
prefix = os.environ["OFFSET_RUN_ID_PREFIX"]
out = pathlib.Path(os.environ["PIPELINE_ROOT"]) / "gates" / "heldout_offset_gate.json"
min_delta = float(os.environ.get("OFFSET_MIN_COLDSTART_DELTA", "0.0"))
files = sorted(glob.glob(str(root / f"{prefix}_seed42" / "result_logs" / "X2_strict_heldout_offset1_seed42_seed42.json")))
payload = {"files": files, "min_delta": min_delta, "ok": bool(files), "reasons": []}
if files:
    data = json.load(open(files[-1], "r", encoding="utf-8"))
    res = data.get("results") or {}
    global_macro = (res.get("global") or {}).get("macro_acc")
    deltas = {}
    for name in ["coldstart", "coldstart_geom", "select_without_local"]:
        val = (res.get(name) or {}).get("macro_acc")
        if val is not None and global_macro is not None:
            deltas[name] = float(val) - float(global_macro)
    best = max(deltas.values()) if deltas else None
    payload.update({"global_macro": global_macro, "deltas_vs_global": deltas, "best_delta": best, "strict_held_out": data.get("strict_held_out")})
    if best is None or best < min_delta:
        payload["ok"] = False
        payload["reasons"].append(f"best offset delta {best} < {min_delta}")
else:
    payload["reasons"].append("missing offset heldout json")
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not payload["ok"]:
    raise SystemExit(2)
PY
}

analysis_zero_gpu () {
  local scan_root="${ANALYSIS_SCAN_ROOT:-$RESULT_ROOT}"
  python scripts/Analysis/build_analysis_ready_manifest.py "$scan_root" \
    --output "$PIPELINE_ROOT/analysis/analysis_ready_manifest.json" || true
  python scripts/Analysis/analyze_router_reliability.py "$scan_root" \
    --include "${ROUTER_INCLUDE_REGEX:-v13_2026071[123]|NX[01]_v13}" \
    --output_json "$PIPELINE_ROOT/analysis/router_reliability.json" \
    --output_md "$PIPELINE_ROOT/analysis/router_reliability.md" || true
}

log "[pipeline] root=$PIPELINE_ROOT"
log "[pipeline] GPU_LIST=${GPU_LIST:-0 1} MAX_PARALLEL=$MAX_PARALLEL"
check_smoke_prereq

log "===== stage0 fingerprint ====="
fingerprint_splits

log "===== stage1 strict held-out splits43/44 ====="
launch_one heldout personalized_eval X2_strict_heldout_seed43 "" 43 43 "$HELDOUT_RUN_ID_PREFIX" \
  --held_out_clients auto_one_per_domain \
  --held_out_policy first \
  --held_out_offset 0 \
  --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
  --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
  --few_shot_caps 5,10 \
  --held_out_route_probe_samples 10 \
  --eval_on_local \
  --cold_start \
  --v11c_mu 0.4
launch_one heldout personalized_eval X2_strict_heldout_seed44 "" 44 44 "$HELDOUT_RUN_ID_PREFIX" \
  --held_out_clients auto_one_per_domain \
  --held_out_policy first \
  --held_out_offset 0 \
  --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
  --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
  --few_shot_caps 5,10 \
  --held_out_route_probe_samples 10 \
  --eval_on_local \
  --cold_start \
  --v11c_mu 0.4
wait_stage heldout
check_logs_clean heldout "$RESULT_ROOT/${HELDOUT_RUN_ID_PREFIX}_seed43/run_logs/*.log" "$RESULT_ROOT/${HELDOUT_RUN_ID_PREFIX}_seed44/run_logs/*.log"
gate_heldout_main

if [ "${RUN_OFFSET_HELDOUT:-1}" = "1" ]; then
  log "===== stage2 strict held-out offset1 split42 ====="
  launch_one offset personalized_eval X2_strict_heldout_offset1_seed42 "" 42 42 "$OFFSET_RUN_ID_PREFIX" \
    --held_out_clients auto_one_per_domain \
    --held_out_policy offset \
    --held_out_offset 1 \
    --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
    --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
    --few_shot_caps 5,10 \
    --held_out_route_probe_samples 10 \
    --eval_on_local \
    --cold_start \
    --v11c_mu 0.4
  wait_stage offset
  check_logs_clean offset "$RESULT_ROOT/${OFFSET_RUN_ID_PREFIX}_seed42/run_logs/*.log"
  gate_offset
fi

log "===== stage3 zero-GPU analysis manifest/reliability ====="
analysis_zero_gpu

log "[pipeline][done] complete. root=$PIPELINE_ROOT"
