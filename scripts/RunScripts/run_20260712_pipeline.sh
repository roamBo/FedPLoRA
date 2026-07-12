#!/usr/bin/env bash
# 20260712 staged controller:
#   smoke is external -> fingerprint all splits -> NX0(v13a + optional v13b)
#   -> gate -> strict held-out cold-start split42 -> gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONE_EXP="$SCRIPT_DIR/run_20260712_one_experiment.sh"

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

export NX0_RUN_ID_PREFIX=${NX0_RUN_ID_PREFIX:-v13_20260712_nx0_35c_dir05_r1_finaleval}
export HELDOUT_RUN_ID_PREFIX=${HELDOUT_RUN_ID_PREFIX:-v13_20260712_strict_heldout_split42}
export RUN_NX0_V13B=${RUN_NX0_V13B:-1}

export PIPELINE_RUN_ID=${PIPELINE_RUN_ID:-v13_20260712_pipeline_$(date +%Y%m%d_%H%M%S)}
export PIPELINE_ROOT="$RESULT_ROOT/$PIPELINE_RUN_ID"
mkdir -p "$PIPELINE_ROOT/pipeline_logs" "$PIPELINE_ROOT/pids" "$PIPELINE_ROOT/gates" "$PIPELINE_ROOT/fingerprints"
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
    if grep -HEn "Traceback|ModuleNotFoundError|CUDA out of memory|RuntimeError|\\[[^]]+\\]\\[error\\]" "${files[@]}"; then
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
  local root seed split
  root="$(dir05_root)"
  for seed in 42 43 44; do
    split="$root/seed_$seed"
    check_benchmark "$split"
    python utilities/benchmark_fingerprint.py "$split" --output "$PIPELINE_ROOT/fingerprints/seed_${seed}.json"
  done
  python - "$PIPELINE_ROOT/fingerprints" "$PIPELINE_ROOT/gates/fingerprint_gate.json" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
payload = {"files": [], "ok": True, "reasons": []}
for p in sorted(root.glob("seed_*.json")):
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
  local root="$RESULT_ROOT/${SMOKE_RUN_ID_20260712:-v13_20260712_smoke_seed42}"
  check_logs_clean smoke_prereq "$root/run_logs/test20260712_main_smoke_smoke_*.log" "$root/run_logs/test20260712_main_*heldout_smoke*.log"
}

gate_nx0 () {
  export NX0_MIN_V13A_LOCAL=${NX0_MIN_V13A_LOCAL:-0.5980}
  export NX0_MAIN_LOCAL_TARGET=${NX0_MAIN_LOCAL_TARGET:-0.6071}
  python - <<'PY'
import glob, json, os, pathlib, sys
root = pathlib.Path(os.environ["RESULT_ROOT"])
prefix = os.environ["NX0_RUN_ID_PREFIX"]
out = pathlib.Path(os.environ["PIPELINE_ROOT"]) / "gates" / "nx0_gate.json"
min_local = float(os.environ.get("NX0_MIN_V13A_LOCAL", "0.5980"))
main_target = float(os.environ.get("NX0_MAIN_LOCAL_TARGET", "0.6071"))
files = sorted(glob.glob(str(root / f"{prefix}_seed42" / "result_logs" / "NX0_v13a*" / "*.json")))
payload = {"files": files, "min_v13a_local": min_local, "main_local_target": main_target}
ok = bool(files)
reasons = []
local = None
if files:
    data = json.load(open(files[-1], "r", encoding="utf-8"))
    final = {}
    for item in reversed(data.get("rounds") or []):
        if not item.get("eval_skipped"):
            final = item; break
    local = final.get("client_local_macro_token_accuracy")
    payload["v13a_local"] = local
    payload["v13a_macro"] = final.get("domain_macro_token_accuracy")
    payload["fingerprint"] = (data.get("benchmark_fingerprint") or {}).get("combined_sha256")
    if local is None:
        ok = False; reasons.append("missing v13a local metric")
    elif float(local) < min_local:
        ok = False; reasons.append(f"v13a local {float(local):.6f} < {min_local:.6f}")
else:
    reasons.append("missing NX0 v13a metrics")
if local is not None:
    payload["decision_hint"] = (
        "main_table_strong_if_3split_mean_holds" if float(local) >= main_target
        else "continue_as_medium_comm_or_capability_line"
    )
payload["ok"] = ok
payload["reasons"] = reasons
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not ok:
    raise SystemExit(2)
PY
}

gate_heldout () {
  export HELDOUT_MIN_COLDSTART_DELTA=${HELDOUT_MIN_COLDSTART_DELTA:-0.01}
  python - <<'PY'
import glob, json, os, pathlib, sys
root = pathlib.Path(os.environ["RESULT_ROOT"])
prefix = os.environ["HELDOUT_RUN_ID_PREFIX"]
out = pathlib.Path(os.environ["PIPELINE_ROOT"]) / "gates" / "heldout_gate.json"
min_delta = float(os.environ.get("HELDOUT_MIN_COLDSTART_DELTA", "0.01"))
files = sorted(glob.glob(str(root / f"{prefix}_seed42" / "result_logs" / "X2_strict_heldout_seed42.json")))
payload = {"files": files, "min_delta": min_delta}
ok = bool(files)
reasons = []
if files:
    data = json.load(open(files[-1], "r", encoding="utf-8"))
    res = data.get("results") or {}
    base_global = (res.get("global") or {}).get("macro_acc")
    deltas = {}
    for name in ["coldstart", "coldstart_geom", "v11c_coldstart", "select_without_local"]:
        val = (res.get(name) or {}).get("macro_acc")
        if val is not None and base_global is not None:
            deltas[name] = float(val) - float(base_global)
    payload["global_macro"] = base_global
    payload["deltas_vs_global"] = deltas
    best = max(deltas.values()) if deltas else None
    payload["best_delta"] = best
    payload["strict_held_out"] = data.get("strict_held_out")
    if best is None:
        ok = False; reasons.append("missing heldout deltas")
    elif best < min_delta:
        ok = False; reasons.append(f"best heldout delta {best:.6f} < {min_delta:.6f}")
else:
    reasons.append("missing heldout eval json")
payload["ok"] = ok
payload["reasons"] = reasons
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not ok:
    raise SystemExit(2)
PY
}

log "[pipeline] root=$PIPELINE_ROOT"
log "[pipeline] GPU_LIST=${GPU_LIST:-0 1} MAX_PARALLEL=$MAX_PARALLEL"
check_smoke_prereq

log "===== stage0 fingerprint ====="
fingerprint_splits

log "===== stage1 NX0 split42 ====="
launch_one nx0 sft NX0_v13a_os_split42_train42 fedplora_v13a_os 42 42 "$NX0_RUN_ID_PREFIX" --force_retrain
if [ "$RUN_NX0_V13B" = "1" ]; then
  launch_one nx0 sft NX0_v13b_bonly_split42_train42 fedplora_v13b_os_bonly 42 42 "$NX0_RUN_ID_PREFIX" --force_retrain
fi
wait_stage nx0
check_logs_clean nx0 "$RESULT_ROOT/${NX0_RUN_ID_PREFIX}_seed42/run_logs/*.log"
gate_nx0

if [ "${RUN_HELDOUT:-1}" = "1" ]; then
  log "===== stage2 strict held-out cold-start split42 ====="
  launch_one heldout personalized_eval X2_strict_heldout_seed42 "" 42 42 "$HELDOUT_RUN_ID_PREFIX" \
    --held_out_clients auto_one_per_domain \
    --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
    --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
    --few_shot_caps 5,10 \
    --held_out_route_probe_samples 10 \
    --eval_on_local \
    --cold_start \
    --v11c_mu 0.4
  wait_stage heldout
  check_logs_clean heldout "$RESULT_ROOT/${HELDOUT_RUN_ID_PREFIX}_seed42/run_logs/*.log"
  gate_heldout
fi

log "[pipeline][done] complete. root=$PIPELINE_ROOT"

