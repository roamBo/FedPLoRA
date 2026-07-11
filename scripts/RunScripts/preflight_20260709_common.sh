#!/usr/bin/env bash
# Common preflight for 20260709 FedPLoRA orders.
# This file is meant to be sourced by:
#   scripts/RunScripts/preflight_20260709_baseline.sh
#   scripts/RunScripts/preflight_20260709_main_algorithm.sh

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[usage][error] 请先进入 bash：exec bash；然后用 source 加载 preflight 脚本。" >&2
  return 2 2>/dev/null || exit 2
fi

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "[usage][error] 这个脚本需要被 source，而不是 bash 直接执行。" >&2
  echo "示例：source scripts/RunScripts/preflight_20260709_baseline.sh" >&2
  exit 2
fi

if [ -z "${FEDPLORA_PREFLIGHT_ROLE:-}" ]; then
  echo "[preflight][error] FEDPLORA_PREFLIGHT_ROLE 未设置；请 source baseline/main wrapper，不要直接 source common。" >&2
  return 2
fi

set -o pipefail

_FEDPLORA_PREFLIGHT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_FEDPLORA_DEFAULT_CODE_DIR="$(cd "$_FEDPLORA_PREFLIGHT_DIR/../.." && pwd)"

case "$FEDPLORA_PREFLIGHT_ROLE" in
  baseline)
    export FEDPLORA_PREFLIGHT_LABEL=baseline
    export RUN_ID_PREFIX=baseline_20260709_35c_dir05_r10_finaleval
    export SMOKE_RUN_ID=baseline_20260709_smoke_seed42
    _FEDPLORA_LOG_PREFIX=test20260709_baseline
    _FEDPLORA_SMOKE_LOG_PREFIX=test20260709_baseline_smoke
    _FEDPLORA_PY_COMPILE_FILES=(
      tasks/fed_train_sft.py
      utilities/utils.py
      utilities/train_eval.py
      methods/lora_expert_baselines.py
      scripts/Analysis/eval_personalized.py
      scripts/Analysis/summarize_fedplora_results.py
    )
    ;;
  main)
    export FEDPLORA_PREFLIGHT_LABEL=main
    export RUN_ID_PREFIX=v12_20260709_main_35c_dir05_r10_finaleval
    export SMOKE_RUN_ID=v12_20260709_main_smoke_seed42
    _FEDPLORA_LOG_PREFIX=test20260709_main
    _FEDPLORA_SMOKE_LOG_PREFIX=test20260709_main_smoke
    _FEDPLORA_PY_COMPILE_FILES=(
      tasks/fed_train_sft.py
      utilities/utils.py
      utilities/train_eval.py
      methods/lora_expert_baselines.py
      methods/v11/v11_common.py
      methods/v11/v11c_gmix.py
      methods/v12/__init__.py
      methods/v12/v12_common.py
      methods/v12/v12a_sched_gmix.py
      methods/v12/v12b_nmi_guard_gmix.py
      methods/v13/__init__.py
      methods/v13/v13_common.py
      methods/v13/v13a_os.py
      methods/v13/v13b_os_bonly.py
      scripts/Analysis/eval_personalized.py
      scripts/Analysis/summarize_fedplora_results.py
      scripts/DataProcessScripts/build_mixed_richness_benchmark.py
    )
    ;;
  *)
    echo "[preflight][error] unknown FEDPLORA_PREFLIGHT_ROLE=$FEDPLORA_PREFLIGHT_ROLE" >&2
    return 2
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "[preflight][error] conda 命令不可用；请先初始化 conda 或从已激活 base 的 shell 运行。" >&2
  return 2
fi

export CONDA_ENV_NAME=${CONDA_ENV_NAME:-FedRepo2}
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]; then
  if ! conda activate "$CONDA_ENV_NAME"; then
    echo "[preflight][error] conda activate $CONDA_ENV_NAME 失败。" >&2
    return 2
  fi
fi

export CODE_DIR=${CODE_DIR:-$_FEDPLORA_DEFAULT_CODE_DIR}
if [ ! -d "$CODE_DIR" ]; then
  echo "[preflight][error] CODE_DIR 不存在：$CODE_DIR" >&2
  echo "[preflight][hint] 服务器默认应为 /data2/minghao/code/FedPLoRA-main；可先 export CODE_DIR=/abs/path/FedPLoRA-main" >&2
  return 2
fi
cd "$CODE_DIR" || return 2

export MODEL_PATH=${MODEL_PATH:-/data2/minghao/model/SmolLM2-135M}
export BENCHMARK_DIR_MAIN=${BENCHMARK_DIR_MAIN:-}
export EXPECTED_NUM_CLIENTS=${EXPECTED_NUM_CLIENTS:-35}

export RESULT_ROOT=${RESULT_ROOT:-/data2/minghao/result/FedPLoRA}
export MODEL_ROOT=${MODEL_ROOT:-/data2/minghao/model/trained_models_LW}

export ROUNDS=${ROUNDS:-10}
export LOCAL_EPOCHS=${LOCAL_EPOCHS:-1}
export LR=${LR:-0.0002}
export LORA_R=${LORA_R:-8}
export LORA_ALPHA=${LORA_ALPHA:-16}
export LORA_DROPOUT=${LORA_DROPOUT:-0.05}
export BATCH_SIZE=${BATCH_SIZE:-2}
export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-256}
export TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
export TARGET_MODULES=${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
export EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-0}
export RUN_TAG_DATASET=${RUN_TAG_DATASET:-dir05}
export BENCHMARK_READY=0
export RAW_DOMAIN_JSONL=${RAW_DOMAIN_JSONL:-$CODE_DIR/data/raw/domain_7_all.jsonl}
export DIR05_OUTPUT_DIR=${DIR05_OUTPUT_DIR:-$CODE_DIR/data/domain_benchmark_35c_dir05}
export BENCHMARK_BUILD_SEEDS=${BENCHMARK_BUILD_SEEDS:-"42 43 44"}
export BENCHMARK_BUILD_ALPHAS=${BENCHMARK_BUILD_ALPHAS:-"0.5 0.1"}

benchmark_has_expected_clients () {
  local dir="$1"
  python - "$dir/clients.json" "$EXPECTED_NUM_CLIENTS" <<'PY'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(2)
clients = json.loads(path.read_text(encoding="utf-8"))
domain_counts = collections.Counter(str(row["domain"]) for row in clients)
if len(clients) != expected:
    raise SystemExit(2)
if expected == 35 and (len(domain_counts) != 7 or set(domain_counts.values()) != {5}):
    raise SystemExit(2)
PY
}

build_dirichlet_benchmarks () {
  local raw="$RAW_DOMAIN_JSONL"
  local builder="$CODE_DIR/scripts/DataProcessScripts/build_domain_benchmark_v2.py"

  if [ ! -f "$builder" ]; then
    echo "[preflight][error] missing builder: $builder" >&2
    return 2
  fi
  if [ ! -f "$raw" ]; then
    echo "[preflight][error] missing raw data: $raw" >&2
    echo "[preflight][hint] 请先同步 data/raw/domain_7_all.jsonl，或设置 RAW_DOMAIN_JSONL=/abs/path/domain_7_all.jsonl" >&2
    return 2
  fi

  local seed alpha suffix out split
  for seed in $BENCHMARK_BUILD_SEEDS; do
    for alpha in $BENCHMARK_BUILD_ALPHAS; do
      suffix="${alpha/./}"
      if [ "$suffix" = "05" ]; then
        out="$DIR05_OUTPUT_DIR"
      else
        out="$CODE_DIR/data/domain_benchmark_35c_dir${suffix}"
      fi
      split="$out/seed_${seed}"

      if benchmark_has_expected_clients "$split" >/dev/null 2>&1; then
        echo "[preflight] benchmark exists and ok: $split"
        continue
      fi

      if [ -f "$split/clients.json" ]; then
        echo "[preflight][warn] benchmark exists but is not 35 clients; rebuilding: $split" >&2
      else
        echo "[preflight] building benchmark -> $split"
      fi

      python "$builder" \
        --input_jsonl "$raw" \
        --output_dir "$out" \
        --num_clients_per_domain 5 \
        --seed "$seed" \
        --partition dirichlet \
        --dirichlet_alpha "$alpha" \
        --subtopic kmeans \
        --n_subtopics 10

      if ! benchmark_has_expected_clients "$split" >/dev/null 2>&1; then
        echo "[preflight][error] built split is still not 35 clients: $split" >&2
        return 2
      fi
    done
  done

  export BENCHMARK_DIR_MAIN="$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42"
  export BENCHMARK_DIR="$BENCHMARK_DIR_MAIN"
  echo "[preflight] built/resolved BENCHMARK_DIR_MAIN=$BENCHMARK_DIR_MAIN"
  return 0
}

resolve_benchmark_dir () {
  if [ -n "${BENCHMARK_DIR_MAIN:-}" ] && [ -f "$BENCHMARK_DIR_MAIN/clients.json" ]; then
    if benchmark_has_expected_clients "$BENCHMARK_DIR_MAIN" >/dev/null 2>&1; then
      export BENCHMARK_DIR="$BENCHMARK_DIR_MAIN"
      echo "[preflight] use explicit BENCHMARK_DIR_MAIN=$BENCHMARK_DIR_MAIN"
      return 0
    fi
    echo "[preflight][warn] explicit BENCHMARK_DIR_MAIN exists but is not 35 clients: $BENCHMARK_DIR_MAIN" >&2
  fi

  local dir
  for dir in \
    "$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42" \
    "$CODE_DIR/../data/domain_benchmark_35c_dir05/seed_42" \
    "/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_35c_dir05/seed_42" \
    "/home/minghao/code/FedPLoRA-main/data/domain_benchmark_35c_dir05/seed_42" \
    "/data2/minghao/data/domain_benchmark_35c_dir05/seed_42" \
    "/data/yaominghao/gb/FedPLoRA/data/domain_benchmark_35c_dir05/seed_42"
  do
    if [ -f "$dir/clients.json" ]; then
      if benchmark_has_expected_clients "$dir" >/dev/null 2>&1; then
        export BENCHMARK_DIR_MAIN="$dir"
        export BENCHMARK_DIR="$dir"
        echo "[preflight] auto-resolved BENCHMARK_DIR_MAIN=$BENCHMARK_DIR_MAIN"
        return 0
      fi
      echo "[preflight][warn] found invalid 35c_dir05 split; will rebuild: $dir" >&2
    fi
  done

  echo "[preflight][warn] cannot find 35c_dir05 benchmark clients.json." >&2
  if [ -f "$CODE_DIR/data/domain_benchmark_35c/seed_42/clients.json" ]; then
    echo "[preflight][hint] found old split: $CODE_DIR/data/domain_benchmark_35c/seed_42" >&2
    echo "[preflight][hint] 但本轮 order 默认要求 domain_benchmark_35c_dir05；不要无确认混用旧 split。" >&2
  fi

  echo "[preflight] dir05 不存在或不是 35 clients；按历史逻辑自动构建 dir05/dir01 × seed42/43/44。" >&2
  if build_dirichlet_benchmarks; then
    return 0
  fi

  echo "[preflight][hint] 如果 benchmark 在别处，请先执行：" >&2
  echo "  export BENCHMARK_DIR_MAIN=/abs/path/to/domain_benchmark_35c_dir05/seed_42" >&2
  echo "[preflight][hint] 也可以在服务器上查找：" >&2
  echo "  find /data2/minghao/code /home/minghao/code /data /data2 -path '*/domain_benchmark_35c_dir05/seed_42/clients.json' -print 2>/dev/null" >&2
  return 2
}

if resolve_benchmark_dir; then
  export BENCHMARK_READY=1
else
  export BENCHMARK_READY=0
  echo "[preflight][stop] benchmark 未解析成功。请先设置 BENCHMARK_DIR_MAIN 后重新 source 本脚本；暂时不要运行后续实验段。" >&2
fi

check_benchmark () {
  local dir="$1"
  if [ "${BENCHMARK_READY:-0}" != "1" ]; then
    echo "[preflight][error] BENCHMARK_READY=0；请先修正 BENCHMARK_DIR_MAIN 并重新 source preflight 脚本。" >&2
    return 2
  fi
  python - "$dir/clients.json" "$EXPECTED_NUM_CLIENTS" <<'PY'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"[preflight][error] missing clients.json: {path}")
clients = json.loads(path.read_text(encoding="utf-8"))
domain_counts = collections.Counter(str(row["domain"]) for row in clients)
if len(clients) != expected:
    raise SystemExit(f"[preflight][error] expected {expected} clients, found {len(clients)} in {path}")
if expected == 35 and (len(domain_counts) != 7 or set(domain_counts.values()) != {5}):
    raise SystemExit(f"[preflight][error] expected 7 domains x 5 clients, found {dict(domain_counts)}")
print(f"[preflight] benchmark_ok path={path.parent} clients={len(clients)} domains={dict(sorted(domain_counts.items()))}")
PY
}

check_required_imports () {
  python - <<'PY'
import importlib.util

required = [
    "methods.v8",
    "methods.v9",
    "methods.v10",
    "methods.v11",
    "methods.v12",
    "methods.v13",
    "methods.lora_expert_baselines",
    "utilities.utils",
    "utilities.train_eval",
]

missing = []
for name in required:
    if importlib.util.find_spec(name) is None:
        missing.append(name)

if missing:
    print("[preflight][error] required Python modules/packages are missing:")
    for name in missing:
        print(f"  - {name}")
    print("[preflight][hint] 服务器代码同步不完整；请同步 methods/v10、v11、v12、v13 以及相关 tasks/utilities/scripts 后再 source preflight。")
    raise SystemExit(2)

print("[preflight] module_path_ok modules=v8,v9,v10,v11,v12,v13,lora_expert,utils,train_eval")
PY
}

ensure_preflight_ready () {
  if [ "${BENCHMARK_READY:-0}" != "1" ] || [ -z "${BENCHMARK_DIR:-}" ] || [ ! -f "$BENCHMARK_DIR/clients.json" ]; then
    echo "[guard][$FEDPLORA_PREFLIGHT_LABEL][error] benchmark 未就绪。" >&2
    echo "[guard][$FEDPLORA_PREFLIGHT_LABEL][hint] BENCHMARK_READY=${BENCHMARK_READY:-unset}" >&2
    echo "[guard][$FEDPLORA_PREFLIGHT_LABEL][hint] BENCHMARK_DIR=${BENCHMARK_DIR:-unset}" >&2
    echo "[guard][$FEDPLORA_PREFLIGHT_LABEL][hint] BENCHMARK_DIR_MAIN=${BENCHMARK_DIR_MAIN:-unset}" >&2
    echo "[guard][$FEDPLORA_PREFLIGHT_LABEL][hint] 当前 BENCHMARK_DIR 下必须存在 clients.json；dir05 实验请先 source preflight，FlowerTune 实验请先设置 BENCHMARK_DIR_FLOWERTUNE。" >&2
    return 2
  fi
  if [ ! -f "$CODE_DIR/tasks/fed_train_sft.py" ]; then
    echo "[guard][$FEDPLORA_PREFLIGHT_LABEL][error] CODE_DIR 不正确或代码未同步：$CODE_DIR/tasks/fed_train_sft.py 不存在。" >&2
    return 2
  fi
}

assert_role_run () {
  local method="$1"
  local agg="$2"

  if [ "$FEDPLORA_PREFLIGHT_ROLE" = "baseline" ]; then
    case "$agg" in
      normal|ecolora|fedalt|flexlora|flora|ffa|fedsa_lora|hydralora|hilora|fedlease|yoco|feddat|fedplora_oneshot|fedplora_v9_mix_ab)
        ;;
      *)
        echo "[guard][baseline][error] agg=$agg 不属于 baseline 白名单；主算法请 source preflight_20260709_main_algorithm.sh。" >&2
        return 2
        ;;
    esac
    case "$method" in
      smoke_*|OS1_*|X3_*|P2_*|M3_mixrich_*|M3_os_mixrich_*|N7_baseline_*|N9_flower_*)
        ;;
      *)
        echo "[guard][baseline][error] method=$method 命名不像 baseline 任务；为防误跑已拒绝。" >&2
        return 2
        ;;
    esac
  else
    case "$agg" in
      fedplora_v8|fedplora_v11c_gmix|fedplora_v11a_relaxed_a|fedplora_v12a_sched_gmix|fedplora_v12b_nmi_guard_gmix|fedplora_v13a_os|fedplora_v13b_os_bonly|fedplora_os|fedplora_os_bonly)
        ;;
      *)
        echo "[guard][main][error] agg=$agg 不属于主算法白名单；baseline 请 source preflight_20260709_baseline.sh。" >&2
        return 2
        ;;
    esac
    case "$method" in
      smoke_v8*|smoke_v11*|smoke_v12*|smoke_v13*|X1*|X2*|NX1*|NX2*|NX3*|NX4*|NX5*|NX6*|OS1_v8*|OS1_v11*|OS1_v13*|X3_v11*|X3_v12*|M3_mixrich_v8*|M3_mixrich_v11*|M3_mixrich_v12*|M3_os_mixrich_v8*|M3_os_mixrich_v11*|M3_os_mixrich_v13*|N7_ours_*)
        ;;
      *)
        echo "[guard][main][error] method=$method 命名不像主算法任务；为防误跑已拒绝。" >&2
        return 2
        ;;
    esac
  fi
}

set_run_paths () {
  ensure_preflight_ready || return $?
  export SEED="$1"
  export RUN_ID="${RUN_ID_PREFIX}_seed${SEED}"
  export RUN_ROOT="$RESULT_ROOT/$RUN_ID"
  export TRAINED_MODELS_ROOT="$MODEL_ROOT/$RUN_ID"
  export RUN_TAG=SmolLM2-135M_${RUN_TAG_DATASET}_r${ROUNDS}_e${LOCAL_EPOCHS}_lr${LR}
  mkdir -p "$RUN_ROOT/run_logs" "$RUN_ROOT/result_logs" "$RUN_ROOT/result_files/client_states" "$TRAINED_MODELS_ROOT"
  printf '[run][%s] SEED=%s\n[run][%s] BENCHMARK_DIR=%s\n[run][%s] RUN_ID=%s\n[run][%s] RUN_ROOT=%s\n' \
    "$FEDPLORA_PREFLIGHT_LABEL" "$SEED" \
    "$FEDPLORA_PREFLIGHT_LABEL" "$BENCHMARK_DIR" \
    "$FEDPLORA_PREFLIGHT_LABEL" "$RUN_ID" \
    "$FEDPLORA_PREFLIGHT_LABEL" "$RUN_ROOT"
}

run_sft_full () {
  local method="$1"
  local agg="$2"
  shift 2
  assert_role_run "$method" "$agg" || return $?
  ensure_preflight_ready || return $?
  local gpu="${GPU:-0}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u tasks/fed_train_sft.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --num_clients "$EXPECTED_NUM_CLIENTS" \
    --agg_type "$agg" \
    --rounds "$ROUNDS" --local_epochs "$LOCAL_EPOCHS" --lr "$LR" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
    --client_state_dir "$RUN_ROOT/result_files/client_states/$method" \
    --metrics_output_dir "$RUN_ROOT/result_logs/$method" \
    --save_run_checkpoint_dir "$TRAINED_MODELS_ROOT/${method}_${RUN_TAG}_seed${SEED}" \
    --trained_models_root "$TRAINED_MODELS_ROOT" \
    --eval_max_batches "$EVAL_MAX_BATCHES" --seed "$SEED" \
    --save_client_state_to_disk \
    --gradient_checkpointing \
    --eval_personalization_metrics \
    --eval_final_only \
    --skip_post_agg_snapshots \
    "$@" \
    > "$RUN_ROOT/run_logs/${_FEDPLORA_LOG_PREFIX}_${method}_${RUN_TAG}_seed${SEED}.log" 2>&1 &
}

export SMOKE_ROOT="$RESULT_ROOT/$SMOKE_RUN_ID"
export SMOKE_TRAINED_MODELS_ROOT="$MODEL_ROOT/$SMOKE_RUN_ID"
mkdir -p "$SMOKE_ROOT/run_logs" "$SMOKE_ROOT/result_logs" "$SMOKE_ROOT/result_files/client_states" "$SMOKE_TRAINED_MODELS_ROOT"

run_sft_smoke () {
  local method="$1"
  local agg="$2"
  shift 2
  assert_role_run "$method" "$agg" || return $?
  ensure_preflight_ready || return $?
  local gpu="${GPU:-0}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u tasks/fed_train_sft.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$BENCHMARK_DIR_MAIN" \
    --num_clients 35 \
    --agg_type "$agg" \
    --rounds 1 --local_epochs 1 --lr "$LR" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
    --client_state_dir "$SMOKE_ROOT/result_files/client_states/$method" \
    --metrics_output_dir "$SMOKE_ROOT/result_logs/$method" \
    --save_run_checkpoint_dir "$SMOKE_TRAINED_MODELS_ROOT/${method}_smoke_seed42" \
    --trained_models_root "$SMOKE_TRAINED_MODELS_ROOT" \
    --eval_max_batches 1 --seed 42 \
    --train_max_steps_per_client 1 \
    --max_train_samples_per_client 10 \
    --save_client_state_to_disk \
    --gradient_checkpointing \
    --eval_personalization_metrics \
    --eval_final_only \
    --skip_post_agg_snapshots \
    "$@" \
    > "$SMOKE_ROOT/run_logs/${_FEDPLORA_SMOKE_LOG_PREFIX}_${method}_seed42.log" 2>&1 &
}

if [ "$FEDPLORA_PREFLIGHT_ROLE" = "main" ]; then
  run_personalized_eval () {
    local name="$1"
    shift
    case "$name" in
      X2_*|N2_*) ;;
      *)
        echo "[guard][main][error] personalized eval 名称必须以 X2_ 或 N2_ 开头，当前 name=$name" >&2
        return 2
        ;;
    esac
    ensure_preflight_ready || return $?
    local gpu="${GPU:-0}"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -u scripts/Analysis/eval_personalized.py \
      --model "$MODEL_PATH" \
      --benchmark_dir "$BENCHMARK_DIR" \
      --target_modules "$TARGET_MODULES" \
      --torch_dtype "$TORCH_DTYPE" \
      --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
      --lr "$LR" --local_epochs "$LOCAL_EPOCHS" \
      --eval_max_batches "$EVAL_MAX_BATCHES" \
      --seed "$SEED" \
      --schemes local,global,v7,v11c,select \
      --select_candidates local,v7,global,v11c \
      --cold_start --eval_on_local \
      "$@" \
      --out "$RUN_ROOT/result_logs/${name}_seed${SEED}.json" \
      > "$RUN_ROOT/run_logs/${_FEDPLORA_LOG_PREFIX}_${name}_seed${SEED}.log" 2>&1 &
  }
fi

if python -m py_compile "${_FEDPLORA_PY_COMPILE_FILES[@]}" \
  && check_required_imports \
  && check_benchmark "$BENCHMARK_DIR_MAIN"; then
  set_run_paths 42
  echo "[preflight][ok] $FEDPLORA_PREFLIGHT_LABEL preflight loaded."
  echo "[preflight][ok] 后续可直接运行 order 中的 smoke / 正式实验段。"
else
  echo "[preflight][stop] 代码或 benchmark 前置检查失败；请修正后重新 source 本脚本，再跑实验段。" >&2
fi
