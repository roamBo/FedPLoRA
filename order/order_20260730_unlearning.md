# Unlearning Dividend Phase-0 实验命令（20260730）

######### FedPLoRA v14 Unlearning Dividend Protocol 命令-20260730 #########

【命令介绍】

本文件把 `claude/idea_unlearning_dividend_protocol_20260730.md` 中的 Phase 0 证伪实验落成可执行命令。核心流程分三步：

1. 从已有正式 checkpoint 读取 shared A 与各 client LoRA-B，生成 A1–A6/A7 synthetic eval-only checkpoints。
2. 用后台队列脚本逐个评测 synthetic checkpoint；父脚本一个 nohup 日志，每个子实验一个 pid 与一个运行日志。
3. 汇总 eval JSON，自动判断 `proj_auto` 是否满足 GO/NO-GO 条件。

【命令目的】

验证“规范不变的 LoRA-B 子空间投影移除某域方向”是否能打破 naive `pool_all` 平均：即对每个 forget domain，比较 `proj_auto` 在其余域上的表现是否超过 `pool_all`、Task Arithmetic 取负和 random projection，并确认被移除域本身下降。

【命令设置】

```text
主数据: D1 / domain_benchmark_35c_dir05 / seed_42
模型: SmolLM2-135M
源 checkpoint: 已跑完的 fedplora_v13a_os / rounds=1 / seed=42
生成 arms:
  A1 base
  A2 pool_all
  A3 pool_loo
  A4 proj_auto, energy_tau=0.90
  A5 task_arith lambda=0.5,1.0
  A6 random_proj_auto trial=0
  A7 routed_domain（上界参照，不进 GO 条件）
权重: sample-weighted，读取 clients.json 的 n_train
评测: 先 smoke，再 D1 seed42 screening；screening 通过后再补 full eval / 多 seed / FT
```

【实验产物位置说明】

```text
run_logs:
/data2/minghao/result/FedPLoRA/unlearning_20260730/run_logs/

synthetic checkpoints:
/data2/minghao/result/FedPLoRA/unlearning_20260730/result_files/unlearning_phase0/

eval result_logs:
/data2/minghao/result/FedPLoRA/unlearning_20260730/result_logs/unlearning_phase0/

summary:
/data2/minghao/result/FedPLoRA/unlearning_20260730/result_files/unlearning_phase0_summary/

pids/status:
/data2/minghao/result/FedPLoRA/unlearning_20260730/pids/
/data2/minghao/result/FedPLoRA/unlearning_20260730/status/
```

【实验运行涉及场景】

```text
D1 35-client / 7-domain / Dirichlet alpha=0.5 / one-shot FedPLoRA v13a checkpoint 后处理
服务器侧 unlearning / no retraining / eval-only
```

【实验前置命令】

```bash
conda activate FedRepo2

export CODE_DIR=${CODE_DIR:-/data2/minghao/code/FedPLoRA-main}
cd "$CODE_DIR"

export MODEL_PATH=${MODEL_PATH:-/data2/minghao/model/SmolLM2-135M}
export BENCHMARK_DIR_D1_42=${BENCHMARK_DIR_D1_42:-$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42}
export RESULT_ROOT=${RESULT_ROOT:-/data2/minghao/result/FedPLoRA/unlearning_20260730}
export CKPT_SEARCH_ROOTS=${CKPT_SEARCH_ROOTS:-"/data2/minghao/model/trained_models_LW /data2/minghao/result/FedPLoRA /data/yaominghao/gb/models/trained_models_LW"}

mkdir -p "$RESULT_ROOT/run_logs" "$RESULT_ROOT/result_logs" "$RESULT_ROOT/result_files" "$RESULT_ROOT/pids" "$RESULT_ROOT/status"

python -m py_compile \
  methods/v14/__init__.py \
  methods/v14/unlearning_dividend.py \
  scripts/Analysis/build_unlearning_dividend_phase0.py \
  scripts/Analysis/summarize_unlearning_dividend_eval.py \
  tasks/fed_train_sft.py

bash -n scripts/RunScripts/launch_unlearning_phase0_eval_queue.sh

python - "$BENCHMARK_DIR_D1_42/clients.json" <<'PY'
import collections, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[preflight][error] missing clients.json: {path}")
rows = json.loads(path.read_text(encoding="utf-8"))
domains = collections.Counter(str(row["domain"]) for row in rows)
if len(rows) != 35 or set(domains.values()) != {5}:
    raise SystemExit(f"[preflight][error] expected 35 clients = 7 domains x 5, got n={len(rows)} domains={dict(domains)}")
print(f"[preflight][ok] D1 clients={len(rows)} domains={dict(sorted(domains.items()))}")
PY

export SOURCE_CKPT_D1_42="$(python scripts/Analysis/checkpoint_manifest.py \
  --roots $CKPT_SEARCH_ROOTS \
  --resolve \
  --agg_type fedplora_v13a_os \
  --seed 42 \
  --model_contains SmolLM2-135M \
  --benchmark_contains "domain_benchmark_35c_dir05/seed_42" \
  --bundle_contains "r1")"

test -f "$SOURCE_CKPT_D1_42/run_checkpoint_meta.json"
test -f "$SOURCE_CKPT_D1_42/global_shared.pt"
test -d "$SOURCE_CKPT_D1_42/clients"
echo "[preflight][ok] SOURCE_CKPT_D1_42=$SOURCE_CKPT_D1_42"
```

如果 checkpoint 解析到多个候选，先运行下面这条看清楚再手动指定：

```bash
python scripts/Analysis/checkpoint_manifest.py \
  --roots $CKPT_SEARCH_ROOTS \
  --resolve --list_matches \
  --agg_type fedplora_v13a_os \
  --seed 42 \
  --model_contains SmolLM2-135M \
  --benchmark_contains "domain_benchmark_35c_dir05/seed_42"
```

【实验运行命令】

## 1. smoke：只 forget `code`，先检查生成与 eval-only 链路

### 1.1 CPU 生成 synthetic checkpoints（不占 GPU）

```bash
export SYNTH_SMOKE="$RESULT_ROOT/result_files/unlearning_phase0/d1_seed42_smoke_code"

nohup /usr/bin/time -v python -u scripts/Analysis/build_unlearning_dividend_phase0.py \
  --checkpoint_dir "$SOURCE_CKPT_D1_42" \
  --clients_json "$BENCHMARK_DIR_D1_42/clients.json" \
  --benchmark_dir "$BENCHMARK_DIR_D1_42" \
  --model "$MODEL_PATH" \
  --output_dir "$SYNTH_SMOKE" \
  --forget_domains code \
  --projection_ranks auto \
  --task_arith_lambdas 0.5,1.0 \
  --random_trials 1 \
  --expected_clients 35 \
  --seed 42 \
  --force \
  > "$RESULT_ROOT/run_logs/test20260730_unlearning_build_smoke_d1_seed42.log" 2>&1 &
```

查看生成是否完成：

```bash
tail -n 40 "$RESULT_ROOT/run_logs/test20260730_unlearning_build_smoke_d1_seed42.log"
python -m json.tool "$SYNTH_SMOKE/phase0_manifest.json" >/dev/null
find "$SYNTH_SMOKE/checkpoints" -mindepth 1 -maxdepth 1 -type d | wc -l
```

### 1.2 GPU smoke eval-only（后台队列，1 卡，最多 1 个并发）

```bash
export SYNTH_ROOT="$SYNTH_SMOKE"

nohup /usr/bin/time -v env \
  CODE_DIR="$CODE_DIR" \
  SYNTH_ROOT="$SYNTH_ROOT" \
  RESULT_ROOT="$RESULT_ROOT" \
  MODEL_PATH="$MODEL_PATH" \
  BENCHMARK_DIR="$BENCHMARK_DIR_D1_42" \
  GPU_IDS="${GPU_IDS:-0}" \
  MAX_PARALLEL=1 \
  EVAL_MAX_BATCHES=1 \
  EVAL_BATCH_SIZE=2 \
  EVAL_MAX_SEQ_LENGTH=256 \
  bash scripts/RunScripts/launch_unlearning_phase0_eval_queue.sh \
  > "$RESULT_ROOT/run_logs/test20260730_unlearning_eval_queue_smoke_d1_seed42.log" 2>&1 &
```

查看 smoke 队列：

```bash
tail -n 80 "$RESULT_ROOT/run_logs/test20260730_unlearning_eval_queue_smoke_d1_seed42.log"
find "$RESULT_ROOT/result_logs/unlearning_phase0" -path '*d1_seed42_smoke_code*' -name '*.json' | sort
```

### 1.3 汇总 smoke

```bash
nohup python -u scripts/Analysis/summarize_unlearning_dividend_eval.py \
  --eval_json_glob "$RESULT_ROOT/result_logs/unlearning_phase0/d1_seed42_smoke_code/*/*.json" \
  --output_csv "$RESULT_ROOT/result_files/unlearning_phase0_summary/d1_seed42_smoke_code.csv" \
  --output_json "$RESULT_ROOT/result_files/unlearning_phase0_summary/d1_seed42_smoke_code_go.json" \
  --projection_arm proj_auto \
  > "$RESULT_ROOT/run_logs/test20260730_unlearning_summary_smoke_d1_seed42.log" 2>&1 &
```

## 2. 正式 Phase-0 screening：D1 seed42 全 7 个 forget domain

说明：screening 先用 `EVAL_MAX_BATCHES=10`，目的是快速证伪。若 GO 条件已经明显不成立，按 idea 文档纪律停止，不追加多 seed 稀释负结果。

### 2.1 CPU 生成全域 synthetic checkpoints

```bash
export SYNTH_D1_42="$RESULT_ROOT/result_files/unlearning_phase0/d1_seed42_all_domains"

nohup /usr/bin/time -v python -u scripts/Analysis/build_unlearning_dividend_phase0.py \
  --checkpoint_dir "$SOURCE_CKPT_D1_42" \
  --clients_json "$BENCHMARK_DIR_D1_42/clients.json" \
  --benchmark_dir "$BENCHMARK_DIR_D1_42" \
  --model "$MODEL_PATH" \
  --output_dir "$SYNTH_D1_42" \
  --forget_domains all \
  --projection_ranks auto \
  --task_arith_lambdas 0.5,1.0 \
  --random_trials 1 \
  --expected_clients 35 \
  --seed 42 \
  --force \
  > "$RESULT_ROOT/run_logs/test20260730_unlearning_build_d1_seed42_all.log" 2>&1 &
```

### 2.2 GPU eval-only screening 队列

```bash
export SYNTH_ROOT="$SYNTH_D1_42"

nohup /usr/bin/time -v env \
  CODE_DIR="$CODE_DIR" \
  SYNTH_ROOT="$SYNTH_ROOT" \
  RESULT_ROOT="$RESULT_ROOT" \
  MODEL_PATH="$MODEL_PATH" \
  BENCHMARK_DIR="$BENCHMARK_DIR_D1_42" \
  GPU_IDS="${GPU_IDS:-0}" \
  MAX_PARALLEL="${MAX_PARALLEL:-1}" \
  EVAL_MAX_BATCHES=10 \
  EVAL_BATCH_SIZE=2 \
  EVAL_MAX_SEQ_LENGTH=256 \
  bash scripts/RunScripts/launch_unlearning_phase0_eval_queue.sh \
  > "$RESULT_ROOT/run_logs/test20260730_unlearning_eval_queue_d1_seed42_screening.log" 2>&1 &
```

### 2.3 汇总 GO/NO-GO

```bash
nohup python -u scripts/Analysis/summarize_unlearning_dividend_eval.py \
  --eval_json_glob "$RESULT_ROOT/result_logs/unlearning_phase0/d1_seed42_all_domains/*/*.json" \
  --output_csv "$RESULT_ROOT/result_files/unlearning_phase0_summary/d1_seed42_screening.csv" \
  --output_json "$RESULT_ROOT/result_files/unlearning_phase0_summary/d1_seed42_screening_go.json" \
  --projection_arm proj_auto \
  --min_success_domains 4 \
  > "$RESULT_ROOT/run_logs/test20260730_unlearning_summary_d1_seed42_screening.log" 2>&1 &
```

查看判定：

```bash
python - <<'PY'
import json, os
path = os.path.expandvars("$RESULT_ROOT/result_files/unlearning_phase0_summary/d1_seed42_screening_go.json")
payload = json.load(open(path, "r", encoding="utf-8"))
print(json.dumps(payload["verdict"], indent=2, ensure_ascii=False))
for row in payload["go_rows"]:
    print(row["forget_domain"], row["go_for_domain"], row["conditions"], "delta_proj_pool=", row["delta_proj_minus_pool_all"])
PY
```

## 3. 只有 Phase-0 screening 通过后才跑：full eval 与扩展

### 3.1 D1 seed42 full eval

```bash
export SYNTH_ROOT="$SYNTH_D1_42"
export METRICS_ROOT="$RESULT_ROOT/result_logs/unlearning_phase0_full"
export RUN_LOG_DIR="$RESULT_ROOT/run_logs/unlearning_phase0_full"
export PID_DIR="$RESULT_ROOT/pids/unlearning_phase0_full"
export STATUS_DIR="$RESULT_ROOT/status/unlearning_phase0_full"

nohup /usr/bin/time -v env \
  CODE_DIR="$CODE_DIR" \
  SYNTH_ROOT="$SYNTH_ROOT" \
  RESULT_ROOT="$RESULT_ROOT" \
  METRICS_ROOT="$METRICS_ROOT" \
  RUN_LOG_DIR="$RUN_LOG_DIR" \
  PID_DIR="$PID_DIR" \
  STATUS_DIR="$STATUS_DIR" \
  MODEL_PATH="$MODEL_PATH" \
  BENCHMARK_DIR="$BENCHMARK_DIR_D1_42" \
  GPU_IDS="${GPU_IDS:-0}" \
  MAX_PARALLEL="${MAX_PARALLEL:-1}" \
  EVAL_MAX_BATCHES=0 \
  EVAL_BATCH_SIZE=2 \
  EVAL_MAX_SEQ_LENGTH=256 \
  bash scripts/RunScripts/launch_unlearning_phase0_eval_queue.sh \
  > "$RESULT_ROOT/run_logs/test20260730_unlearning_eval_queue_d1_seed42_full.log" 2>&1 &
```

汇总 full eval：

```bash
nohup python -u scripts/Analysis/summarize_unlearning_dividend_eval.py \
  --eval_json_glob "$RESULT_ROOT/result_logs/unlearning_phase0_full/*/*.json" \
  --output_csv "$RESULT_ROOT/result_files/unlearning_phase0_summary/d1_seed42_full.csv" \
  --output_json "$RESULT_ROOT/result_files/unlearning_phase0_summary/d1_seed42_full_go.json" \
  --projection_arm proj_auto \
  --min_success_domains 4 \
  > "$RESULT_ROOT/run_logs/test20260730_unlearning_summary_d1_seed42_full.log" 2>&1 &
```

### 3.2 rank 敏感性（只在 full eval 值得继续时运行）

```bash
export SYNTH_D1_42_RANK="$RESULT_ROOT/result_files/unlearning_phase0/d1_seed42_rank_sensitivity"

nohup /usr/bin/time -v python -u scripts/Analysis/build_unlearning_dividend_phase0.py \
  --checkpoint_dir "$SOURCE_CKPT_D1_42" \
  --clients_json "$BENCHMARK_DIR_D1_42/clients.json" \
  --benchmark_dir "$BENCHMARK_DIR_D1_42" \
  --model "$MODEL_PATH" \
  --output_dir "$SYNTH_D1_42_RANK" \
  --forget_domains all \
  --projection_ranks auto,1,2,4 \
  --task_arith_lambdas 0.5,1.0 \
  --random_trials 1 \
  --expected_clients 35 \
  --seed 42 \
  --force \
  > "$RESULT_ROOT/run_logs/test20260730_unlearning_build_d1_seed42_rank.log" 2>&1 &
```

然后复用 2.2 的队列命令，把 `SYNTH_ROOT="$SYNTH_D1_42_RANK"`，`METRICS_ROOT` 改成 `unlearning_phase0_rank`。

### 3.3 多 seed / FT 的触发条件

只有 `d1_seed42_full_go.json` 中 `verdict.go=true` 时再进入：

1. D1 seed43/44：把 `BENCHMARK_DIR_D1_42`、`SOURCE_CKPT_D1_42` 换成对应 seed 的路径，重复第 2–3 节。
2. FlowerTune-Mixed：先确认有 `fedplora_v13a_os` 正式 checkpoint 且包含 `global_shared.pt + clients/`；若没有，先补跑 1-round v13a states，再重复本协议。

FlowerTune checkpoint 解析模板：

```bash
export BENCHMARK_DIR_FT_42=${BENCHMARK_DIR_FT_42:-$CODE_DIR/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42}
export SOURCE_CKPT_FT_42="$(python scripts/Analysis/checkpoint_manifest.py \
  --roots $CKPT_SEARCH_ROOTS \
  --resolve \
  --agg_type fedplora_v13a_os \
  --seed 42 \
  --model_contains SmolLM2-135M \
  --benchmark_contains "domain_benchmark_flowertune_mixed_20c_dir05/seed_42" \
  --bundle_contains "r1")"
```

【注意事项】

1. `build_unlearning_dividend_phase0.py` 默认使用相对软链接复用重复 client states，避免 synthetic checkpoint 暴涨；如文件系统不支持软链接，会自动 copy。
2. A1/A2 是 global arm，只评一次；summarizer 会按 forget domain 派生其“remaining-domain macro”。
3. A7 `routed_domain` 只作上界参照，不参与 GO/NO-GO 的四个条件。
4. 若 `proj_auto` 在 D1 seed42 screening 的 `pool_all` 条件都不成立，按原 idea 停止，不跑多 seed 和 FT。
5. 如需全后台自动串行“build → eval → summarize”，可以把 2.1、2.2、2.3 三段放入一个 tmux/nohup 脚本；当前文档先保持每段边界清晰，便于证伪阶段人工看一次方向。

【命令之间的串行和并行逻辑】

```text
前置命令必须先跑完。
1.1 build smoke（CPU）完成后 → 1.2 smoke eval 队列（GPU，内部按 MAX_PARALLEL 并发/串行）→ 1.3 smoke summary。
2.1 build all domains（CPU）完成后 → 2.2 screening eval 队列（GPU，内部按 MAX_PARALLEL 并发/串行）→ 2.3 GO/NO-GO summary。
只有 2.3 GO 后 → 3.1 full eval / 3.2 rank sensitivity / 3.3 多 seed 与 FT。
```
