# I1 Eval-Only：In-Domain / Worst In-Domain（gb 单卡 GPU1 版）

> 由 `order/order_eval_only_worst_indomain_20260723.md` 适配。仅替换路径、conda、GPU 与交互 shell 安全写法；指标定义不变。

对已完成 checkpoint **不重训**，只在各 client 的 matched（home）域独立 test 上评测，产出 `*_matched_domain.json`（**不覆盖**原 result JSON）。主表两列：

```text
in_domain_domain_test_token_accuracy          # client-macro In-Domain
in_domain_domain_test_worst_token_accuracy    # Worst In-Dom.（域内最差）
```

## 0. 统一设置（gb）

```text
<<<<<<< HEAD
代码:         /data/yaominghao/gb/FedPLoRA
Fed 结果根:   /data/yaominghao/gb/result/FedPLoRA          ← 读历史 run（order_0709/0712/…）
0723 批次根:  /data/yaominghao/gb/result/FedPLoRA/order_0723  ← 本订单一切输出
I1 输出:      order_0723/eval_only_worst_indomain_20260723/
模型根:       /data/yaominghao/gb/models/trained_models_LW
Python:       /data/yaominghao/miniconda3/envs/fedplora/bin/python
GPU:          物理 1 号卡；D1 与 FlowerTune 串行
工作量:       39 D1 + 21 FlowerTune = 60 eval-only jobs
```

**路径约定**：读主表 checkpoint 的 result JSON 在 `FedPLoRA/order_MMDD/`；**I1 产物、日志、summary 一律写在** `order_0723/` **下**，不要写到 `FedPLoRA/` 根目录。

=======
代码:     /data/yaominghao/gb/FedPLoRA
结果根:   /data/yaominghao/gb/result/FedPLoRA
模型根:   /data/yaominghao/gb/models/trained_models_LW
Python:   /data/yaominghao/miniconda3/envs/fedplora/bin/python
GPU:      物理 1 号卡（CUDA_VISIBLE_DEVICES=1）；D1 与 FlowerTune **串行**，不要双卡并行
工作量:   39 D1 + 21 FlowerTune = 60 eval-only jobs
```

>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
每个新 shell 先执行：

```bash
exec bash
export CODE_DIR=/data/yaominghao/gb/FedPLoRA
<<<<<<< HEAD
export FED_RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export ORDER_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0723
export OUTPUT_ROOT="$ORDER_ROOT/eval_only_worst_indomain_20260723"
export PY=/data/yaominghao/miniconda3/envs/fedplora/bin/python
export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}"
export GPU_ID=1
mkdir -p "$ORDER_ROOT/launcher_logs" "$OUTPUT_ROOT"
cd "$CODE_DIR"
```



### 误写到 FedPLoRA 根目录时搬迁（一次性）

若已有 `$FED_RESULT_ROOT/eval_only_worst_indomain_20260723/`，迁入 `order_0723`：

```bash
export FED_RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export ORDER_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0723
src="$FED_RESULT_ROOT/eval_only_worst_indomain_20260723"
dst="$ORDER_ROOT/eval_only_worst_indomain_20260723"
mkdir -p "$ORDER_ROOT"
if [[ -d "$src" && "$src" != "$dst" ]]; then
  if [[ -e "$dst" ]]; then
    echo "[conflict] $dst exists; merge manually"
  else
    mv "$src" "$dst"
    echo "[ok] moved -> $dst"
  fi
fi
```



=======
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export PY=/data/yaominghao/miniconda3/envs/fedplora/bin/python
export PATH="/data/yaominghao/miniconda3/envs/fedplora/bin:${PATH}"
export GPU_ID=1
cd "$CODE_DIR"
```

>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
## 1. 需同步到 gb 的代码文件

```text
tasks/fed_train_sft.py
scripts/RunScripts/run_eval_only_matched_domain.sh
scripts/Analysis/summarize_matched_domain_eval.py
```

<<<<<<< HEAD


## 1.5 I1-0 快速检查（重组 order_MMDD 后先跑）

```bash
export FED_RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA

find "$FED_RESULT_ROOT/order_0709" -path '*/OS1_normal/*.json' 2>/dev/null | wc -l   # 期望 3
find "$FED_RESULT_ROOT/order_0712" -path '*NX0_v13a*/*.json' 2>/dev/null | wc -l       # 期望 ≥1
find "$FED_RESULT_ROOT/order_0711" -path '*NX1_v13a*/*.json' 2>/dev/null | wc -l       # 期望 ≥2
find "$FED_RESULT_ROOT/order_0715" -path '*/flowertune_20260715_core8_seed*/result_logs/*/*.json' 2>/dev/null | wc -l   # 期望 21
```



## 1.6 是否跑完（验收计数）

```bash
export OUTPUT_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0723/eval_only_worst_indomain_20260723

echo "D1 matched_domain:" $(find "$OUTPUT_ROOT/d1" -type f -name '*_matched_domain.json' 2>/dev/null | wc -l) "/ 39"
echo "Flower matched_domain:" $(find "$OUTPUT_ROOT/flowertune" -type f -name '*_matched_domain.json' 2>/dev/null | wc -l) "/ 21"
test -f "$OUTPUT_ROOT/d1_summary.tsv" && echo "d1_summary: ok" || echo "d1_summary: missing"
test -f "$OUTPUT_ROOT/flowertune_summary.tsv" && echo "flower_summary: ok" || echo "flower_summary: missing"

# 若 d1=39 且 flower=21 且两个 summary 存在 → I1 完成
# 仅有 d1/ + source txt + d1_eval.log 且 d1<39 → D1 还在跑或未跑完 Flower
if [[ -f "$OUTPUT_ROOT/d1_eval.pid" ]]; then
  pid=$(cat "$OUTPUT_ROOT/d1_eval.pid")
  ps -p "$pid" >/dev/null 2>&1 && echo "D1 job still running pid=$pid" || echo "D1 pid file stale (job ended)"
fi
```



## 2. 构建主表 run 列表（0 GPU）

**【gb】** 源 JSON 在 `FED_RESULT_ROOT/order_MMDD/`；列表文件写到 `OUTPUT_ROOT`（`order_0723/...`）。

```bash
cd /data/yaominghao/gb/FedPLoRA
export FED_RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export ORDER_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0723
export OUTPUT_ROOT="$ORDER_ROOT/eval_only_worst_indomain_20260723"
export RUNNER=scripts/RunScripts/run_eval_only_matched_domain.sh
export SUMMARIZER=scripts/Analysis/summarize_matched_domain_eval.py
mkdir -p "$OUTPUT_ROOT/d1" "$OUTPUT_ROOT/flowertune" "$ORDER_ROOT/launcher_logs"

bash -c '
set -eo pipefail
FED_RESULT_ROOT="'"$FED_RESULT_ROOT"'"
OUTPUT_ROOT="'"$OUTPUT_ROOT"'"

first_existing_dir() {
  local d
  for d in "$@"; do
    if [[ -d "$d" ]]; then echo "$d"; return 0; fi
  done
  echo "[error] none of these dirs exist:" "$@" >&2
  return 1
}

d1_baseline_base_for_seed() {
  local seed="$1"
  first_existing_dir \
    "$FED_RESULT_ROOT/order_0709/os_20260709_baseline_35c_dir05_r1_finaleval_seed${seed}/result_logs" \
    "$FED_RESULT_ROOT/os_20260709_baseline_35c_dir05_r1_finaleval_seed${seed}/result_logs"
}

d1_ours_dir_for_seed() {
  local seed="$1"
  case "$seed" in
    42) first_existing_dir \
          "$FED_RESULT_ROOT/order_0712/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_v13a_os_split42_train42" \
          "$FED_RESULT_ROOT/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_v13a_os_split42_train42" ;;
    43) first_existing_dir \
          "$FED_RESULT_ROOT/order_0711/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/result_logs/NX1_v13a_os_split43_train43" \
          "$FED_RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/result_logs/NX1_v13a_os_split43_train43" ;;
    44) first_existing_dir \
          "$FED_RESULT_ROOT/order_0711/v13_20260711_nx1_35c_dir05_r1_finaleval_seed44/result_logs/NX1_v13a_os_split44_train44" \
          "$FED_RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed44/result_logs/NX1_v13a_os_split44_train44" ;;
    *) echo "[error] unsupported seed=$seed" >&2; return 1 ;;
  esac
}

flower_base_for_seed() {
  local seed="$1"
  first_existing_dir \
    "$FED_RESULT_ROOT/order_0715/flowertune_20260715_core8_seed${seed}/result_logs" \
    "$FED_RESULT_ROOT/flowertune_20260715_core8_seed${seed}/result_logs"
=======
在含 `/data/yaominghao/gb/models/trained_models_LW` 与正式 result 目录的节点执行（即 gb 本机）。

## 2. 构建主表 run 列表（0 GPU）

**【gb】** 列表构建放在 `bash -c` 子 shell 里，避免交互 tmux 中 `exit 1` 直接杀掉 pane。

```bash
cd /data/yaominghao/gb/FedPLoRA
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export OUTPUT_ROOT="$RESULT_ROOT/eval_only_worst_indomain_20260723"
export RUNNER=scripts/RunScripts/run_eval_only_matched_domain.sh
export SUMMARIZER=scripts/Analysis/summarize_matched_domain_eval.py

bash -c '
set -eo pipefail
RESULT_ROOT="'"$RESULT_ROOT"'"
OUTPUT_ROOT="'"$OUTPUT_ROOT"'"
RUNNER="'"$RUNNER"'"
mkdir -p "$OUTPUT_ROOT/d1" "$OUTPUT_ROOT/flowertune"

flower_base_for_seed() {
  local seed="$1"
  local a="$RESULT_ROOT/flowertune_20260715_core8_seed${seed}/result_logs"
  local b="$RESULT_ROOT/order_0715/flowertune_20260715_core8_seed${seed}/result_logs"
  if [[ -d "$a" ]]; then echo "$a"; return 0; fi
  if [[ -d "$b" ]]; then echo "$b"; return 0; fi
  echo "[error] missing FlowerTune result_logs for seed=${seed}" >&2
  return 1
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
}

D1_METHOD_DIRS=(
  OS1_normal OS1_ffa OS1_flora OS1_flexlora OS1_ecolora OS1_fedsa_lora
  OS1_feddat OS1_yoco OS1_fedalt OS1_hydralora OS1_hilora OS1_fedlease
)

D1_RESULTS=()
for seed in 42 43 44; do
<<<<<<< HEAD
  base="$(d1_baseline_base_for_seed "$seed")"
=======
  base="$RESULT_ROOT/os_20260709_baseline_35c_dir05_r1_finaleval_seed${seed}/result_logs"
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
  for method_dir in "${D1_METHOD_DIRS[@]}"; do
    mapfile -t hits < <(find "$base/$method_dir" -maxdepth 1 -type f -name "*.json" | sort)
    if [[ "${#hits[@]}" -ne 1 ]]; then
      echo "Expected one result JSON, found ${#hits[@]}: $base/$method_dir" >&2
      exit 1
    fi
    D1_RESULTS+=("${hits[0]}")
  done
<<<<<<< HEAD
  result_dir="$(d1_ours_dir_for_seed "$seed")"
=======
done

D1_OURS_DIRS=(
  "$RESULT_ROOT/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_v13a_os_split42_train42"
  "$RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/result_logs/NX1_v13a_os_split43_train43"
  "$RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed44/result_logs/NX1_v13a_os_split44_train44"
)
for result_dir in "${D1_OURS_DIRS[@]}"; do
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
  mapfile -t hits < <(find "$result_dir" -maxdepth 1 -type f -name "*.json" | sort)
  if [[ "${#hits[@]}" -ne 1 ]]; then
    echo "Expected one result JSON, found ${#hits[@]}: $result_dir" >&2
    exit 1
  fi
  D1_RESULTS+=("${hits[0]}")
done

if [[ "${#D1_RESULTS[@]}" -ne 39 ]]; then
  echo "D1 list must contain 39 runs, found ${#D1_RESULTS[@]}" >&2
  exit 1
fi

FLOWER_METHOD_DIRS=(
  N9_flower_normal N9_flower_ecolora N9_flower_fedsa_lora N9_flower_fedalt
  N9_flower_hydralora N9_flower_fedlease N7_ours_flower_v13a
)

FLOWER_RESULTS=()
for seed in 42 43 44; do
  base="$(flower_base_for_seed "$seed")"
  for method_dir in "${FLOWER_METHOD_DIRS[@]}"; do
    mapfile -t hits < <(find "$base/$method_dir" -maxdepth 1 -type f -name "*.json" | sort)
    if [[ "${#hits[@]}" -ne 1 ]]; then
      echo "Expected one result JSON, found ${#hits[@]}: $base/$method_dir" >&2
      exit 1
    fi
    FLOWER_RESULTS+=("${hits[0]}")
  done
done

if [[ "${#FLOWER_RESULTS[@]}" -ne 21 ]]; then
  echo "FlowerTune list must contain 21 runs, found ${#FLOWER_RESULTS[@]}" >&2
  exit 1
fi

printf "%s\n" "${D1_RESULTS[@]}" > "$OUTPUT_ROOT/d1_source_results.txt"
printf "%s\n" "${FLOWER_RESULTS[@]}" > "$OUTPUT_ROOT/flowertune_source_results.txt"
echo "[I1][ok] D1=${#D1_RESULTS[@]} Flower=${#FLOWER_RESULTS[@]}"
<<<<<<< HEAD
echo "[I1][ok] lists -> $OUTPUT_ROOT"
=======
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
'
echo "[I1] build lists exit code=$?"
```

<<<<<<< HEAD


## 3. 启动 matched-domain eval-only（GPU1 串行）

`EVAL_MAX_BATCHES=0` = 全量独立域 test。日志可镜像到 `order_0723/launcher_logs/`。
=======
FedP-OneShot  intentionally 排除。若某目录 JSON 数量 ≠ 1，先 `find ... -name '*.json'` 排查是否缺跑或路径漂移。

## 3. 启动 matched-domain eval-only（GPU1 串行）

`EVAL_MAX_BATCHES=0` = 全量独立域 test；不训练、不聚合。
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f

**Stage A — D1（39 jobs）**

```bash
cd /data/yaominghao/gb/FedPLoRA
<<<<<<< HEAD
export ORDER_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0723
export OUTPUT_ROOT="$ORDER_ROOT/eval_only_worst_indomain_20260723"
=======
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export OUTPUT_ROOT="$RESULT_ROOT/eval_only_worst_indomain_20260723"
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
export RUNNER=scripts/RunScripts/run_eval_only_matched_domain.sh
mapfile -t D1_RESULTS < "$OUTPUT_ROOT/d1_source_results.txt"

nohup env \
  CUDA_VISIBLE_DEVICES=1 \
  EVAL_MAX_BATCHES=0 \
  MATCHED_DOMAIN_OUTPUT_ROOT="$OUTPUT_ROOT/d1" \
  bash "$RUNNER" "${D1_RESULTS[@]}" \
  > "$OUTPUT_ROOT/d1_eval.log" 2>&1 &
echo $! > "$OUTPUT_ROOT/d1_eval.pid"
<<<<<<< HEAD
ln -sf "$OUTPUT_ROOT/d1_eval.log" "$ORDER_ROOT/launcher_logs/test20260723_i1_d1_eval.log"
echo "D1 PID=$(cat "$OUTPUT_ROOT/d1_eval.pid")"
```

```bash
tail -f "$OUTPUT_ROOT/d1_eval.log"
find "$OUTPUT_ROOT/d1" -type f -name '*_matched_domain.json' | wc -l   # 目标 39

export OUTPUT_ROOT=/data/yaominghao/gb/result/FedPLoRA/eval_only_worst_indomain_20260723
echo "D1:" $(find "$OUTPUT_ROOT/d1" -name '*_matched_domain.json' 2>/dev/null | wc -l) "/ 39"
echo "Flower:" $(find "$OUTPUT_ROOT/flowertune" -name '*_matched_domain.json' 2>/dev/null | wc -l) "/ 21"
ls "$OUTPUT_ROOT"/*.tsv 2>/dev/null
tail -n 5 "$OUTPUT_ROOT/d1_eval.log"
```

**Stage B — FlowerTune（21 jobs，D1 结束后再跑）**

```bash
cd /data/yaominghao/gb/FedPLoRA
export ORDER_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0723
export OUTPUT_ROOT="$ORDER_ROOT/eval_only_worst_indomain_20260723"
=======
echo "D1 PID=$(cat "$OUTPUT_ROOT/d1_eval.pid")  log=$OUTPUT_ROOT/d1_eval.log"
```

进度（勿在交互 shell 里 `wait`，用计数/日志）：

```bash
tail -f "$OUTPUT_ROOT/d1_eval.log"
find "$OUTPUT_ROOT/d1" -type f -name '*_matched_domain.json' | wc -l   # 目标 39
```

**Stage B — FlowerTune（21 jobs，等 D1 结束后再贴）**

```bash
cd /data/yaominghao/gb/FedPLoRA
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export OUTPUT_ROOT="$RESULT_ROOT/eval_only_worst_indomain_20260723"
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
export RUNNER=scripts/RunScripts/run_eval_only_matched_domain.sh
mapfile -t FLOWER_RESULTS < "$OUTPUT_ROOT/flowertune_source_results.txt"

nohup env \
  CUDA_VISIBLE_DEVICES=1 \
  EVAL_MAX_BATCHES=0 \
  MATCHED_DOMAIN_OUTPUT_ROOT="$OUTPUT_ROOT/flowertune" \
  bash "$RUNNER" "${FLOWER_RESULTS[@]}" \
  > "$OUTPUT_ROOT/flowertune_eval.log" 2>&1 &
echo $! > "$OUTPUT_ROOT/flowertune_eval.pid"
<<<<<<< HEAD
ln -sf "$OUTPUT_ROOT/flowertune_eval.log" "$ORDER_ROOT/launcher_logs/test20260723_i1_flower_eval.log"
echo "Flower PID=$(cat "$OUTPUT_ROOT/flowertune_eval.pid")"
=======
echo "Flower PID=$(cat "$OUTPUT_ROOT/flowertune_eval.pid")  log=$OUTPUT_ROOT/flowertune_eval.log"
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
```

```bash
tail -f "$OUTPUT_ROOT/flowertune_eval.log"
find "$OUTPUT_ROOT/flowertune" -type f -name '*_matched_domain.json' | wc -l   # 目标 21
```

<<<<<<< HEAD


=======
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
## 4. 验收与三 seed 汇总（0 GPU）

```bash
cd /data/yaominghao/gb/FedPLoRA
<<<<<<< HEAD
export ORDER_ROOT=/data/yaominghao/gb/result/FedPLoRA/order_0723
export OUTPUT_ROOT="$ORDER_ROOT/eval_only_worst_indomain_20260723"
=======
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export OUTPUT_ROOT="$RESULT_ROOT/eval_only_worst_indomain_20260723"
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
export PY=/data/yaominghao/miniconda3/envs/fedplora/bin/python
export SUMMARIZER=scripts/Analysis/summarize_matched_domain_eval.py

bash -c '
set -eo pipefail
OUTPUT_ROOT="'"$OUTPUT_ROOT"'"
PY="'"$PY"'"
SUMMARIZER="'"$SUMMARIZER"'"

find "$OUTPUT_ROOT/d1" -type f -name "*_matched_domain.json" | sort \
  > "$OUTPUT_ROOT/d1_eval_results.txt"
find "$OUTPUT_ROOT/flowertune" -type f -name "*_matched_domain.json" | sort \
  > "$OUTPUT_ROOT/flowertune_eval_results.txt"

d1_n=$(wc -l < "$OUTPUT_ROOT/d1_eval_results.txt")
fl_n=$(wc -l < "$OUTPUT_ROOT/flowertune_eval_results.txt")
if [[ "$d1_n" -ne 39 || "$fl_n" -ne 21 ]]; then
  echo "[I1][error] counts d1=$d1_n flower=$fl_n (expect 39/21)" >&2
  exit 1
fi

"$PY" "$SUMMARIZER" "$OUTPUT_ROOT/d1" | tee "$OUTPUT_ROOT/d1_summary.tsv"
"$PY" "$SUMMARIZER" "$OUTPUT_ROOT/flowertune" | tee "$OUTPUT_ROOT/flowertune_summary.tsv"
<<<<<<< HEAD
echo "[I1][ok] summaries -> $OUTPUT_ROOT"
=======
echo "[I1][ok] summaries written"
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
'
echo "[I1] summarize exit code=$?"
```

<<<<<<< HEAD


## 5. 与 §7.3 P2（E-prep/E0/E1）的区别


| 项      | I1（本文件）                                         | P2 external lm-eval            |
| ------ | ----------------------------------------------- | ------------------------------ |
| 脚本     | `run_eval_only_matched_domain.sh`               | `run_external_lm_eval.py`      |
| 任务     | 各 client home 域 test                            | MMLU / PubMedQA / MBPP         |
| 输出根    | `order_0723/eval_only_worst_indomain_20260723/` | `order_0723/external_*`        |
| 读源 run | `FedPLoRA/order_0709` 等                         | `order_0723` checkpoint export |
| 订单     | 本文件                                             | `order_gb_0723new.md` §7.3 E-* |


=======
## 5. 与 §7.3 P2（E-prep/E0/E1）的区别

| 项 | I1（本文件） | P2 external lm-eval |
|----|-------------|---------------------|
| 脚本 | `run_eval_only_matched_domain.sh` | `run_external_lm_eval.py` |
| 任务 | 各 client home 域 test | MMLU / PubMedQA / MBPP |
| 输出 | `*_matched_domain.json` | `external_eval_summary.json` |
| 订单 | 本文件 | `order_gb_0723new.md` §7.3 E-* |
>>>>>>> 604bd264148a6c6b446055152dab09eee9ff6c6f
