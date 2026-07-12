# FedPLoRA one-shot v13 实验命令（gb 服务器）- 20260711

> 由 `order/order_20260711.md` 适配。超参数、算法与实验设计不变；仅修改路径、conda、**单卡串行**（gb 不做 minghao 侧 0–3 四卡并行）。

######### FedPLoRA one-shot v13 主算法复核与阶段流水命令 #########

【命令介绍】

本文件分两部分：

1. 单实验 nohup 命令区：每个实验一条 `nohup bash scripts/RunScripts/run_20260711_one_experiment.sh ...`；每条对应独立 run log、result log、client state、checkpoint、pid 文件。
2. 阶段流水脚本区：pipeline nohup 命令；同阶段内按 `MAX_PARALLEL` 并行，阶段间 gate。

注意：smoke 和正式流水分开。先单独跑 smoke；pipeline 默认只检查 smoke 是否通过，不主动跑 smoke。

**gb 与 minghao 的关键差异**：gb 单卡内存有限，默认 **串行**（`MAX_PARALLEL=1`，`GPU_LIST` 只设一张卡）；不要用 minghao 文档里的 `--gpu 0/1/2/3` 四路并行。

**v13 路径修复（20260712）**：旧版 `run_20260711_one_experiment.sh` 在 source preflight 后会把 CLI 的 `--run-id-prefix` 覆盖成 **v12**，导致 NX1 产物误写入 `v12_20260709_main_..._seed42`，pipeline 检查 `v13_20260711_nx1_...` 失败。现已修复：使用 `preflight_20260711_main_algorithm.sh`（v13 角色）并保留 CLI 前缀。**重跑前务必 git pull 同步最新脚本。**

【命令目的】

本轮优先验证：

- `fedplora_v13a_os`：主算法，true A-delta sketch + routed B，`alpha=1.0`，去 μ，去 A/B local regularizer。
- `fedplora_v13b_os_bonly`：低通信归因分支，v8 routed 的 one-shot/cached-A 通信修正版。
- 最高优先级：**NX1**，v13a/v13b × split/train seed43/44。

【命令设置（gb）】

```text
代码目录: /data/yaominghao/gb/FedPLoRA
模型: /data/yaominghao/gb/models/SmolLM2-135M
数据: data/domain_benchmark_35c_dir05/seed_{42,43,44}
客户端: 35 = 7 domains × 5 clients
轮次: one-shot / r=1
local_epochs: 1
学习率: 0.0002
LoRA: r=8, alpha=16, dropout=0.05
target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
评测: eval_final_only; formal 默认 EVAL_MAX_BATCHES=0
GPU: 默认物理 1 号卡；export GB_GPU=1（空闲时可改 0）；下文用 ${GB_GPU:-1}
conda: fedplora
```

【路径对照（minghao → gb）】


| 项          | order_20260711.md                         | order_gb_0711.md                                |
| ---------- | ----------------------------------------- | ----------------------------------------------- |
| conda      | `FedRepo2`                                | `fedplora`                                      |
| 代码         | `/data2/minghao/code/FedPLoRA-main`       | `/data/yaominghao/gb/FedPLoRA`                  |
| 模型         | `/data2/minghao/model/SmolLM2-135M`       | `/data/yaominghao/gb/models/SmolLM2-135M`       |
| 结果         | `/data2/minghao/result/FedPLoRA/`         | `/data/yaominghao/gb/result/FedPLoRA/`          |
| checkpoint | `/data2/minghao/model/trained_models_LW/` | `/data/yaominghao/gb/models/trained_models_LW/` |
| GPU        | `0 1 2 3` 四卡并行                            | `${GB_GPU:-1}` 单卡串行（默认 1 号卡）                  |
| split-seed | `--split-seed` 由脚本自动切 benchmark           | 同左（`run_20260711_one_experiment.sh` 已正确处理）      |


【实验产物位置说明】

```text
单实验 run_logs:
/data/yaominghao/gb/result/FedPLoRA/<RUN_ID>_seed*/run_logs/

单实验 result_logs:
/data/yaominghao/gb/result/FedPLoRA/<RUN_ID>_seed*/result_logs/<method>/

单实验 client_states:
/data/yaominghao/gb/result/FedPLoRA/<RUN_ID>_seed*/result_files/client_states/<method>/

单实验 checkpoint:
/data/yaominghao/gb/models/trained_models_LW/<RUN_ID>_seed*/<method>_*

单实验 PID:
/data/yaominghao/gb/result/FedPLoRA/<RUN_ID>_seed*/pids/<method>.pid

pipeline 控制日志 / PID / gate:
/data/yaominghao/gb/result/FedPLoRA/<PIPELINE_RUN_ID>/pipeline_logs/pipeline.log
/data/yaominghao/gb/result/FedPLoRA/<PIPELINE_RUN_ID>/pids/
/data/yaominghao/gb/result/FedPLoRA/<PIPELINE_RUN_ID>/gates/
```

【实验前置命令】

## 0.1 代码同步

minghao 侧用 `sync_code_20260709_to_server.sh` 推到 minghao 服务器；**gb 需单独 git pull / rsync 到 `/data/yaominghao/gb/FedPLoRA`**，确保含以下脚本：

```text
scripts/RunScripts/preflight_20260711_main_algorithm.sh   # v13 专用，必含
scripts/RunScripts/run_20260711_one_experiment.sh
scripts/RunScripts/run_20260711_oneshot_pipeline.sh
methods/v13/
```

**重跑前在 gb 上执行**：`cd /data/yaominghao/gb/FedPLoRA && git pull`（或 rsync 同步上述文件）。

## 0.2 gb 环境变量（每次开新 shell 先执行）

`run_20260711_one_experiment.sh` 内部会 source preflight，但 **MODEL_PATH / RESULT_ROOT 默认仍是 minghao 路径**，必须先 export：

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA

export CODE_DIR=/data/yaominghao/gb/FedPLoRA
export CONDA_ENV_NAME=fedplora
export MODEL_PATH=/data/yaominghao/gb/models/SmolLM2-135M
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models/trained_models_LW
export BENCHMARK_DIR_MAIN=$CODE_DIR/data/domain_benchmark_35c_dir05/seed_42
export TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export GB_GPU=1

conda activate fedplora
```

可选 preflight 自检（不跑实验，只验证 benchmark / v13 import）：

```bash
source scripts/RunScripts/preflight_20260711_main_algorithm.sh
# 应看到 [preflight][ok] v13 preflight loaded. 且不应出现 v12_20260709_main 的 RUN_ROOT
```

## 0.3 gb 串行 helper（推荐）

单卡上每条实验跑完再发下一条，避免内存堆满：

```bash
run_one_exp () {
  local launch_log="$1"
  shift
  nohup bash scripts/RunScripts/run_20260711_one_experiment.sh "$@" \
    > "$launch_log" 2>&1 &
  wait $!
  echo "[done] launch_log=$launch_log"
}
```

---

# 第一部分：每个实验一个命令（gb 串行版）

说明：`run_20260711_one_experiment.sh` 内部 `exec python`，`nohup ... &; wait $!` 会等该实验完整结束。`--split-seed` 会自动把 `BENCHMARK_DIR` 切到 `seed_${split-seed}`。修复后 `[one-exp] log=` 必须落在 **v13** 目录，例如：

```text
NX1: .../v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/run_logs/...
NX4: .../v13_20260711_nx4_personalized_eval_seed43/run_logs/...
```

若仍出现 `v12_20260709_main_...`，说明脚本未同步，停止并重拉代码。

## 1. Smoke：正式流水前单独运行

gb 单卡：**两条 smoke 串行**，不要像 minghao 那样 `--gpu 0` 和 `--gpu 1` 同时起。

```bash
cd /data/yaominghao/gb/FedPLoRA
mkdir -p /data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/pipeline_logs

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/pipeline_logs/smoke_v13a_os.launch.log \
  --kind smoke \
  --method smoke_v13a_os \
  --agg fedplora_v13a_os \
  --gpu "${GB_GPU:-1}" \
  -- --force_retrain

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/pipeline_logs/smoke_v13b_os_bonly.launch.log \
  --kind smoke \
  --method smoke_v13b_os_bonly \
  --agg fedplora_v13b_os_bonly \
  --gpu "${GB_GPU:-1}" \
  -- --force_retrain
```

Smoke 主日志：

```text
/data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/run_logs/test20260711_main_smoke_smoke_v13a_os_seed42.log
/data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/run_logs/test20260711_main_smoke_smoke_v13b_os_bonly_seed42.log
```

Smoke 检查（确认路径在 v13 smoke 目录）：

```bash
grep -E "Traceback|CUDA out of memory|error" \
  /data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/run_logs/*.log || true
find /data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/result_logs -name '*.json' | sort
grep "\[one-exp\] log=" /data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/pipeline_logs/*.launch.log
```

## 2. NX1：protocol-aligned 主复核（最高优先级）

4 条正式实验，gb **逐条串行**。也可直接用第二部分 pipeline（`MAX_PARALLEL=1`）。

```bash
cd /data/yaominghao/gb/FedPLoRA
mkdir -p /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/NX1_v13a_os_split43_train43.launch.log \
  --kind sft \
  --method NX1_v13a_os_split43_train43 \
  --agg fedplora_v13a_os \
  --seed 43 \
  --split-seed 43 \
  --run-id-prefix v13_20260711_nx1_35c_dir05_r1_finaleval \
  --gpu "${GB_GPU:-1}" \
  -- --force_retrain

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/NX1_v13b_bonly_split43_train43.launch.log \
  --kind sft \
  --method NX1_v13b_bonly_split43_train43 \
  --agg fedplora_v13b_os_bonly \
  --seed 43 \
  --split-seed 43 \
  --run-id-prefix v13_20260711_nx1_35c_dir05_r1_finaleval \
  --gpu "${GB_GPU:-1}" \
  -- --force_retrain

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/NX1_v13a_os_split44_train44.launch.log \
  --kind sft \
  --method NX1_v13a_os_split44_train44 \
  --agg fedplora_v13a_os \
  --seed 44 \
  --split-seed 44 \
  --run-id-prefix v13_20260711_nx1_35c_dir05_r1_finaleval \
  --gpu "${GB_GPU:-1}" \
  -- --force_retrain

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/NX1_v13b_bonly_split44_train44.launch.log \
  --kind sft \
  --method NX1_v13b_bonly_split44_train44 \
  --agg fedplora_v13b_os_bonly \
  --seed 44 \
  --split-seed 44 \
  --run-id-prefix v13_20260711_nx1_35c_dir05_r1_finaleval \
  --gpu "${GB_GPU:-1}" \
  -- --force_retrain
```

NX1 产物目录示例：

```text
/data/yaominghao/gb/result/FedPLoRA/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/result_logs/NX1_v13a_os_split43_train43/
/data/yaominghao/gb/result/FedPLoRA/v13_20260711_nx1_35c_dir05_r1_finaleval_seed44/result_logs/NX1_v13b_bonly_split44_train44/
```

## 3. NX4：cold-start/select eval

建议在 NX1 指标通过后运行。gb 三条串行。

```bash
cd /data/yaominghao/gb/FedPLoRA
mkdir -p /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/X2_v13_coldstart_select_seed42.launch.log \
  --kind personalized_eval \
  --method X2_v13_coldstart_select_seed42 \
  --seed 42 \
  --split-seed 42 \
  --run-id-prefix v13_20260711_nx4_personalized_eval \
  --gpu "${GB_GPU:-1}" \
  -- --v11c_mu 0.4

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/X2_v13_coldstart_select_seed43.launch.log \
  --kind personalized_eval \
  --method X2_v13_coldstart_select_seed43 \
  --seed 43 \
  --split-seed 43 \
  --run-id-prefix v13_20260711_nx4_personalized_eval \
  --gpu "${GB_GPU:-1}" \
  -- --v11c_mu 0.4

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/X2_v13_coldstart_select_seed44.launch.log \
  --kind personalized_eval \
  --method X2_v13_coldstart_select_seed44 \
  --seed 44 \
  --split-seed 44 \
  --run-id-prefix v13_20260711_nx4_personalized_eval \
  --gpu "${GB_GPU:-1}" \
  -- --v11c_mu 0.4
```

## 4. Optional NX3：ablation 补洞

默认不抢主批次；NX1 接近阈值或需解释 alpha/μ 时再跑。gb 三条串行。

```bash
cd /data/yaominghao/gb/FedPLoRA
mkdir -p /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/NX3_v11a_alpha100_split42_train43.launch.log \
  --kind sft \
  --method NX3_v11a_alpha100_split42_train43 \
  --agg fedplora_v11a_relaxed_a \
  --seed 43 \
  --split-seed 42 \
  --run-id-prefix v13_20260711_nx3_ablation_split42_r1_finaleval \
  --gpu "${GB_GPU:-1}" \
  -- --v10_a_correction_alpha 1.0 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0 --force_retrain

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/NX3_v11c_mu020_split42_train42.launch.log \
  --kind sft \
  --method NX3_v11c_mu020_split42_train42 \
  --agg fedplora_v11c_gmix \
  --seed 42 \
  --split-seed 42 \
  --run-id-prefix v13_20260711_nx3_ablation_split42_r1_finaleval \
  --gpu "${GB_GPU:-1}" \
  -- --v10_a_correction_alpha 1.0 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0 --v11_global_b_mix_mu 0.2 --force_retrain

run_one_exp \
  /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/NX3_v11c_mu020_split42_train44.launch.log \
  --kind sft \
  --method NX3_v11c_mu020_split42_train44 \
  --agg fedplora_v11c_gmix \
  --seed 44 \
  --split-seed 42 \
  --run-id-prefix v13_20260711_nx3_ablation_split42_r1_finaleval \
  --gpu "${GB_GPU:-1}" \
  -- --v10_a_correction_alpha 1.0 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0 --v11_global_b_mix_mu 0.2 --force_retrain
```

---

# 第二部分：阶段流水脚本（gb 单卡版）

说明：pipeline 不跑 smoke；先完成 §1 smoke。gb 用 `**MAX_PARALLEL=1**`，同一时刻只跑 1 个实验。

默认 gate：

```text
NX1_MIN_COMPLETED_RUNS=4
NX1_MIN_V13A_LOCAL=0.604
```

若只想检查产物完整性、不用性能阈值拦截，设 `NX1_MIN_V13A_LOCAL=0`。

## 5. 默认正式流水：检查 smoke → NX2 → NX1 gate → NX4

**先完成 §0.2 环境 export**，再：

```bash
cd /data/yaominghao/gb/FedPLoRA

export PIPELINE_RUN_ID=v13_20260711_pipeline_control_$(date +%Y%m%d_%H%M%S)
mkdir -p /data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pipeline_logs
mkdir -p /data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pids

REQUIRE_SMOKE_OK=1 \
GPU_LIST="${GB_GPU:-1}" \
MAX_PARALLEL=1 \
NX1_MIN_COMPLETED_RUNS=4 \
NX1_MIN_V13A_LOCAL=0.604 \
PIPELINE_RUN_ID="$PIPELINE_RUN_ID" \
nohup bash scripts/RunScripts/run_20260711_oneshot_pipeline.sh \
  > /data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pipeline_logs/pipeline.log 2>&1 &

echo $! > /data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pids/pipeline.nohup.pid
echo "[pipeline] log=/data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pipeline_logs/pipeline.log"
```

跟踪 pipeline：

```bash
tail -f /data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pipeline_logs/pipeline.log
cat /data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/gates/nx1_gate.json 2>/dev/null || true
```

## 6. 带 optional NX3 的流水

```bash
cd /data/yaominghao/gb/FedPLoRA

export PIPELINE_RUN_ID=v13_20260711_pipeline_nx3_$(date +%Y%m%d_%H%M%S)
mkdir -p /data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pipeline_logs
mkdir -p /data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pids

RUN_NX3=1 \
REQUIRE_SMOKE_OK=1 \
GPU_LIST="${GB_GPU:-1}" \
MAX_PARALLEL=1 \
NX1_MIN_COMPLETED_RUNS=4 \
NX1_MIN_V13A_LOCAL=0.604 \
PIPELINE_RUN_ID="$PIPELINE_RUN_ID" \
nohup bash scripts/RunScripts/run_20260711_oneshot_pipeline.sh \
  > /data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pipeline_logs/pipeline.log 2>&1 &

echo $! > /data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pids/pipeline.nohup.pid
echo "[pipeline] log=/data/yaominghao/gb/result/FedPLoRA/${PIPELINE_RUN_ID}/pipeline_logs/pipeline.log"
```

---

# 命令之间的串行和并行逻辑

## A. gb 手工模式（推荐）

```text
Smoke:
  smoke_v13a_os          → wait 结束
  smoke_v13b_os_bonly    → wait 结束

NX1:
  split43 v13a           → wait 结束
  split43 v13b           → wait 结束
  split44 v13a           → wait 结束
  split44 v13b           → wait 结束

NX4:
  seed42 eval            → wait 结束
  seed43 eval            → wait 结束
  seed44 eval            → wait 结束

NX3 optional:
  三条 ablation          → 各 wait 结束
```

## B. pipeline 自动逻辑（gb 上 MAX_PARALLEL=1）

与 minghao 相同阶段顺序，但 NX1/NX4/NX3 各阶段内**一次只跑 1 个实验**，不会四路抢 GPU/内存。

## C. 日志关系

```text
pipeline 自己的 nohup 日志:
  /data/yaominghao/gb/result/FedPLoRA/<PIPELINE_RUN_ID>/pipeline_logs/pipeline.log

每个实验主日志:
  /data/yaominghao/gb/result/FedPLoRA/<RUN_ID>_seed*/run_logs/*.log

每个实验 PID:
  /data/yaominghao/gb/result/FedPLoRA/<RUN_ID>_seed*/pids/<method>.pid
```

---

# gb 运行提示

```text
1. 默认 1 号卡：export GB_GPU=1；若 1 号卡被占用可 export GB_GPU=0。所有 --gpu 用 "${GB_GPU:-1}"。
2. v13 路径：每条实验 launch 日志里必须有 [one-exp] RUN_ID_PREFIX=v13_... 且 log= 指向 v13 目录；禁止 v12_20260709_main。
3. 误写入 v12 的旧 NX1 结果不可用；修复后请重跑 smoke + pipeline（或 §2 NX1 + §3 NX4）。
4. 杀误跑进程：pkill -u "$USER" -f "run_20260711_one_experiment.sh" ; pkill -u "$USER" -f "run_20260711_oneshot_pipeline.sh" ; pkill -u "$USER" -f "fed_train_sft.py"
5. 若 pipeline 报 smoke 未通过，先跑完 §1 再重试 pipeline（REQUIRE_SMOKE_OK=1）。
6. baseline 实验请继续用 order_gb_0709.md，不要与本 v13 order 混跑同一 GPU。
```

---

# 重跑全流程（修复后推荐顺序）

旧 pipeline 因路径错误已停；**同步代码后按此顺序从头重跑**：

```bash
# 0) 同步 + 环境（§0.1 + §0.2 全部复制）
# 1) §1 smoke 两条（--force_retrain）
# 2) 确认 [one-exp] log= 在 v13_20260711_smoke_seed42 下
# 3) §5 启动 pipeline（MAX_PARALLEL=1, GPU_LIST="${GB_GPU:-1}"）
# 4) tail -f pipeline.log；NX1 通过后应出现 gates/nx1_gate.json 且 ok=true
# 5) 自动进入 NX4；最终检查：
find $RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed*/result_logs/NX1_* -name '*.json'
find $RESULT_ROOT/v13_20260711_nx4_personalized_eval_seed*/result_logs -name '*.json'
```

NX1 期望目录（修复后）：

```text
/data/yaominghao/gb/result/FedPLoRA/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/
/data/yaominghao/gb/result/FedPLoRA/v13_20260711_nx1_35c_dir05_r1_finaleval_seed44/
```

