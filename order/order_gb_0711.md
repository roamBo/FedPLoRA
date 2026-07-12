# FedPLoRA one-shot v13 实验命令（gb 服务器）- 20260711

> 由 `order/order_20260711.md` 适配。超参数、算法与实验设计不变；仅修改路径、conda、**全部 GPU=1**，并改为 **0705 风格后台指令**（`run_v13_*` 后台 nohup，无 `wait`，可逐条粘贴）。

######### FedPLoRA one-shot v13 主算法复核命令（gb 单卡 GPU1 版） #########

【命令介绍】

本文档包含 5 组命令：

1. 前置检查与公共运行函数（source `preflight_gb_20260711_v13.sh`）。
2. Smoke 测试（正式 NX1 前单独跑）。
3. NX1：protocol-aligned 主复核（最高优先级，v13a/v13b × seed43/44）。
4. NX4：cold-start/select eval（NX1 四条 json 齐全后再跑）。
5. Optional NX3：ablation 补洞（默认不抢主批次）。

**不再推荐 pipeline 串行流水**；附录保留可选 pipeline 命令供参考。

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
GPU: 全部物理 1 号卡（CUDA_VISIBLE_DEVICES=1）
conda: fedplora
```

【路径对照（minghao → gb）】

| 项 | order_20260711.md | order_gb_0711.md |
|----|-------------------|------------------|
| conda | `FedRepo2` | `fedplora` |
| 代码 | `/data2/minghao/code/FedPLoRA-main` | `/data/yaominghao/gb/FedPLoRA` |
| 模型 | `/data2/minghao/model/SmolLM2-135M` | `/data/yaominghao/gb/models/SmolLM2-135M` |
| 结果 | `/data2/minghao/result/FedPLoRA/` | `/data/yaominghao/gb/result/FedPLoRA/` |
| checkpoint | `/data2/minghao/model/trained_models_LW/` | `/data/yaominghao/gb/models/trained_models_LW/` |
| GPU | `0 1 2 3` 四卡并行 | 全部 `GPU=1`，后台 nohup |
| split-seed | `--split-seed` 由脚本自动切 benchmark | 同左 |

【实验产物位置说明】

```text
run_logs:
/data/yaominghao/gb/result/FedPLoRA/${RUN_ID}/run_logs/test20260711_main_*.log

result_logs:
/data/yaominghao/gb/result/FedPLoRA/${RUN_ID}/result_logs/<method>/

client_states:
/data/yaominghao/gb/result/FedPLoRA/${RUN_ID}/result_files/client_states/<method>/

checkpoints:
/data/yaominghao/gb/models/trained_models_LW/${RUN_ID}/<method>_*

launch 日志（preflight + [one-exp] RUN_ROOT 行）:
/data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/<method>.launch.log
```

v13 期望 RUN_ID：

```text
smoke:  v13_20260711_smoke_seed42
NX1:    v13_20260711_nx1_35c_dir05_r1_finaleval_seed{43,44}
NX4:    v13_20260711_nx4_personalized_eval_seed{42,43,44}
NX3:    v13_20260711_nx3_ablation_split42_r1_finaleval_seed{42,43,44}
```

**v13 路径修复（20260712）**：旧版会把 CLI `--run-id-prefix` 覆盖成 v12，产物误写入 `v12_20260709_main_...`。现已修复。**重跑前务必 git pull 同步**，且 launch 日志里 `[one-exp] RUN_ID_PREFIX=` 必须是 `v13_...`。

---

【实验前置命令】

## 0.1 代码同步

gb 需单独 git pull / rsync 到 `/data/yaominghao/gb/FedPLoRA`，确保含：

```text
scripts/RunScripts/preflight_gb_20260711_v13.sh          # gb 并行 helper（本 order 专用）
scripts/RunScripts/preflight_20260711_main_algorithm.sh
scripts/RunScripts/run_20260711_one_experiment.sh
methods/v13/
```

```bash
cd /data/yaominghao/gb/FedPLoRA && git pull
```

## 0.2 加载 preflight（每次开新 shell 先执行）

**source 一次 + `export GPU=1`，后续逐条粘贴 `run_v13_*` 即可后台运行，无需 wait**。单卡 GPU1 同时只跑一个实验，上一条结束或确认 pid 在跑后再贴下一条，避免 OOM。

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
conda activate fedplora
source scripts/RunScripts/preflight_gb_20260711_v13.sh
export GPU=1
```

应看到：

```text
[preflight_gb_v13][ok] GPU default=1 LAUNCH_DIR=...
[preflight_gb_v13][ok] 用法: run_v13_nx1 <method> <agg> <seed> [--force_retrain ...]
```

可选自检（不跑实验）：

```bash
grep -E 'RUN_ID_PREFIX|v13' <<< "$(source scripts/RunScripts/preflight_20260711_main_algorithm.sh 2>&1 | tail -5)" || true
python -m py_compile methods/v13/*.py tasks/fed_train_sft.py
```

## 0.3 运行函数说明

| 函数 | 用途 | 示例 |
|------|------|------|
| `run_v13_smoke` | 1-round smoke | `run_v13_smoke smoke_v13a_os fedplora_v13a_os --force_retrain` |
| `run_v13_nx1` | NX1 SFT（split=train seed） | `run_v13_nx1 NX1_v13a_os_split43_train43 fedplora_v13a_os 43 --force_retrain` |
| `run_v13_nx4` | cold-start eval | `run_v13_nx4 X2_v13_coldstart_select_seed42 42 --v11c_mu 0.4` |
| `run_v13_nx3` | ablation（split≠train） | `run_v13_nx3 NX3_v11a_alpha100_split42_train43 fedplora_v11a_relaxed_a 43 42 --v10_a_correction_alpha 1.0 ...` |

每条命令会 echo `pid=` 和 `main=` 主日志路径。**两层日志**：launch 日志看 preflight；主日志看训练进度。

路径自检（任意一条 launch 后）：

```bash
grep -E '\[one-exp\] (RUN_ID_PREFIX|log)=' "$RESULT_ROOT/manual_launch_logs_20260711/<method>.launch.log"
# 必须含 v13_，禁止 v12_20260709_main
```

若旧 pipeline / 误跑进程仍在，先停掉：

```bash
pkill -u "$USER" -f run_20260711_oneshot_pipeline.sh || true
pkill -u "$USER" -f run_20260711_one_experiment.sh || true
```

---

【实验运行命令】

以下命令均假设已执行 §0.2（`export GPU=1` 已生效）。行首**不要**写 `GPU=0`。

## 1. Smoke：正式 NX1 前单独运行

说明：两条 smoke 串行粘贴（均 GPU1）。smoke 结果不能作为论文数值。

```bash
run_v13_smoke smoke_v13a_os fedplora_v13a_os --force_retrain
run_v13_smoke smoke_v13b_os_bonly fedplora_v13b_os_bonly --force_retrain
```

smoke 主日志：

```text
/data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/run_logs/test20260711_main_smoke_smoke_v13a_os_seed42.log
/data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/run_logs/test20260711_main_smoke_smoke_v13b_os_bonly_seed42.log
```

smoke 检查：

```bash
grep -E "Traceback|CUDA out of memory|error" \
  /data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/run_logs/*.log || true
find /data/yaominghao/gb/result/FedPLoRA/v13_20260711_smoke_seed42/result_logs -name '*.json' | sort
grep "\[one-exp\] log=" /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260711/smoke_*.launch.log
```

---

## 2. NX1：protocol-aligned 主复核（最高优先级）

说明：四条 NX1 逐条粘贴，全部 GPU1。每条后台启动，**等上一条跑完再贴下一条**（或 `tail -f` 确认结束）。

### 2.1 seed=43

```bash
run_v13_nx1 NX1_v13a_os_split43_train43 fedplora_v13a_os 43 --force_retrain
run_v13_nx1 NX1_v13b_bonly_split43_train43 fedplora_v13b_os_bonly 43 --force_retrain
```

### 2.2 seed=44

```bash
run_v13_nx1 NX1_v13a_os_split44_train44 fedplora_v13a_os 44 --force_retrain
run_v13_nx1 NX1_v13b_bonly_split44_train44 fedplora_v13b_os_bonly 44 --force_retrain
```

### 2.3 一键复制：NX1 全部四条

```bash
run_v13_nx1 NX1_v13a_os_split43_train43 fedplora_v13a_os 43 --force_retrain
run_v13_nx1 NX1_v13b_bonly_split43_train43 fedplora_v13b_os_bonly 43 --force_retrain
run_v13_nx1 NX1_v13a_os_split44_train44 fedplora_v13a_os 44 --force_retrain
run_v13_nx1 NX1_v13b_bonly_split44_train44 fedplora_v13b_os_bonly 44 --force_retrain
```

NX1 验收：

```bash
find $RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed*/result_logs/NX1_* -name '*.json' | sort
grep -l "Traceback\|CUDA out of memory" $RESULT_ROOT/v13_20260711_nx1_*/run_logs/*.log || echo "NX1 logs OK"
tail -f $RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/run_logs/*.log
```

期望目录：

```text
/data/yaominghao/gb/result/FedPLoRA/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/
/data/yaominghao/gb/result/FedPLoRA/v13_20260711_nx1_35c_dir05_r1_finaleval_seed44/
```

---

## 3. NX4：cold-start/select eval

说明：NX1 四条 json 齐全后再跑。三条逐条粘贴，全部 GPU1。

### 3.1 seed=42

```bash
run_v13_nx4 X2_v13_coldstart_select_seed42 42 --v11c_mu 0.4
```

### 3.2 seed=43

```bash
run_v13_nx4 X2_v13_coldstart_select_seed43 43 --v11c_mu 0.4
```

### 3.3 seed=44

```bash
run_v13_nx4 X2_v13_coldstart_select_seed44 44 --v11c_mu 0.4
```

### 3.4 一键复制：NX4 全部三条

```bash
run_v13_nx4 X2_v13_coldstart_select_seed42 42 --v11c_mu 0.4
run_v13_nx4 X2_v13_coldstart_select_seed43 43 --v11c_mu 0.4
run_v13_nx4 X2_v13_coldstart_select_seed44 44 --v11c_mu 0.4
```

NX4 验收：

```bash
find $RESULT_ROOT/v13_20260711_nx4_personalized_eval_seed*/result_logs -name '*.json' | sort
```

---

## 4. Optional NX3：ablation 补洞

说明：默认不抢主批次；NX1 接近阈值或需解释 alpha/μ 时再跑。三条逐条粘贴，全部 GPU1。

### 4.1 NX3_v11a_alpha100_split42_train43

```bash
run_v13_nx3 NX3_v11a_alpha100_split42_train43 fedplora_v11a_relaxed_a 43 42 \
  --v10_a_correction_alpha 1.0 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 \
  --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0 --force_retrain
```

### 4.2 NX3_v11c_mu020_split42_train42

```bash
run_v13_nx3 NX3_v11c_mu020_split42_train42 fedplora_v11c_gmix 42 42 \
  --v10_a_correction_alpha 1.0 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 \
  --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0 --v11_global_b_mix_mu 0.2 --force_retrain
```

### 4.3 NX3_v11c_mu020_split42_train44

```bash
run_v13_nx3 NX3_v11c_mu020_split42_train44 fedplora_v11c_gmix 44 42 \
  --v10_a_correction_alpha 1.0 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 \
  --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0 --v11_global_b_mix_mu 0.2 --force_retrain
```

### 4.4 一键复制：NX3 全部三条

```bash
run_v13_nx3 NX3_v11a_alpha100_split42_train43 fedplora_v11a_relaxed_a 43 42 --v10_a_correction_alpha 1.0 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0 --force_retrain
run_v13_nx3 NX3_v11c_mu020_split42_train42 fedplora_v11c_gmix 42 42 --v10_a_correction_alpha 1.0 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0 --v11_global_b_mix_mu 0.2 --force_retrain
run_v13_nx3 NX3_v11c_mu020_split42_train44 fedplora_v11c_gmix 44 42 --v10_a_correction_alpha 1.0 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0 --v11_global_b_mix_mu 0.2 --force_retrain
```

---

# 推荐执行顺序（0705 风格，无 pipeline）

```text
0) git pull + source preflight + export GPU=1
1) §1 smoke 两条（串行，均 GPU1）
2) 确认 launch 日志里 log= 指向 v13_20260711_smoke_seed42
3) §2 NX1 四条（逐条粘贴，等上一条结束）
4) NX1 json 齐全后 §3 NX4 三条
5) 可选 §4 NX3
```

一键验收：

```bash
find $RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed*/result_logs/NX1_* -name '*.json'
find $RESULT_ROOT/v13_20260711_nx4_personalized_eval_seed*/result_logs -name '*.json'
```

---

# 附录：可选 pipeline（不推荐，仅参考）

gb 单卡内存有限时曾用 `MAX_PARALLEL=1` 串行 pipeline。**手工 `run_v13_*` 更灵活**。

```bash
cd /data/yaominghao/gb/FedPLoRA
export PIPELINE_RUN_ID=v13_20260711_pipeline_control_$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULT_ROOT/${PIPELINE_RUN_ID}/pipeline_logs $RESULT_ROOT/${PIPELINE_RUN_ID}/pids

REQUIRE_SMOKE_OK=1 GPU_LIST=1 MAX_PARALLEL=1 NX1_MIN_COMPLETED_RUNS=4 NX1_MIN_V13A_LOCAL=0.604 \
PIPELINE_RUN_ID="$PIPELINE_RUN_ID" \
nohup bash scripts/RunScripts/run_20260711_oneshot_pipeline.sh \
  > $RESULT_ROOT/${PIPELINE_RUN_ID}/pipeline_logs/pipeline.log 2>&1 &

echo $! > $RESULT_ROOT/${PIPELINE_RUN_ID}/pids/pipeline.nohup.pid
echo "[pipeline] log=$RESULT_ROOT/${PIPELINE_RUN_ID}/pipeline_logs/pipeline.log"
```

---

# gb 运行提示

```text
1. 全部 GPU1：§0.2 里 export GPU=1 后，行首勿写 GPU=0。
2. 单卡同时只跑一个实验；多条命令请等上一条结束再贴下一条，避免 OOM。
3. v13 路径：launch 日志 [one-exp] RUN_ID_PREFIX=v13_... 且 log= 指向 v13 目录；禁止 v12_20260709_main。
4. 误写入 v12 的旧 NX1 结果不可用；修复后请重跑 smoke + NX1 + NX4。
5. 杀误跑进程：pkill -u "$USER" -f "run_20260711_one_experiment.sh" ; pkill -u "$USER" -f "run_20260711_oneshot_pipeline.sh" ; pkill -u "$USER" -f "fed_train_sft.py"
6. baseline 实验请继续用 order_gb_0709.md，不要与本 v13 order 混跑同一 GPU。
7. 查看进度：tail -f <主日志路径>（run_v13_* echo 的 main= 行）
8. 查看后台任务：jobs -l 或 ps -u "$USER" -f | grep run_20260711_one_experiment
```
