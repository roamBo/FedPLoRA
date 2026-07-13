# FedPLoRA 20260712：NX0 主表补齐与 strict held-out（gb 服务器）

> 由 `order/order_20260712.md` 适配。超参数、算法与实验设计不变；仅修改路径、conda、**全部 GPU=1**，并改为 **0705/0711 风格后台指令**（`run_v13_`* 后台 nohup，无 `wait`，可逐条粘贴）。

######### FedPLoRA-v13a one-shot 主表补齐 + strict held-out 能力实验（gb 单卡 GPU1 版） #########

【命令介绍】

本文档包含 6 组命令：

1. 前置检查与公共运行函数（source `preflight_gb_20260712_v13.sh`）。
2. Smoke 测试（含 strict held-out smoke；正式 NX0 前单独跑）。
3. P0 fingerprint：三 split benchmark 指纹审计（0 GPU）。
4. P0 NX0：`NX0_v13a_os_split42_train42`，补合法 split42 主表。
5. P2 NX0-v13b：`NX0_v13b_bonly_split42_train42`，通信归因补齐。
6. P1 strict held-out cold-start split42（建议 NX0_v13a 出数后跑）。

**不再推荐 pipeline 串行流水**；附录保留可选 pipeline 命令供参考。

【命令目的】

两份 20260712 分析文档结论一致：当前不应新造 v14 或继续扫 μ/A 网格；最高信息价值是补合法 split42 的 v13a 主表，并修复 cold-start 的 A 泄漏。

本轮命令覆盖：

- P0：三 split benchmark fingerprint，防止 `/data2` 与 `/data` 同名异版再次混表。
- P0：`NX0_v13a_os_split42_train42`，完成 v13a 与 baseline 同协议的合法 3-split 主表。
- P2：`NX0_v13b_bonly_split42_train42`，补 9.49 MiB 通信归因点。
- P1：strict held-out cold-start split42，每域留 1 个 client 完全不参与 A/B 训练；报告域标签 oracle B pool 与 few-shot geometry route 两种变体。

【命令设置（gb）】

```text
代码目录: /data/yaominghao/gb/FedPLoRA
模型: /data/yaominghao/gb/models/SmolLM2-135M
benchmark: data/domain_benchmark_35c_dir05/seed_{42,43,44}
客户端: 35 = 7 domains × 5 clients
主训练轮次: rounds=1
local_epochs: 1
lr: 0.0002
LoRA: r=8, alpha=16, dropout=0.05
target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
formal eval: EVAL_MAX_BATCHES=0
smoke eval: EVAL_MAX_BATCHES=1 / train_max_steps_per_client=1
GPU: 全部物理 1 号卡（CUDA_VISIBLE_DEVICES=1）
conda: fedplora
```

【路径对照（minghao → gb）】


| 项          | order_20260712.md                         | order_gb_0712.md                                |
| ---------- | ----------------------------------------- | ----------------------------------------------- |
| conda      | `FedRepo2`                                | `fedplora`                                      |
| 代码         | `/data2/minghao/code/FedPLoRA-main`       | `/data/yaominghao/gb/FedPLoRA`                  |
| 模型         | `/data2/minghao/model/SmolLM2-135M`       | `/data/yaominghao/gb/models/SmolLM2-135M`       |
| 结果         | `/data2/minghao/result/FedPLoRA/`         | `/data/yaominghao/gb/result/FedPLoRA/`          |
| checkpoint | `/data2/minghao/model/trained_models_LW/` | `/data/yaominghao/gb/models/trained_models_LW/` |
| GPU        | `0 1 2 3` 四卡并行                            | 全部 `GPU=1`，后台 nohup                             |
| split-seed | `--split-seed` 由脚本自动切 benchmark           | 同左                                              |


【实验产物位置说明】

```text
run_logs:
/data/yaominghao/gb/result/FedPLoRA/${RUN_ID}/run_logs/test20260712_main_*.log

result_logs:
/data/yaominghao/gb/result/FedPLoRA/${RUN_ID}/result_logs/<method>/

client_states:
/data/yaominghao/gb/result/FedPLoRA/${RUN_ID}/result_files/client_states/<method>/

checkpoints:
/data/yaominghao/gb/models/trained_models_LW/${RUN_ID}/<method>_*

launch 日志（preflight + [one-exp] RUN_ROOT 行）:
/data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260712/<method>.launch.log
```

v13 期望 RUN_ID：

```text
smoke:       v13_20260712_smoke_seed42
heldout smoke: v13_20260712_smoke_seed42  （run-id-prefix=v13_20260712_smoke）
fingerprint: v13_20260712_fingerprint/fingerprints/seed_{42,43,44}.json
NX0:         v13_20260712_nx0_35c_dir05_r1_finaleval_seed42
held-out:    v13_20260712_strict_heldout_split42_seed42
```

---

【实验前置命令】

## 0.1 代码同步

gb 需单独 git pull / rsync 到 `/data/yaominghao/gb/FedPLoRA`，确保含：

```text
scripts/RunScripts/preflight_gb_20260712_v13.sh          # gb helper（本 order 专用）
scripts/RunScripts/run_20260712_one_experiment.sh
scripts/RunScripts/run_20260712_pipeline.sh
scripts/RunScripts/preflight_20260711_main_algorithm.sh
utilities/benchmark_fingerprint.py
methods/v13/
```

```bash
cd /data/yaominghao/gb/FedPLoRA && git pull
```

## 0.2 加载 preflight（每次开新 shell 先执行）

**source 一次 + `export GPU=1`，后续逐条粘贴 `run_v13_`* 即可后台运行，无需 wait**。单卡 GPU1 同时只跑一个实验，上一条结束或确认 pid 在跑后再贴下一条，避免 OOM。

```bash
exec bash
cd /data/yaominghao/gb/FedPLoRA
conda activate fedplora
source scripts/RunScripts/preflight_gb_20260712_v13.sh
export GPU=1
```

应看到：

```text
[preflight_gb_v13_12][ok] GPU default=1 LAUNCH_DIR=...
[preflight_gb_v13_12][ok] 用法: export GPU=1 && run_v13_nx0 <method> <agg> [--force_retrain ...]
```

可选自检（不跑实验）：

```bash
python -m py_compile methods/v13/*.py tasks/fed_train_sft.py utilities/benchmark_fingerprint.py
python -m py_compile scripts/Analysis/eval_personalized.py
```

## 0.3 运行函数说明


| 函数                      | 用途                      | 示例                                                                         |
| ----------------------- | ----------------------- | -------------------------------------------------------------------------- |
| `run_v13_smoke`         | v13a/v13b 1-round smoke | `run_v13_smoke smoke_v13a_os fedplora_v13a_os --force_retrain`             |
| `run_v13_heldout_smoke` | strict held-out smoke   | `run_v13_heldout_smoke`                                                    |
| `run_v13_fingerprint`   | 三 split 指纹审计（0 GPU）     | `run_v13_fingerprint`                                                      |
| `run_v13_nx0`           | NX0 SFT split42/train42 | `run_v13_nx0 NX0_v13a_os_split42_train42 fedplora_v13a_os --force_retrain` |
| `run_v13_heldout`       | strict held-out 正式 eval | `run_v13_heldout`                                                          |


每条命令会 echo `pid=` 和 `main=` 主日志路径。**两层日志**：launch 日志看 preflight；主日志看训练/评测进度。

路径自检（任意一条 launch 后）：

```bash
grep -E '\[one-exp\] (RUN_ID_PREFIX|RUN_ID|RUN_ROOT|log)=' "$RESULT_ROOT/manual_launch_logs_20260712/<method>.launch.log"
# 必须含 v13_20260712，禁止 v12_20260709_main
```

**若主日志目录未生成**：先看 launch 日志（nohup  stderr 也在这里）：

```bash
tail -n 80 $RESULT_ROOT/manual_launch_logs_20260712/NX0_v13a_os_split42_train42.launch.log
```

常见原因：`source preflight` 失败（v13 角色未同步）、benchmark 路径不对、conda 环境名错误。

若旧 pipeline / 误跑进程仍在，先停掉：

```bash
pkill -u "$USER" -f run_20260712_pipeline.sh || true
pkill -u "$USER" -f run_20260712_one_experiment.sh || true
pkill -u "$USER" -f run_20260711_one_experiment.sh || true
```

---

【实验运行命令】

以下命令均假设已执行 §0.2（`export GPU=1` 已生效）。行首**不要**写 `GPU=0`。

## 1. Smoke：正式 NX0 前单独运行

说明：三条 smoke 逐条粘贴（均 GPU1）。smoke 结果不能作为论文数值。

### 1.1 v13a smoke

```bash
run_v13_smoke smoke_v13a_os fedplora_v13a_os --force_retrain
```

### 1.2 v13b smoke

```bash
run_v13_smoke smoke_v13b_os_bonly fedplora_v13b_os_bonly --force_retrain
```

### 1.3 strict held-out eval smoke

```bash
run_v13_heldout_smoke
```

### 1.4 一键复制：smoke 全部三条

```bash
run_v13_smoke smoke_v13a_os fedplora_v13a_os --force_retrain
run_v13_smoke smoke_v13b_os_bonly fedplora_v13b_os_bonly --force_retrain
run_v13_heldout_smoke
```

smoke 主日志：

```text
/data/yaominghao/gb/result/FedPLoRA/v13_20260712_smoke_seed42/run_logs/test20260712_main_smoke_smoke_v13a_os_seed42.log
/data/yaominghao/gb/result/FedPLoRA/v13_20260712_smoke_seed42/run_logs/test20260712_main_smoke_smoke_v13b_os_bonly_seed42.log
/data/yaominghao/gb/result/FedPLoRA/v13_20260712_smoke_seed42/run_logs/test20260712_main_X2_strict_heldout_smoke_seed42_seed42.log
```

smoke 检查：

```bash
grep -E "Traceback|CUDA out of memory|error" \
  /data/yaominghao/gb/result/FedPLoRA/v13_20260712_smoke_seed42/run_logs/*.log || true
find /data/yaominghao/gb/result/FedPLoRA/v13_20260712_smoke_seed42/result_logs -name '*.json' | sort
grep "\[one-exp\] log=" /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260712/smoke_*.launch.log
grep "\[one-exp\] log=" /data/yaominghao/gb/result/FedPLoRA/manual_launch_logs_20260712/X2_strict_heldout_smoke_seed42.launch.log
```

---

## 2. P0 fingerprint：三 split 指纹审计

说明：0 GPU 命令，可与 smoke 同时后台跑；建议 preflight 通过后再跑。逐条粘贴即可。

```bash
run_v13_fingerprint
```

产物：

```text
/data/yaominghao/gb/result/FedPLoRA/v13_20260712_fingerprint/fingerprints/seed_42.json
/data/yaominghao/gb/result/FedPLoRA/v13_20260712_fingerprint/fingerprints/seed_43.json
/data/yaominghao/gb/result/FedPLoRA/v13_20260712_fingerprint/fingerprints/seed_44.json
/data/yaominghao/gb/result/FedPLoRA/v13_20260712_fingerprint/pipeline_logs/fingerprint_3splits.log
```

验收：

```bash
ls -la $RESULT_ROOT/v13_20260712_fingerprint/fingerprints/
tail -n 40 $RESULT_ROOT/v13_20260712_fingerprint/pipeline_logs/fingerprint_3splits.log
```

---

## 3. P0 NX0：v13a 合法 split42/train42

说明：最高优先级主表补齐。逐条粘贴，等上一条结束再跑下一条。

```bash
run_v13_nx0 NX0_v13a_os_split42_train42 fedplora_v13a_os --force_retrain
```

主日志：

```text
/data/yaominghao/gb/result/FedPLoRA/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/run_logs/test20260712_main_NX0_v13a_os_split42_train42_SmolLM2-135M_dir05_r1_e1_lr0.0002_seed42.log
```

验收：

```bash
find $RESULT_ROOT/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_v13a_os_split42_train42 -name '*.json'
grep -l "Traceback\|CUDA out of memory" \
  $RESULT_ROOT/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/run_logs/*.log || echo "NX0 v13a log OK"
```

---

## 4. P2 NX0-v13b：通信归因补齐

说明：可与 NX0_v13a 串行跑（gb 单卡）；资源够时等 v13a 结束再贴。

```bash
run_v13_nx0 NX0_v13b_bonly_split42_train42 fedplora_v13b_os_bonly --force_retrain
```

验收：

```bash
find $RESULT_ROOT/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_v13b_bonly_split42_train42 -name '*.json'
```

### 4.1 一键复制：NX0 两条

```bash
run_v13_nx0 NX0_v13a_os_split42_train42 fedplora_v13a_os --force_retrain
run_v13_nx0 NX0_v13b_bonly_split42_train42 fedplora_v13b_os_bonly --force_retrain
```

---

## 5. P1 strict held-out cold-start split42

说明：建议 **NX0_v13a 出数后**再跑；held-out 每域留 1 个 client 完全不参与 A/B pool。

```bash
run_v13_heldout
```

主日志与 JSON：

```text
/data/yaominghao/gb/result/FedPLoRA/v13_20260712_strict_heldout_split42_seed42/run_logs/test20260712_main_X2_strict_heldout_seed42_seed42.log
/data/yaominghao/gb/result/FedPLoRA/v13_20260712_strict_heldout_split42_seed42/result_logs/X2_strict_heldout_seed42_seed42.json
```

验收：

```bash
find $RESULT_ROOT/v13_20260712_strict_heldout_split42_seed42/result_logs -name '*.json' | sort
grep -l "Traceback\|CUDA out of memory" \
  $RESULT_ROOT/v13_20260712_strict_heldout_split42_seed42/run_logs/*.log || echo "held-out log OK"
```

---

# 推荐执行顺序（0705 风格，无 pipeline）

```text
0) git pull + source preflight_gb_20260712_v13.sh + export GPU=1
1) §1 smoke 三条（逐条粘贴）
2) §2 fingerprint（可与 smoke 并行，0 GPU）
3) 确认 launch 日志 log= 指向 v13_20260712_*
4) §3 NX0_v13a（必跑）
5) §4 NX0_v13b（可选，串行）
6) NX0_v13a json 齐全后 §5 strict held-out
```

一键验收：

```bash
find $RESULT_ROOT/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_* -name '*.json'
find $RESULT_ROOT/v13_20260712_strict_heldout_split42_seed42/result_logs -name '*.json'
ls $RESULT_ROOT/v13_20260712_fingerprint/fingerprints/seed_*.json
```

---

# 附录：可选 pipeline（不推荐，仅参考）

gb 单卡建议用手工 `run_v13_*`。若仍要用 pipeline，设 `MAX_PARALLEL=1`、`GPU_LIST=1`：

```bash
cd /data/yaominghao/gb/FedPLoRA
export PIPELINE_RUN_ID=v13_20260712_pipeline_$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULT_ROOT/${PIPELINE_RUN_ID}/pipeline_logs $RESULT_ROOT/${PIPELINE_RUN_ID}/pids

REQUIRE_SMOKE_OK=1 RUN_NX0_V13B=1 RUN_HELDOUT=1 \
GPU_LIST=1 MAX_PARALLEL=1 \
NX0_MIN_V13A_LOCAL=0.5980 NX0_MAIN_LOCAL_TARGET=0.6071 HELDOUT_MIN_COLDSTART_DELTA=0.01 \
PIPELINE_RUN_ID="$PIPELINE_RUN_ID" \
nohup bash scripts/RunScripts/run_20260712_pipeline.sh \
  > $RESULT_ROOT/${PIPELINE_RUN_ID}/pipeline_logs/pipeline.log 2>&1 &

echo $! > $RESULT_ROOT/${PIPELINE_RUN_ID}/pids/pipeline.nohup.pid
echo "[pipeline] log=$RESULT_ROOT/${PIPELINE_RUN_ID}/pipeline_logs/pipeline.log"
```

只跑 NX0_v13a、不跑 held-out/v13b：

```bash
REQUIRE_SMOKE_OK=1 RUN_NX0_V13B=0 RUN_HELDOUT=0 GPU_LIST=1 MAX_PARALLEL=1 \
PIPELINE_RUN_ID="$PIPELINE_RUN_ID" \
nohup bash scripts/RunScripts/run_20260712_pipeline.sh \
  > $RESULT_ROOT/${PIPELINE_RUN_ID}/pipeline_logs/pipeline.log 2>&1 &
```

---

# gb 运行提示

```text
1. 全部 GPU1：§0.2 里 export GPU=1 后，行首勿写 GPU=0。
2. 单卡同时只跑一个 GPU 实验；fingerprint 是 0 GPU，可与 smoke 并行。
3. v13 路径：launch 日志 [one-exp] RUN_ID_PREFIX=v13_20260712_* 且 log= 指向 v13 目录。
4. NX0 gate 参考：v13a Local >= 0.5980 再继续 held-out；强主表目标 0.6071 仅作 hint。
5. 杀误跑进程：pkill -u "$USER" -f "run_20260712_one_experiment.sh" ; pkill -u "$USER" -f "run_20260712_pipeline.sh" ; pkill -u "$USER" -f "fed_train_sft.py"
6. baseline / 0711 order 请分开 GPU 时段跑，避免混抢 GPU1。
7. 查看进度：tail -f <主日志路径>（run_v13_* echo 的 main= 行）
8. 查看后台任务：jobs -l 或 ps -u "$USER" -f | grep run_20260712_one_experiment
```

