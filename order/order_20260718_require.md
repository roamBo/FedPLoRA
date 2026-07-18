# FedPLoRA 20260718：顶会投稿实验完整性复核与待运行命令

######### FedPLoRA 投稿前实验补强命令-20260718 #########

【命令目的】

1. 完成 FlowerTune 5-fold leave-one-client-out：offset0 已完成，补 offset1–4 × seeds42/43/44。
2. 补 route-probe 1/2/5/10-shot 敏感性，明确信息预算。
3. 在 A100 9k canonical split 上重建并重跑 centralized-per-domain 参照界。
4. 补 A/B subspace 与 B-swap seeds43/44。
5. 补同一 v13a 下的 B-routing 核心消融。
6. 生成 cold-start paired-Δ/CI 与 router margin-performance 审计。
7. exact-v13 protocol gate 通过后，再条件执行 Qwen2.5-3B cold-start 与主表三方法。

【统一设置】

```text
服务器: 172.26.191.30 / minghao
代码: /data2/minghao/code/FedPLoRA-main
结果: /data2/minghao/result/FedPLoRA
135M: /data2/minghao/model/SmolLM2-135M
3B: /data2/minghao/model/Qwen2.5-3B
D1 canonical: /data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_{42,43,44}
FlowerTune: /data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_{42,43,44}
rounds=1, local_epochs=1, lr=2e-4
LoRA r=8, alpha=16, dropout=0.05
batch=2, max_seq_length=256, dtype=bfloat16
full eval: eval_max_batches=0
```

说明：Qwen2.5-3B 官方模型卡当前标注的是 **Qwen Research License**，不是 Claude 文档所写的 Apache-2.0；论文和 artifact 文档应按实际模型卡填写。模型卡：<https://huggingface.co/Qwen/Qwen2.5-3B>。

【产物位置】

```text
run_logs:    /data2/minghao/result/FedPLoRA/<RUN_ID>/run_logs/
result_logs: /data2/minghao/result/FedPLoRA/<RUN_ID>/result_logs/
result_files:/data2/minghao/result/FedPLoRA/<RUN_ID>/result_files/
audit:       /data2/minghao/result/FedPLoRA/audit_20260718/
```

本文件中直接启动的训练/评测命令通过 `/usr/bin/time -v` 记录 elapsed time 与 maximum RSS；GPU 型号/显存和软件版本由 R13 单独落盘。项目现有 `run_sft_smoke`/`run_sft_full` helper 会为每次调用建立独立 nohup、日志和产物，但 helper 本身尚未封装 `/usr/bin/time -v`；这些 run 的墙钟时间需从首尾时间戳或调度器补齐，不能在论文中误报为已由 `time -v` 审计。

---

# 第一部分：实验前置与 smoke

## 0. 服务器通用前置

每个新 shell 先执行：

```bash
exec bash
source /home/minghao/anaconda3/etc/profile.d/conda.sh
conda activate FedRepo2
export CODE_DIR=/data2/minghao/code/FedPLoRA-main
export RESULT_ROOT=/data2/minghao/result/FedPLoRA
export MODEL_ROOT=/data2/minghao/model/trained_models_LW
cd "$CODE_DIR"

python -m py_compile \
  tasks/fed_train_sft.py \
  scripts/Analysis/eval_personalized.py \
  scripts/Analysis/diag_subspace_AB.py \
  scripts/Analysis/diag_b_swap.py \
  scripts/Analysis/analyze_router_reliability.py
```

## 0.1 R13 计算环境审计（0 GPU）

```bash
mkdir -p /data2/minghao/result/FedPLoRA/audit_20260718/run_logs

nohup bash -s > /data2/minghao/result/FedPLoRA/audit_20260718/run_logs/compute_environment.log 2>&1 <<'BASH' &
set -euo pipefail
source /home/minghao/anaconda3/etc/profile.d/conda.sh
conda activate FedRepo2
cd /data2/minghao/code/FedPLoRA-main
date -Is
uname -a
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python --version
python - <<'PY'
import importlib
for name in ("torch", "transformers", "peft", "numpy", "scipy", "sklearn"):
    try:
        m = importlib.import_module(name)
        print(f"{name}={getattr(m, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{name}=MISSING ({e})")
PY
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
BASH
```

## 0.2 FlowerTune offset smoke（seed42/offset1）

该 smoke 只检查 held-out 选择、route probe、结果 JSON 和 fingerprint；不能进论文表。

```bash
mkdir -p /data2/minghao/result/FedPLoRA/flowertune_20260718_offset_smoke/pipeline_logs

PATH="/home/minghao/anaconda3/condabin:${PATH}" \
EXPECTED_NUM_CLIENTS=20 \
BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 \
BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" \
RUN_TAG_DATASET=flowertune20c_dir05 \
PIPELINE_EVAL_MAX_BATCHES=1 \
PIPELINE_ROUNDS=1 \
nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval \
  --method X2_flower_offset1_smoke_seed42 \
  --seed 42 --split-seed 42 \
  --run-id-prefix flowertune_20260718_offset_smoke \
  --gpu 0 \
  -- --held_out_clients auto_one_per_domain \
     --held_out_policy offset --held_out_offset 1 \
     --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local \
     --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart \
     --few_shot_caps 5 \
     --held_out_route_probe_samples 2 \
     --eval_on_local --cold_start --max_steps 1 --v11c_mu 0.4 \
  > /data2/minghao/result/FedPLoRA/flowertune_20260718_offset_smoke/pipeline_logs/X2_flower_offset1_smoke_seed42.launch.log 2>&1 &
```

Smoke 完成后检查：

```bash
SMOKE_JSON=/data2/minghao/result/FedPLoRA/flowertune_20260718_offset_smoke_seed42/result_logs/X2_flower_offset1_smoke_seed42_seed42.json
python - "$SMOKE_JSON" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
assert d["strict_held_out"]["selection_offset"] == 1
assert len(d["strict_held_out"]["held_out_clients"]) == 4
assert d["config"]["eval_max_batches"] == 1
assert d["config"]["max_steps"] == 1
print("[smoke][ok]", p, d["protocol_tag"])
PY
```

## 0.3 Qwen2.5-3B 模型下载/检查（条件执行）

当前模型不存在时再下载；已存在则只做 config/tokenizer 检查。

```bash
if [ ! -f /data2/minghao/model/Qwen2.5-3B/config.json ]; then
  command -v modelscope >/dev/null 2>&1 || pip install -U modelscope
  modelscope download --model Qwen/Qwen2.5-3B --local_dir /data2/minghao/model/Qwen2.5-3B
fi

python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
p = "/data2/minghao/model/Qwen2.5-3B"
c = AutoConfig.from_pretrained(p)
t = AutoTokenizer.from_pretrained(p, use_fast=False)
print("[3b][ok]", c.model_type, getattr(c, "num_hidden_layers", None), len(t))
PY
```

## 0.4 3B cold-start smoke（条件执行）

注意：当前代码下该 smoke 仍是 full-A/oracle-domain 与 supervised route-probe 审计；只有 R0 exact evaluator 接入后才能作为 FedPLoRA-OS exact 证据。

```bash
mkdir -p /data2/minghao/result/FedPLoRA/qwen3b_20260718_coldstart_smoke/pipeline_logs

PATH="/home/minghao/anaconda3/condabin:${PATH}" \
MODEL_PATH=/data2/minghao/model/Qwen2.5-3B \
RUN_TAG_MODEL=Qwen2.5-3B \
EXPECTED_NUM_CLIENTS=20 \
BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 \
BENCHMARK_REQUIRED_SPLIT_SEEDS="42" \
RUN_TAG_DATASET=flowertune20c_dir05 \
PIPELINE_EVAL_MAX_BATCHES=1 \
PIPELINE_ROUNDS=1 \
nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh \
  --kind personalized_eval \
  --method X2_qwen3b_flower_coldstart_smoke_seed42 \
  --seed 42 --split-seed 42 \
  --run-id-prefix qwen3b_20260718_coldstart_smoke \
  --gpu 0 \
  -- --held_out_clients auto_one_per_domain \
     --held_out_policy first --held_out_offset 0 \
     --schemes base,global,coldstart,coldstart_geom \
     --few_shot_caps 5 \
     --held_out_route_probe_samples 2 \
     --eval_on_local --cold_start --max_steps 1 \
  > /data2/minghao/result/FedPLoRA/qwen3b_20260718_coldstart_smoke/pipeline_logs/X2_qwen3b_flower_coldstart_smoke_seed42.launch.log 2>&1 &
```

## 0.5 3B Normal/FedALT/v13a smoke（条件执行）

Baseline smoke：

```bash
export MODEL_PATH=/data2/minghao/model/Qwen2.5-3B
export RUN_TAG_MODEL=Qwen2.5-3B
export BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_42
export BENCHMARK_REQUIRED_SPLIT_SEEDS="42"
export EXPECTED_NUM_CLIENTS=35
source scripts/RunScripts/preflight_20260709_baseline.sh

export SMOKE_RUN_ID=qwen3b_20260718_sft_smoke_seed42
export SMOKE_ROOT="$RESULT_ROOT/$SMOKE_RUN_ID"
export SMOKE_TRAINED_MODELS_ROOT="$MODEL_ROOT/$SMOKE_RUN_ID"
mkdir -p "$SMOKE_ROOT/run_logs" "$SMOKE_ROOT/result_logs" "$SMOKE_ROOT/result_files/client_states" "$SMOKE_TRAINED_MODELS_ROOT"

GPU=0 run_sft_smoke smoke_qwen3b_normal normal --force_retrain
GPU=1 run_sft_smoke smoke_qwen3b_fedalt fedalt --force_retrain
```

主方法 smoke：

```bash
export MODEL_PATH=/data2/minghao/model/Qwen2.5-3B
export RUN_TAG_MODEL=Qwen2.5-3B
export BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_42
export BENCHMARK_REQUIRED_SPLIT_SEEDS="42"
export EXPECTED_NUM_CLIENTS=35
source scripts/RunScripts/preflight_20260709_main_algorithm.sh

export SMOKE_RUN_ID=qwen3b_20260718_sft_smoke_seed42
export SMOKE_ROOT="$RESULT_ROOT/$SMOKE_RUN_ID"
export SMOKE_TRAINED_MODELS_ROOT="$MODEL_ROOT/$SMOKE_RUN_ID"
mkdir -p "$SMOKE_ROOT/run_logs" "$SMOKE_ROOT/result_logs" "$SMOKE_ROOT/result_files/client_states" "$SMOKE_TRAINED_MODELS_ROOT"

GPU=2 run_sft_smoke smoke_v13a_qwen3b fedplora_v13a_os --force_retrain
```

三个 smoke 均出现最终 JSON、无 Traceback 后，才运行第 6 节 3B 正式实验。

## 0.6 R16 B-routing 消融 smoke

主方法 `auto` 已有大量 smoke/formal；这里只检查 `global` 与 `domain` 两条 mode 分支。

```bash
export MODEL_PATH=/data2/minghao/model/SmolLM2-135M
export BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_42
export BENCHMARK_REQUIRED_SPLIT_SEEDS="42"
export EXPECTED_NUM_CLIENTS=35
source scripts/RunScripts/preflight_20260709_main_algorithm.sh

export SMOKE_RUN_ID=v13_20260718_route_ablation_smoke_seed42
export SMOKE_ROOT="$RESULT_ROOT/$SMOKE_RUN_ID"
export SMOKE_TRAINED_MODELS_ROOT="$MODEL_ROOT/$SMOKE_RUN_ID"
mkdir -p "$SMOKE_ROOT/run_logs" "$SMOKE_ROOT/result_logs" "$SMOKE_ROOT/result_files/client_states" "$SMOKE_TRAINED_MODELS_ROOT"

GPU=0 run_sft_smoke smoke_v13a_route_global fedplora_v13a_os --expert_cluster_mode global --force_retrain
GPU=1 run_sft_smoke smoke_v13a_route_oracle_domain fedplora_v13a_os --expert_cluster_mode domain --force_retrain
```

---

# 第二部分：P0/P1 正式补强实验

## 1. R1：FlowerTune 完整 leave-one-client-out

### 1.1 设计说明

FlowerTune 每域 5 clients。offset0 已跑；补 offset1–4 后，每个 split 的 20 个 clients 都恰好被 held out 一次。相比只补 offset1/2，这个设计能直接回答 cherry-picking，并为 paired client-level CI 提供完整覆盖。

运行量：`4 offsets × 3 seeds = 12` 个独立 nohup、独立 PID、独立结果目录。以下命令可按 GPU 数并行；同一 GPU 上不要同时启动两个。

### 1.2 offset1 × seeds42/43/44

```bash
mkdir -p /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset1_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260718_loco_offset1 --gpu 0 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset1_seed42.launch.log 2>&1 &

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset1_seed43 --seed 43 --split-seed 43 --run-id-prefix flowertune_20260718_loco_offset1 --gpu 1 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset1_seed43.launch.log 2>&1 &

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset1_seed44 --seed 44 --split-seed 44 --run-id-prefix flowertune_20260718_loco_offset1 --gpu 2 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 1 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset1_seed44.launch.log 2>&1 &
```

### 1.3 offset2 × seeds42/43/44

```bash
PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset2_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260718_loco_offset2 --gpu 0 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset2_seed42.launch.log 2>&1 &

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset2_seed43 --seed 43 --split-seed 43 --run-id-prefix flowertune_20260718_loco_offset2 --gpu 1 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset2_seed43.launch.log 2>&1 &

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset2_seed44 --seed 44 --split-seed 44 --run-id-prefix flowertune_20260718_loco_offset2 --gpu 2 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 2 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset2_seed44.launch.log 2>&1 &
```

### 1.4 offset3 × seeds42/43/44

```bash
PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset3_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260718_loco_offset3 --gpu 0 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset3_seed42.launch.log 2>&1 &

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset3_seed43 --seed 43 --split-seed 43 --run-id-prefix flowertune_20260718_loco_offset3 --gpu 1 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset3_seed43.launch.log 2>&1 &

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset3_seed44 --seed 44 --split-seed 44 --run-id-prefix flowertune_20260718_loco_offset3 --gpu 2 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 3 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset3_seed44.launch.log 2>&1 &
```

### 1.5 offset4 × seeds42/43/44

```bash
PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset4_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260718_loco_offset4 --gpu 0 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset4_seed42.launch.log 2>&1 &

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset4_seed43 --seed 43 --split-seed 43 --run-id-prefix flowertune_20260718_loco_offset4 --gpu 1 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset4_seed43.launch.log 2>&1 &

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42 43 44" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_loco_offset4_seed44 --seed 44 --split-seed 44 --run-id-prefix flowertune_20260718_loco_offset4 --gpu 2 -- --held_out_clients auto_one_per_domain --held_out_policy offset --held_out_offset 4 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart,select_without_local --select_candidates base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/flowertune_20260718_loco_launcher/pipeline_logs/X2_flower_loco_offset4_seed44.launch.log 2>&1 &
```

资源不足时的最小队列：先跑 offset1/2 ×3；但论文若强调 held-out 选择无关，建议补完 offset3/4，形成真正 5-fold client coverage。

## 2. R0b：route-probe 样本数敏感性

offset0/seed42 的 10-shot 已完成，只补 1/2/5-shot。这里的样本是带 SFT response 的本地训练样本，图表和正文必须写 `supervised probe examples`。

```bash
mkdir -p /data2/minghao/result/FedPLoRA/flowertune_20260718_probe_launcher/pipeline_logs

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe1_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260718_probe1 --gpu 0 -- --held_out_clients auto_one_per_domain --held_out_policy first --held_out_offset 0 --schemes base,global,coldstart,coldstart_geom --few_shot_caps 1 --held_out_route_probe_samples 1 --eval_on_local --cold_start > /data2/minghao/result/FedPLoRA/flowertune_20260718_probe_launcher/pipeline_logs/X2_flower_probe1_seed42.launch.log 2>&1 &

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe2_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260718_probe2 --gpu 1 -- --held_out_clients auto_one_per_domain --held_out_policy first --held_out_offset 0 --schemes base,global,coldstart,coldstart_geom --few_shot_caps 2 --held_out_route_probe_samples 2 --eval_on_local --cold_start > /data2/minghao/result/FedPLoRA/flowertune_20260718_probe_launcher/pipeline_logs/X2_flower_probe2_seed42.launch.log 2>&1 &

PATH="/home/minghao/anaconda3/condabin:${PATH}" EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_flower_probe5_seed42 --seed 42 --split-seed 42 --run-id-prefix flowertune_20260718_probe5 --gpu 2 -- --held_out_clients auto_one_per_domain --held_out_policy first --held_out_offset 0 --schemes base,global,coldstart,coldstart_geom --few_shot_caps 5 --held_out_route_probe_samples 5 --eval_on_local --cold_start > /data2/minghao/result/FedPLoRA/flowertune_20260718_probe_launcher/pipeline_logs/X2_flower_probe5_seed42.launch.log 2>&1 &
```

## 3. R5：A100 9k canonical centralized-per-domain ×3

### 3.1 从 35c 原样池化为 7c（0 GPU）

该命令不重新随机切分：保留每个 A100 9k split 的全部 train/val/test rows 与 domain test，只把同域 5 clients 的 `client_id` 合并为一个，因此 centralized reference 与主表严格同源。

```bash
mkdir -p /data2/minghao/result/FedPLoRA/audit_20260718/run_logs

nohup /usr/bin/time -v bash -s > /data2/minghao/result/FedPLoRA/audit_20260718/run_logs/build_a1009k_7c_3seeds.log 2>&1 <<'BASH' &
set -euo pipefail
source /home/minghao/anaconda3/etc/profile.d/conda.sh
conda activate FedRepo2
cd /data2/minghao/code/FedPLoRA-main
python - <<'PY'
import json
import os
import pathlib
import shutil
import tempfile

src_root = pathlib.Path("/data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05")
dst_root = pathlib.Path("/data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_7c_dir05")
dst_root.mkdir(parents=True, exist_ok=True)

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

for seed in (42, 43, 44):
    src = src_root / f"seed_{seed}"
    dst = dst_root / f"seed_{seed}"
    if not (src / "clients.json").is_file():
        raise SystemExit(f"[pool7c][error] missing {src / 'clients.json'}")
    if (dst / "clients.json").is_file():
        clients = json.loads((dst / "clients.json").read_text())
        if len(clients) != 7:
            raise SystemExit(f"[pool7c][error] existing destination is invalid: {dst}")
        print(f"[pool7c][skip] {dst}")
        continue

    src_clients = json.loads((src / "clients.json").read_text())
    domains = []
    for row in src_clients:
        dom = str(row["domain"])
        if dom not in domains:
            domains.append(dom)
    if len(src_clients) != 35 or len(domains) != 7:
        raise SystemExit(f"[pool7c][error] expected 35 clients/7 domains at {src}")
    domain_to_cid = {dom: i for i, dom in enumerate(domains)}

    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"seed_{seed}.tmp.", dir=dst_root))
    counts = {dom: {"n_train": 0, "n_val": 0, "n_local_test": 0} for dom in domains}
    for filename, count_key in (("train.jsonl", "n_train"), ("val.jsonl", "n_val"), ("test_local.jsonl", "n_local_test")):
        with (tmp / filename).open("w", encoding="utf-8") as out:
            for row in read_jsonl(src / filename):
                dom = str(row["domain"])
                row["client_id"] = domain_to_cid[dom]
                counts[dom][count_key] += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    for filename in ("test_domain.jsonl", "test_global.jsonl"):
        shutil.copy2(src / filename, tmp / filename)

    clients = []
    for dom in domains:
        clients.append({"client_id": domain_to_cid[dom], "domain": dom, **counts[dom]})
    (tmp / "clients.json").write_text(json.dumps(clients, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = json.loads((src / "domain_stats.json").read_text())
    for dom in stats:
        stats[dom]["n_clients"] = 1
    (tmp / "domain_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, dst)
    print(f"[pool7c][ok] {src} -> {dst}")
PY

for SEED in 42 43 44; do
  python utilities/benchmark_fingerprint.py \
    "/data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_7c_dir05/seed_${SEED}" \
    --output "/data2/minghao/result/FedPLoRA/audit_20260718/a1009k_7c_seed${SEED}_fingerprint.json"
done
BASH
```

### 3.2 centralized smoke（seed42）

```bash
export CENTRAL_SMOKE_ROOT=/data2/minghao/result/FedPLoRA/ref_20260718_a1009k_centralized_smoke_seed42
mkdir -p "$CENTRAL_SMOKE_ROOT/run_logs" "$CENTRAL_SMOKE_ROOT/result_logs"

CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time -v python -u scripts/Analysis/eval_personalized.py \
  --model /data2/minghao/model/SmolLM2-135M \
  --benchmark_dir /data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_7c_dir05/seed_42 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --torch_dtype bfloat16 --batch_size 2 --max_seq_length 256 \
  --lr 0.0002 --local_epochs 1 --max_steps 1 --eval_max_batches 1 \
  --seed 42 --schemes local --eval_on_local \
  --out "$CENTRAL_SMOKE_ROOT/result_logs/N6_a1009k_centralized_smoke_seed42.json" \
  > "$CENTRAL_SMOKE_ROOT/run_logs/N6_a1009k_centralized_smoke_seed42.log" 2>&1 &
```

### 3.3 centralized formal seeds42/43/44

```bash
export CENTRAL_ROOT42=/data2/minghao/result/FedPLoRA/ref_20260718_a1009k_centralized_7c_seed42
mkdir -p "$CENTRAL_ROOT42/run_logs" "$CENTRAL_ROOT42/result_logs"
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time -v python -u scripts/Analysis/eval_personalized.py --model /data2/minghao/model/SmolLM2-135M --benchmark_dir /data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_7c_dir05/seed_42 --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --torch_dtype bfloat16 --batch_size 2 --max_seq_length 256 --lr 0.0002 --local_epochs 1 --eval_max_batches 0 --seed 42 --schemes local --eval_on_local --out "$CENTRAL_ROOT42/result_logs/N6_a1009k_centralized_7c_seed42.json" > "$CENTRAL_ROOT42/run_logs/N6_a1009k_centralized_7c_seed42.log" 2>&1 &

export CENTRAL_ROOT43=/data2/minghao/result/FedPLoRA/ref_20260718_a1009k_centralized_7c_seed43
mkdir -p "$CENTRAL_ROOT43/run_logs" "$CENTRAL_ROOT43/result_logs"
CUDA_VISIBLE_DEVICES=1 nohup /usr/bin/time -v python -u scripts/Analysis/eval_personalized.py --model /data2/minghao/model/SmolLM2-135M --benchmark_dir /data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_7c_dir05/seed_43 --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --torch_dtype bfloat16 --batch_size 2 --max_seq_length 256 --lr 0.0002 --local_epochs 1 --eval_max_batches 0 --seed 43 --schemes local --eval_on_local --out "$CENTRAL_ROOT43/result_logs/N6_a1009k_centralized_7c_seed43.json" > "$CENTRAL_ROOT43/run_logs/N6_a1009k_centralized_7c_seed43.log" 2>&1 &

export CENTRAL_ROOT44=/data2/minghao/result/FedPLoRA/ref_20260718_a1009k_centralized_7c_seed44
mkdir -p "$CENTRAL_ROOT44/run_logs" "$CENTRAL_ROOT44/result_logs"
CUDA_VISIBLE_DEVICES=2 nohup /usr/bin/time -v python -u scripts/Analysis/eval_personalized.py --model /data2/minghao/model/SmolLM2-135M --benchmark_dir /data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_7c_dir05/seed_44 --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --torch_dtype bfloat16 --batch_size 2 --max_seq_length 256 --lr 0.0002 --local_epochs 1 --eval_max_batches 0 --seed 44 --schemes local --eval_on_local --out "$CENTRAL_ROOT44/result_logs/N6_a1009k_centralized_7c_seed44.json" > "$CENTRAL_ROOT44/run_logs/N6_a1009k_centralized_7c_seed44.log" 2>&1 &
```

## 4. R3：Motivation 诊断 seed43/44

这些命令会训练每个 split 的 35 个 client LoRA。每条命令需要 GPU；`diag_subspace_AB` 与 `diag_b_swap` 不要在同一张卡同时跑。

### 4.1 A/B subspace seed43/44

```bash
mkdir -p /data2/minghao/result/FedPLoRA/audit_20260718/analysis /data2/minghao/result/FedPLoRA/audit_20260718/run_logs

CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time -v python -u scripts/Analysis/diag_subspace_AB.py --model /data2/minghao/model/SmolLM2-135M --benchmark_dir /data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_43 --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --torch_dtype bfloat16 --batch_size 2 --max_seq_length 256 --lr 0.0002 --local_epochs 1 --max_steps 0 --seed 43 --n_null 200 --out /data2/minghao/result/FedPLoRA/audit_20260718/analysis/diag_subspace_AB_seed43.json --save_figs > /data2/minghao/result/FedPLoRA/audit_20260718/run_logs/diag_subspace_AB_seed43.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup /usr/bin/time -v python -u scripts/Analysis/diag_subspace_AB.py --model /data2/minghao/model/SmolLM2-135M --benchmark_dir /data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_44 --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --torch_dtype bfloat16 --batch_size 2 --max_seq_length 256 --lr 0.0002 --local_epochs 1 --max_steps 0 --seed 44 --n_null 200 --out /data2/minghao/result/FedPLoRA/audit_20260718/analysis/diag_subspace_AB_seed44.json --save_figs > /data2/minghao/result/FedPLoRA/audit_20260718/run_logs/diag_subspace_AB_seed44.log 2>&1 &
```

### 4.2 B-swap seed43/44

```bash
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time -v python -u scripts/Analysis/diag_b_swap.py --model /data2/minghao/model/SmolLM2-135M --benchmark_dir /data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_43 --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --torch_dtype bfloat16 --batch_size 2 --eval_batch_size 2 --max_seq_length 256 --lr 0.0002 --local_epochs 1 --max_steps 0 --eval_max_batches 20 --n_peers 4 --n_cross 2 --seed 43 --out /data2/minghao/result/FedPLoRA/audit_20260718/analysis/diag_b_swap_seed43.json > /data2/minghao/result/FedPLoRA/audit_20260718/run_logs/diag_b_swap_seed43.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup /usr/bin/time -v python -u scripts/Analysis/diag_b_swap.py --model /data2/minghao/model/SmolLM2-135M --benchmark_dir /data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_44 --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --torch_dtype bfloat16 --batch_size 2 --eval_batch_size 2 --max_seq_length 256 --lr 0.0002 --local_epochs 1 --max_steps 0 --eval_max_batches 20 --n_peers 4 --n_cross 2 --seed 44 --out /data2/minghao/result/FedPLoRA/audit_20260718/analysis/diag_b_swap_seed44.json > /data2/minghao/result/FedPLoRA/audit_20260718/run_logs/diag_b_swap_seed44.log 2>&1 &
```

注意：即使 seed43/44 完成，当前脚本仍不会产出 Fig.2(a) 的 35×35 pairwise matrix。要画该热力图，必须先让脚本把 A/B pairwise angle/similarity matrix 和 client-domain order 写入 NPZ/JSON；这属于 R0c 代码任务，不得把现有 histogram 冒充热力图。

## 5. R16：B-routing 核心归因消融

主方法 `auto` 已有正式结果。新增两端点：

- `global`：所有 client 的 B 进入一个池，保留 v13a A-sketch，其余配置不变；这是 `w/o B-routing`。
- `domain`：使用真实 domain 分池，是 oracle upper bound；只能进消融/诊断，不能作为 label-free 方法结果。

### 5.1 D1 A100 9k，seed42

```bash
export MODEL_PATH=/data2/minghao/model/SmolLM2-135M
export BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_42
export BENCHMARK_REQUIRED_SPLIT_SEEDS="42"
export EXPECTED_NUM_CLIENTS=35
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export ROUNDS=1 LOCAL_EPOCHS=1 EVAL_MAX_BATCHES=0 RUN_TAG_DATASET=a1009k_35c_dir05
export RUN_ID_PREFIX=v13_20260718_route_ablation_d1
set_run_paths 42

GPU=0 run_sft_full N7_ours_v13a_route_global_d1 fedplora_v13a_os --expert_cluster_mode global --force_retrain
GPU=1 run_sft_full N7_ours_v13a_route_oracle_domain_d1 fedplora_v13a_os --expert_cluster_mode domain --force_retrain
```

### 5.2 FlowerTune，seed42

```bash
export MODEL_PATH=/data2/minghao/model/SmolLM2-135M
export BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42
export BENCHMARK_REQUIRED_SPLIT_SEEDS="42"
export EXPECTED_NUM_CLIENTS=20
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export ROUNDS=1 LOCAL_EPOCHS=1 EVAL_MAX_BATCHES=0 RUN_TAG_DATASET=flowertune20c_dir05
export RUN_ID_PREFIX=v13_20260718_route_ablation_flower
set_run_paths 42

GPU=2 run_sft_full N7_ours_v13a_route_global_flower fedplora_v13a_os --expert_cluster_mode global --force_retrain
GPU=3 run_sft_full N7_ours_v13a_route_oracle_domain_flower fedplora_v13a_os --expert_cluster_mode domain --force_retrain
```

判读顺序：`global` 回答 route B 是否必要；`domain` 与 `auto` 的差距回答无标签聚类距离 oracle 还有多远。若 global≈auto，则“route B 提升性能”的主张不成立，只能保留几何/冷启动机制解释。

## 6. R9/R10：Qwen2.5-3B 条件实验

执行门槛：

1. 第 0.3–0.5 节 smoke 全通过；
2. R0 exact-v13 protocol 已实现，或论文明确把 R9 仅称作 full-A/domain-metadata mechanism probe；
3. 不抢占第 1 节 FlowerTune LOCO。

### 6.1 R9 FlowerTune cold-start @3B，seed42

```bash
mkdir -p /data2/minghao/result/FedPLoRA/qwen3b_20260718_flower_coldstart_launcher/pipeline_logs

PATH="/home/minghao/anaconda3/condabin:${PATH}" MODEL_PATH=/data2/minghao/model/Qwen2.5-3B RUN_TAG_MODEL=Qwen2.5-3B EXPECTED_NUM_CLIENTS=20 BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/domain_benchmark_flowertune_mixed_20c_dir05/seed_42 BENCHMARK_REQUIRED_SPLIT_SEEDS="42" RUN_TAG_DATASET=flowertune20c_dir05 PIPELINE_EVAL_MAX_BATCHES=0 PIPELINE_ROUNDS=1 nohup /usr/bin/time -v bash scripts/RunScripts/run_20260713_one_experiment.sh --kind personalized_eval --method X2_qwen3b_flower_coldstart_seed42 --seed 42 --split-seed 42 --run-id-prefix qwen3b_20260718_flower_coldstart --gpu 0 -- --held_out_clients auto_one_per_domain --held_out_policy first --held_out_offset 0 --schemes base,global,coldstart,coldstart_geom,v11c_coldstart --few_shot_caps 5,10 --held_out_route_probe_samples 10 --eval_on_local --cold_start --v11c_mu 0.4 > /data2/minghao/result/FedPLoRA/qwen3b_20260718_flower_coldstart_launcher/pipeline_logs/X2_qwen3b_flower_coldstart_seed42.launch.log 2>&1 &
```

### 6.2 R10 D1 @3B：Normal/FedALT

```bash
export MODEL_PATH=/data2/minghao/model/Qwen2.5-3B
export RUN_TAG_MODEL=Qwen2.5-3B
export BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_42
export BENCHMARK_REQUIRED_SPLIT_SEEDS="42"
export EXPECTED_NUM_CLIENTS=35
source scripts/RunScripts/preflight_20260709_baseline.sh
export ROUNDS=1 LOCAL_EPOCHS=1 EVAL_MAX_BATCHES=0 RUN_TAG_DATASET=a1009k_35c_dir05
export RUN_ID_PREFIX=qwen3b_20260718_d1_baseline_r1_finaleval
set_run_paths 42

GPU=0 run_sft_full N7_baseline_qwen3b_normal normal --force_retrain
GPU=1 run_sft_full N7_baseline_qwen3b_fedalt fedalt --force_retrain
```

### 6.3 R10 D1 @3B：FedPLoRA-OS/v13a

```bash
export MODEL_PATH=/data2/minghao/model/Qwen2.5-3B
export RUN_TAG_MODEL=Qwen2.5-3B
export BENCHMARK_DIR_MAIN=/data2/minghao/code/FedPLoRA-main/data/A100_domain_benchmark_35c_dir05/seed_42
export BENCHMARK_REQUIRED_SPLIT_SEEDS="42"
export EXPECTED_NUM_CLIENTS=35
source scripts/RunScripts/preflight_20260709_main_algorithm.sh
export ROUNDS=1 LOCAL_EPOCHS=1 EVAL_MAX_BATCHES=0 RUN_TAG_DATASET=a1009k_35c_dir05
export RUN_ID_PREFIX=qwen3b_20260718_d1_ours_r1_finaleval
set_run_paths 42

GPU=2 run_sft_full N7_ours_qwen3b_v13a_os fedplora_v13a_os --force_retrain
```

`run_sft_full` 内部为每个调用分别启动一个 nohup；三方法各自有独立 PID、日志、checkpoint 与结果目录。

---

# 第三部分：0-GPU 汇总、统计与结果检查

## 7. R7/R12：cold-start margin、paired Δ 与 95% bootstrap CI

在 offset1–4 完成后运行。该命令同时读取既有 offset0 与新 offset1–4；输出逐 client 记录、route match、coldstart/global paired delta 和分层 bootstrap CI。

```bash
mkdir -p /data2/minghao/result/FedPLoRA/audit_20260718/analysis /data2/minghao/result/FedPLoRA/audit_20260718/run_logs

nohup /usr/bin/time -v bash -s > /data2/minghao/result/FedPLoRA/audit_20260718/run_logs/flower_loco_paired_stats.log 2>&1 <<'BASH' &
set -euo pipefail
source /home/minghao/anaconda3/etc/profile.d/conda.sh
conda activate FedRepo2
python - <<'PY'
import csv
import json
import pathlib
import random
import statistics

root = pathlib.Path("/data2/minghao/result/FedPLoRA")
out_root = root / "audit_20260718" / "analysis"
out_root.mkdir(parents=True, exist_ok=True)

paths = []
paths += sorted(root.glob("flowertune_20260717_strict_heldout_seed*/result_logs/*.json"))
for off in (1, 2, 3, 4):
    paths += sorted(root.glob(f"flowertune_20260718_loco_offset{off}_seed*/result_logs/*.json"))
if len(paths) != 15:
    raise SystemExit(f"[stats][error] expected 15 offset×seed JSONs (offset0 existing + offset1-4), found {len(paths)}")

rows = []
for p in paths:
    d = json.loads(p.read_text())
    cfg = d["config"]
    strict = d["strict_held_out"]
    seed = int(cfg["seed"])
    offset = int(strict["selection_offset"])
    global_acc = d["results"]["global"]["per_client_acc"]
    cold_acc = d["results"]["coldstart"]["per_client_acc"]
    geom = d["results"].get("coldstart_geom", {})
    geom_acc = geom.get("per_client_acc", {})
    margins = geom.get("geom_route_margin_by_client", {})
    matches = geom.get("geom_route_oracle_match_by_client", {})
    domains = strict["held_out_domains"]
    for cid, g in global_acc.items():
        rows.append({
            "seed": seed,
            "offset": offset,
            "client_id": int(cid),
            "domain": domains[str(cid)],
            "global_acc": float(g),
            "coldstart_acc": float(cold_acc[cid]),
            "coldstart_geom_acc": float(geom_acc[cid]),
            "delta_oracle_vs_global": float(cold_acc[cid]) - float(g),
            "delta_geom_vs_global": float(geom_acc[cid]) - float(g),
            "route_margin": None if margins.get(cid) is None else float(margins[cid]),
            "route_match": None if cid not in matches else bool(matches[cid]),
        })

csv_path = out_root / "flower_loco_client_paired_delta.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

# Hierarchical bootstrap: resample split seed, then offset, then domain/client.
rng = random.Random(20260718)
by_seed = {}
for row in rows:
    by_seed.setdefault(row["seed"], []).append(row)
seeds = sorted(by_seed)
boot_oracle, boot_geom = [], []
for _ in range(20000):
    sampled = []
    for seed in [rng.choice(seeds) for _ in seeds]:
        seed_rows = by_seed[seed]
        offsets = sorted({r["offset"] for r in seed_rows})
        for off in [rng.choice(offsets) for _ in offsets]:
            cell = [r for r in seed_rows if r["offset"] == off]
            sampled.extend(rng.choice(cell) for _ in cell)
    boot_oracle.append(statistics.fmean(r["delta_oracle_vs_global"] for r in sampled))
    boot_geom.append(statistics.fmean(r["delta_geom_vs_global"] for r in sampled))

def ci(xs):
    xs = sorted(xs)
    return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs))]]

report = {
    "protocol_warning": "coldstart=oracle/domain-metadata full-A; coldstart_geom=supervised route probe full-A; neither is exact v13a A-sketch in current evaluator",
    "n_json": len(paths),
    "n_client_evaluations": len(rows),
    "oracle_delta_mean": statistics.fmean(r["delta_oracle_vs_global"] for r in rows),
    "oracle_delta_hier_bootstrap_95ci": ci(boot_oracle),
    "geom_delta_mean": statistics.fmean(r["delta_geom_vs_global"] for r in rows),
    "geom_delta_hier_bootstrap_95ci": ci(boot_geom),
    "route_match_rate": statistics.fmean(float(r["route_match"]) for r in rows if r["route_match"] is not None),
    "route_margin_mean": statistics.fmean(r["route_margin"] for r in rows if r["route_margin"] is not None),
    "source_jsons": [str(p) for p in paths],
}
(out_root / "flower_loco_paired_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("[stats][ok]", csv_path)
PY
BASH
```

说明：CI 的统计单位和层级必须在论文 caption/appendix 中写清；不要把 60 个 client evaluations 当成 60 次完全独立训练。若只完成 offset1/2，应修改脚本预期数量，并把结果称作 partial offset robustness，不能称 full LOCO。

## 8. JSON 与失败日志总检查

```bash
python - <<'PY'
import json
import pathlib

root = pathlib.Path("/data2/minghao/result/FedPLoRA")
groups = {
    "flower_loco_new": list(root.glob("flowertune_20260718_loco_offset*_seed*/result_logs/*.json")),
    "centralized_9k": list(root.glob("ref_20260718_a1009k_centralized_7c_seed*/result_logs/*.json")),
    "diag_new": list((root / "audit_20260718" / "analysis").glob("diag_*seed4[34].json")),
    "route_ablation": list(root.glob("v13_20260718_route_ablation_*_seed42/result_logs/**/*.json")),
    "qwen3b": list(root.glob("qwen3b_20260718_*/**/*.json")),
}
for name, paths in groups.items():
    bad = []
    for p in paths:
        try:
            json.loads(p.read_text())
        except Exception as e:
            bad.append((str(p), repr(e)))
    print(f"[check] {name}: json={len(paths)} bad={len(bad)}")
    for row in bad:
        print("  ", row)

assert len(groups["flower_loco_new"]) in (6, 12), "FlowerTune should be partial offset1/2=6 or full offset1-4=12"
assert len(groups["centralized_9k"]) == 3, "canonical centralized must have seed42/43/44"
assert len(groups["diag_new"]) == 4, "expected subspace+B-swap for seed43/44"
assert len(groups["route_ablation"]) == 4, "expected global/domain routing ablations on D1 and FlowerTune"
PY

rg -n -i 'Traceback|CUDA out of memory|ModuleNotFoundError|FileNotFoundError|\[.*error\]|nan' \
  /data2/minghao/result/FedPLoRA/flowertune_20260718_* \
  /data2/minghao/result/FedPLoRA/ref_20260718_* \
  /data2/minghao/result/FedPLoRA/audit_20260718/run_logs \
  /data2/minghao/result/FedPLoRA/qwen3b_20260718_* 2>/dev/null || true
```

---

# 第四部分：BLOCKED——需先落代码，当前不能靠命令补齐

## 9. R0 exact-v13a strict-heldout

必须先给 `eval_personalized.py` 增加一个明确 scheme，例如 `v13a_coldstart_exact`：

1. 从非 held-out clients 的 A 更新构造与 `fedplora_v13a_os` 完全相同的 rank-k A-sketch/reconstruction；
2. 从非 held-out clients 构造 routed B pools；
3. 明确 routing 信息预算：metadata-domain、supervised probe、真正 query-only/unlabeled 三者不可混写；
4. JSON 写入 `a_payload`、sketch rank、upload/downlink bytes、route supervision、probe samples；
5. 用相同 held-out client test_local 评估 full-A upper bound 与 exact-v13a，量化 sketch 造成的差值。

在这项实现完成前，第 1/2/5/6 节结果可用于 mechanism/upper-bound 分析，但不能直接作为 exact FedPLoRA-OS cold-start headline。

## 10. R4 exact per-client/per-domain Local

当前 `fed_train_sft.py` formal JSON 只落 `client_local_macro_token_accuracy`，没有各 client 的最终下发模型准确率；`eval_personalized.py` 的 `v7/v11c` 也不是 Normal/FedALT/Hydra/v13a formal checkpoint 的精确回放。

需要先实现：

- final personalization eval 返回 `per_client_acc/loss/ppl`；
- 每条记录带 client/domain/seed/split/fingerprint；
- eval-only from checkpoint 能精确重放各方法最终下发状态；
- 然后只补 D1+FT 的 `{Normal,FedALT,Hydra,v13a,v13b}`，不必 FULL-16 全补。

这项完成后再做主表 Local 的 client/domain clustered bootstrap 与 paired test。

## 11. R0c 35×35 pairwise matrix

需修改 `diag_subspace_AB.py`，至少保存：

```text
client_ids
domains
A_pairwise_angle_deg[35,35]
B_pairwise_angle_deg[35,35]
A_pairwise_similarity[35,35]
B_pairwise_similarity[35,35]
```

当前 `--save_figs` 只生成 intra/inter/null histogram，不是 `claude_Experiment_20260718.md` 计划中的块对角 heatmap。

## 12. R14 真实下游任务指标

当前仓库没有将 Normal/FedALT/v13a 的 exact global/personalized adapter 导出并接入 lm-eval 的完整脚本，因此不能用一条 `lm_eval` 命令伪装完成。建议最小实现后只评：

```text
模型: SmolLM2-1.7B 或 Qwen2.5-3B
方法: base, Normal, FedALT, v13a
任务: 与训练域对应且无训练泄漏的公开 MCQ/EM 子集
指标: exact match / multiple-choice accuracy
```

如果无法在截稿前实现，应在论文 limitation 中明确：当前评测是 teacher-forced token accuracy/PPL，不主张生成质量全面提升。

---

# 第五部分：串行与并行逻辑

```text
Stage 0（串行，0 GPU）
  环境审计 -> Flower offset smoke -> smoke JSON 断言

Stage 1（最高优先）
  代码侧：R0 exact-v13 evaluator
  GPU 侧：在 R0 未完成时，最多先跑 offset1/2 作为 upper-bound robustness

Stage 2（同级并行）
  GPU组A：FlowerTune offset1–4；同一 GPU 串行，不同 GPU 并行
  GPU组B：canonical centralized seed42/43/44
  GPU组C：diag_subspace seed43/44；结束后同卡串行 B-swap seed43/44
  GPU组D：R16 route global/domain；与主表 auto 做归因
  CPU：R13 环境记录、结果清点

Stage 3（等待 Stage 2 的 Flower JSON）
  R7/R12 margin + paired delta + hierarchical bootstrap

Stage 4（条件执行，同事卡）
  Qwen3B smoke -> R9 cold-start -> R10 {Normal,FedALT,v13a}
  R9 与 R10 可在不同卡并行；同一张卡必须串行

Stage 5（需要代码后再运行）
  R4 per-client exact eval -> main Local paired test
  R0c pairwise matrix -> Fig.2(a)
  R14 downstream task metric
```

最有效率的投稿前顺序：

1. **先修 cold-start 协议表述/实现**，否则更多 offset/3B 只会重复一个无法支撑 headline 的协议。
2. 同时跑 Flower offset 与 canonical centralized；两者都是 135M、论文价值高。
3. diag seed43/44 放空闲卡串行；它不是 0-GPU。
4. 3B 只在 smoke 与 protocol gate 通过后启动；7–8B、Dolly、更多 rank/μ 不进入默认队列。

【注意事项】

1. 不重复运行 D1 A1009k offset1；它已经完成。
2. 不把 19k `domain_benchmark_7c` centralized 结果混入 A100 9k 主表。
3. 不把 `coldstart` 写成 label-free；它使用真实 domain。
4. 不把 `coldstart_geom` 写成 zero-data；当前正式设置使用 10 条带 response 的 SFT probe 样本。
5. 不把 full-A cold-start 数值直接标成 v13a A-sketch exact。
6. 不把 `diag_subspace_AB --save_figs` 的 histogram 当成 35×35 heatmap。
7. Qwen2.5-3B 的 license 以下载时模型卡/LICENSE 文件为准，不写成未经核验的 Apache-2.0。
8. 所有正式 JSON 必须保存 benchmark fingerprint、seed、held-out offset、probe samples 和完整有效超参。
