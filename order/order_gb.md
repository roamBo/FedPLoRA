# FedPLoRA-v10 几何保持 A-correction 实验命令（gb 服务器）

> 由 `order/order_20260630.md` 适配。超参数、算法与 SmolLM2-135M 配置不变；仅修改代码/数据/产物路径，**默认 GPU=1**。

######### FedPLoRA-v10 A-correction + B-geometry 与 baseline 对比命令 #########

【命令介绍】

本文档包含 2 组命令：

1. smoke 测试：对本轮所有要运行的 v10 主方法、v8/v9 对照和 baseline 各跑 1 轮、每客户端 1 个 train step、eval 只测 1 个 batch，用于确认算法注册、训练、聚合、最终评估、metrics JSON 和 checkpoint 写盘流程可跑通。
2. 正式 10 轮实验：主测 `fedplora_v10_sketch_a` 与 `fedplora_v10_geom_a`，并复跑 v8/v9 关键对照和主流 baseline。正式命令全部使用 `--eval_final_only`，只在所有轮次训练完成后做一次 full eval，避免每轮评测浪费时间。

【命令目的】

20260630 的两份分析文档均指出：v9 的固定或自适应 lambda 只是 `v8_global` 与 `v8` 两个端点之间的插值，无法突破由“低通信、macro、B 域几何”构成的不可能三角。v10 的目的不是继续扫 lambda，而是验证一条新的 Pareto 路径：

```text
本地训练 A+B
服务器仅采用受控 A-correction 更新 shared A
B 仍按 B-subspace expert pool 硬路由
sketch 版本按低秩 A-delta 估算有效通信
```

判定标准是：`fedplora_v10_sketch_a` 能否把 Macro 拉近 A+B baseline，同时保持 B-domain 聚类质量（NMI/ARI）和明显低于完整 A+B 的有效通信。

【命令设置】

```text
代码目录: /data/yaominghao/gb/FedPLoRA
数据（跨域 task-shift，无域内 α）: data/domain_benchmark_35c/seed_42
数据（7 域 + 域内 Dirichlet feature-skew）: data/domain_benchmark_35c_dir{01,05,10}/seed_<42|43|44>
场景: 7 domains x 5 clients = 35 clients
模型: SmolLM2-135M
LoRA: r=8, alpha=16, dropout=0.05
训练: 10 rounds, local epoch=1, lr=2e-4, batch size=2, max seq length=256
精度: bfloat16
正式评测: eval_max_batches=0, full eval, eval_personalization_metrics 开启
评测频率: --eval_final_only, 只在 final round 后评测一次
seed: 42（多 seed 时改 --seed 与 RUN_ID）
GPU: 默认物理 1 号卡（CUDA_VISIBLE_DEVICES=1）
```

【两套 35c 数据说明】

| 类型 | 目录示例 | 有无 Dirichlet α |
|------|----------|------------------|
| **跨域 task-shift**（原版 order） | `domain_benchmark_35c/seed_42` | **无**（异质性来自 7 个 domain） |
| **域内 feature-skew**（v2 构建器） | `domain_benchmark_35c_dir05/seed_42` | **有** α=0.1 / 0.5 / 1.0（域内客户端子主题偏斜） |

域内 non-IID 用 `build_domain_benchmark_v2.py`；**α 越小越 non-IID**（不用 α=0，用 **0.1** 表示强偏斜）。

---

## 0. 构建域内 Dirichlet 三套数据（α=0.1 / 0.5 / 1.0）

原始语料（每行含 `domain` / `prompt` / `response`）：

```bash
export RAW_JSONL="$DATA_ROOT/raw/domain_7_all.jsonl"
ls "$RAW_JSONL"
```

> 你给的命令**缺了必填项 `--input_jsonl`**；`--subtopic kmeans` 需 sklearn，服务器无 sklearn 时会**自动退回 length**。与仓库 `PERSONALIZED_EVAL_V7_COMMANDS.md` 一致时可用 `--subtopic length`。

```bash
cd /data/yaominghao/gb/FedPLoRA
conda activate fedplora
export DATA_ROOT="$PWD/data"
mkdir -p log_build

# α=0.1（强域内 non-IID）
python -u scripts/DataProcessScripts/build_domain_benchmark_v2.py \
  --input_jsonl "$RAW_JSONL" \
  --output_dir "$DATA_ROOT/domain_benchmark_35c_dir01" \
  --num_clients_per_domain 5 --seed 42 \
  --dedup prompt --target_per_domain 2000 \
  --partition dirichlet --dirichlet_alpha 0.1 \
  --subtopic kmeans --n_subtopics 10 \
  2>&1 | tee log_build/build_35c_dir01_seed42.log

# α=0.5（中度域内 non-IID）
python -u scripts/DataProcessScripts/build_domain_benchmark_v2.py \
  --input_jsonl "$RAW_JSONL" \
  --output_dir "$DATA_ROOT/domain_benchmark_35c_dir05" \
  --num_clients_per_domain 5 --seed 42 \
  --dedup prompt --target_per_domain 2000 \
  --partition dirichlet --dirichlet_alpha 0.5 \
  --subtopic kmeans --n_subtopics 10 \
  2>&1 | tee log_build/build_35c_dir05_seed42.log

# α=1.0（弱域内 non-IID，接近域内 IID）
python -u scripts/DataProcessScripts/build_domain_benchmark_v2.py \
  --input_jsonl "$RAW_JSONL" \
  --output_dir "$DATA_ROOT/domain_benchmark_35c_dir10" \
  --num_clients_per_domain 5 --seed 42 \
  --dedup prompt --target_per_domain 2000 \
  --partition dirichlet --dirichlet_alpha 1.0 \
  --subtopic kmeans --n_subtopics 10 \
  2>&1 | tee log_build/build_35c_dir10_seed42.log
```

多 seed（42 / 43 / 44）：**同一 `output_dir`，只改 `--seed`**，产物在 `.../seed_43`、`.../seed_44`：

```bash
for S in 42 43 44; do
  python -u scripts/DataProcessScripts/build_domain_benchmark_v2.py \
    --input_jsonl "$RAW_JSONL" \
    --output_dir "$DATA_ROOT/domain_benchmark_35c_dir05" \
    --num_clients_per_domain 5 --seed "$S" \
    --dedup prompt --target_per_domain 2000 \
    --partition dirichlet --dirichlet_alpha 0.5 \
    --subtopic kmeans --n_subtopics 10
done
```

构建成功应看到：`[leakcheck] PASS — zero prompt-level leakage`。检查：

```bash
ls "$DATA_ROOT/domain_benchmark_35c_dir05/seed_42/clients.json"
```

---

【实验产物位置说明】

```text
run_logs:
/data/yaominghao/gb/result/FedPLoRA/v10_20260630_35c_r10_finaleval_seed42/run_logs/test20260630_*.log

result_logs:
/data/yaominghao/gb/result/FedPLoRA/v10_20260630_35c_r10_finaleval_seed42/result_logs/<method>/

result_files/client_states:
/data/yaominghao/gb/result/FedPLoRA/v10_20260630_35c_r10_finaleval_seed42/result_files/client_states/<method>/

checkpoints:
/data/yaominghao/gb/models/trained_models_LW/v10_20260630_35c_r10_finaleval_seed42/<method>_SmolLM2-135M_35c_r10_e1_lr0.0002_seed42
```

【实验运行涉及场景】

35-client 跨域 SFT，7 个域分别为 code、education、finance、general、legal、math、medical。核心比较对象是 domain-macro token accuracy、worst-domain token accuracy、client-local personalization accuracy、local-off personalization gap、B-subspace domain NMI/ARI 和 effective communication。

【实验前置命令】

```bash
set -euo pipefail

conda activate fedplora
cd /data/yaominghao/gb/FedPLoRA

export CODE_ROOT=/data/yaominghao/gb/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models
export DATA_ROOT="$CODE_ROOT/data"

export MODEL_PATH="$MODEL_ROOT/SmolLM2-135M"
export BENCHMARK_DIR="$DATA_ROOT/domain_benchmark_35c/seed_42"
export EXPECTED_NUM_CLIENTS=35

export SEED=42
export RUN_ID=v10_20260630_35c_r10_finaleval_seed${SEED}
export RUN_ROOT=/data/yaominghao/gb/result/FedPLoRA/$RUN_ID
export TRAINED_MODELS_ROOT="$MODEL_ROOT/trained_models_LW/$RUN_ID"
export RUN_TAG=SmolLM2-135M_35c_r10_e1_lr0.0002

export ROUNDS=10
export LOCAL_EPOCHS=1
export LR=0.0002
export LORA_R=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.05
export BATCH_SIZE=2
export MAX_SEQ_LENGTH=256
export TORCH_DTYPE=bfloat16
export TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export EVAL_MAX_BATCHES=0

# 默认物理 GPU 1；并行多任务时可 GPU=2 run_sft_full ...
export GPU=1

# 切换到域内 Dirichlet 某套数据（α=0.1|0.5|1.0），并重设 RUN_ID 防 checkpoint 覆盖
use_domain_dirichlet () {
  local alpha="$1"
  local bench_seed="${2:-${SEED:-42}}"
  case "$alpha" in
    0.1|01|dir01) export DIR_ALPHA_TAG=dir01; export DIRICHLET_ALPHA=0.1 ;;
    0.5|05|dir05) export DIR_ALPHA_TAG=dir05; export DIRICHLET_ALPHA=0.5 ;;
    1.0|10|dir10) export DIR_ALPHA_TAG=dir10; export DIRICHLET_ALPHA=1.0 ;;
    *) echo "[use_domain_dirichlet] unknown alpha=$alpha (use 0.1 / 0.5 / 1.0)" >&2; return 1 ;;
  esac
  export BENCHMARK_DIR="$DATA_ROOT/domain_benchmark_35c_${DIR_ALPHA_TAG}/seed_${bench_seed}"
  export RUN_ID="v10_35c_r10_${DIR_ALPHA_TAG}_finaleval_seed${bench_seed}"
  export RUN_ROOT="/data/yaominghao/gb/result/FedPLoRA/$RUN_ID"
  export TRAINED_MODELS_ROOT="$MODEL_ROOT/trained_models_LW/$RUN_ID"
  mkdir -p "$RUN_ROOT/run_logs" "$RUN_ROOT/result_logs" "$RUN_ROOT/result_files/client_states" "$TRAINED_MODELS_ROOT"
  echo "[benchmark] dirichlet_alpha=${DIRICHLET_ALPHA} BENCHMARK_DIR=${BENCHMARK_DIR} RUN_ID=${RUN_ID}"
}

run_sft_full () {
  local method="$1"
  local agg="$2"
  shift 2
  local gpu="${GPU:-1}"
  local log_path="$RUN_ROOT/run_logs/test20260630_${method}_${RUN_TAG}_seed${SEED}.log"
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
    > "$log_path" 2>&1 &
  echo "[run_sft_full] method=${method} agg=${agg} gpu=${gpu} pid=$! log=${log_path}"
}

export SMOKE_RUN_ID=v10_20260630_35c_smoke_seed${SEED}
export SMOKE_ROOT=/data/yaominghao/gb/result/FedPLoRA/$SMOKE_RUN_ID
export SMOKE_TRAINED_MODELS_ROOT="$MODEL_ROOT/trained_models_LW/$SMOKE_RUN_ID"
mkdir -p "$SMOKE_ROOT/run_logs" "$SMOKE_ROOT/result_logs" "$SMOKE_ROOT/result_files/client_states" "$SMOKE_TRAINED_MODELS_ROOT"

run_sft_smoke () {
  local method="$1"
  local agg="$2"
  shift 2
  local gpu="${GPU:-1}"
  local log_path="$SMOKE_ROOT/run_logs/test20260630_smoke_${method}_seed${SEED}.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u tasks/fed_train_sft.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --num_clients "$EXPECTED_NUM_CLIENTS" \
    --agg_type "$agg" \
    --rounds 1 --local_epochs 1 --lr "$LR" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
    --client_state_dir "$SMOKE_ROOT/result_files/client_states/$method" \
    --metrics_output_dir "$SMOKE_ROOT/result_logs/$method" \
    --save_run_checkpoint_dir "$SMOKE_TRAINED_MODELS_ROOT/${method}_${RUN_TAG}_smoke_seed${SEED}" \
    --trained_models_root "$SMOKE_TRAINED_MODELS_ROOT" \
    --eval_max_batches 1 --seed "$SEED" \
    --train_max_steps_per_client 1 \
    --max_train_samples_per_client 10 \
    --save_client_state_to_disk \
    --gradient_checkpointing \
    --eval_personalization_metrics \
    --eval_final_only \
    --skip_post_agg_snapshots \
    "$@" \
    > "$log_path" 2>&1 &
  echo "[run_sft_smoke] method=${method} agg=${agg} gpu=${gpu} pid=$! log=${log_path}"
}

python - "$BENCHMARK_DIR/clients.json" "$EXPECTED_NUM_CLIENTS" <<'PY'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
clients = json.loads(path.read_text(encoding="utf-8"))
domain_counts = collections.Counter(str(row["domain"]) for row in clients)
if len(clients) != expected:
    raise SystemExit(f"[preflight][error] expected {expected} clients, found {len(clients)}")
if len(domain_counts) != 7 or set(domain_counts.values()) != {5}:
    raise SystemExit(f"[preflight][error] expected 7 domains x 5 clients, found {dict(domain_counts)}")
print(f"[preflight] benchmark_ok clients={len(clients)} domains={dict(sorted(domain_counts.items()))}")
PY

mkdir -p "$RUN_ROOT/run_logs" "$RUN_ROOT/result_logs" "$RUN_ROOT/result_files/client_states" "$TRAINED_MODELS_ROOT"

python -m py_compile \
  tasks/fed_train_sft.py \
  utilities/utils.py \
  utilities/train_eval.py \
  methods/lora_expert_baselines.py \
  methods/v8/__init__.py \
  methods/v8/bsim_lora.py \
  methods/v9/__init__.py \
  methods/v9/mix_lora.py \
  methods/v10/__init__.py \
  methods/v10/geom_a.py
```

【实验运行命令】

## 1. 所有待运行算法的 smoke 测试

说明：先跑本节。每条命令只启动一个 1-round smoke，用于检查新算法、旧算法和 baseline 都能写出 metrics JSON 与 final bundle。smoke 结果不能作为论文数值。

### 1.1 v10 主方法 smoke

```bash
GPU=1 run_sft_smoke smoke_fedplora_v10_sketch_a_rank1 fedplora_v10_sketch_a --v10_a_sketch_rank 1 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v10_sketch_a_rank2 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v10_sketch_a_rank4 fedplora_v10_sketch_a --v10_a_sketch_rank 4 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v10_sketch_a_alpha020 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.20 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v10_sketch_a_alpha050 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.50 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v10_geom_a_alpha035 fedplora_v10_geom_a --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
```

### 1.2 v8/v9 对照 smoke

```bash
GPU=1 run_sft_smoke smoke_fedplora_v9_mix_lam07 fedplora_v9_mix --v9_mix_lambda 0.7 --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5 --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v8 fedplora_v8 --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v8_global fedplora_v8 --expert_cluster_mode global --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v8_warma fedplora_v8_warma --v8_a_warmup_rounds 1 --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v8_periodic_T5 fedplora_v8_periodic --v8_a_refresh_interval 5 --v8_cache_shared_a_downlink
GPU=1 run_sft_smoke smoke_fedplora_v8_ab fedplora_v8_ab --v8_cache_shared_a_downlink
```

### 1.3 baseline smoke

```bash
GPU=1 run_sft_smoke smoke_baseline_normal normal
GPU=1 run_sft_smoke smoke_baseline_ffa ffa
GPU=1 run_sft_smoke smoke_baseline_flora flora
GPU=1 run_sft_smoke smoke_baseline_flexlora flexlora
GPU=1 run_sft_smoke smoke_baseline_ecolora ecolora --ecolora_keep_ratio 0.25
GPU=1 run_sft_smoke smoke_baseline_fedlease fedlease
GPU=1 run_sft_smoke smoke_baseline_hilora hilora
GPU=1 run_sft_smoke smoke_baseline_hydralora hydralora
GPU=1 run_sft_smoke smoke_baseline_fedalt fedalt
GPU=1 run_sft_smoke smoke_baseline_fedsa_lora fedsa_lora
GPU=1 run_sft_smoke smoke_baseline_fedplora_oneshot fedplora_oneshot
GPU=1 run_sft_smoke smoke_baseline_yoco yoco
GPU=1 run_sft_smoke smoke_baseline_feddat feddat
```

## 2. 正式 10 轮 v10 主方法与消融

说明：本节是 20260630 后最重要的实验。`rank2 + alpha0.35` 是主推荐点；rank1/rank4 用于通信-性能曲线；alpha0.20/0.50 用于判断 A-correction 强度；`geom_a` 是完整 A-correction 上界。

### 2.1 fedplora_v10_sketch_a rank=1

```bash
GPU=1 run_sft_full fedplora_v10_sketch_a_rank1 fedplora_v10_sketch_a --v10_a_sketch_rank 1 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
```

### 2.2 fedplora_v10_sketch_a rank=2（主方法）

```bash
GPU=1 run_sft_full fedplora_v10_sketch_a_rank2 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
```

### 2.3 fedplora_v10_sketch_a rank=4

```bash
GPU=1 run_sft_full fedplora_v10_sketch_a_rank4 fedplora_v10_sketch_a --v10_a_sketch_rank 4 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
```

### 2.4 fedplora_v10_sketch_a alpha=0.20

```bash
GPU=1 run_sft_full fedplora_v10_sketch_a_rank2_alpha020 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.20 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
```

### 2.5 fedplora_v10_sketch_a alpha=0.50

```bash
GPU=1 run_sft_full fedplora_v10_sketch_a_rank2_alpha050 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.50 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
```

### 2.6 fedplora_v10_geom_a 完整 A-correction 上界

```bash
GPU=1 run_sft_full fedplora_v10_geom_a_alpha035 fedplora_v10_geom_a --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5 --v8_cache_shared_a_downlink
```

## 3. v8/v9 关键对照

说明：这些命令用于确认 v10 是否真的突破 v9 被支配的插值线，而不是只在旧 trade-off 上移动。

### 3.1 fedplora_v9_mix lambda=0.7

```bash
GPU=1 run_sft_full fedplora_v9_mix_lam07 fedplora_v9_mix --v9_mix_lambda 0.7 --v8_cache_shared_a_downlink
```

### 3.2 fedplora_v9_mix_ab lambda=0.5

```bash
GPU=1 run_sft_full fedplora_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5 --v8_cache_shared_a_downlink
```

### 3.3 fedplora_v8

```bash
GPU=1 run_sft_full fedplora_v8 fedplora_v8 --v8_cache_shared_a_downlink
```

### 3.4 fedplora_v8_global

```bash
GPU=1 run_sft_full fedplora_v8_global fedplora_v8 --expert_cluster_mode global --v8_cache_shared_a_downlink
```

### 3.5 fedplora_v8_warma

```bash
GPU=1 run_sft_full fedplora_v8_warma fedplora_v8_warma --v8_a_warmup_rounds 1 --v8_cache_shared_a_downlink
```

### 3.6 fedplora_v8_periodic_T5

```bash
GPU=1 run_sft_full fedplora_v8_periodic_T5 fedplora_v8_periodic --v8_a_refresh_interval 5 --v8_cache_shared_a_downlink
```

### 3.7 fedplora_v8_ab

```bash
GPU=1 run_sft_full fedplora_v8_ab fedplora_v8_ab --v8_cache_shared_a_downlink
```

## 4. baseline 正式 10 轮

说明：本节覆盖当前表格中最强的全局 baseline、低通信 baseline、LoRA expert baseline 与之前缺结果的 YOCO/FedDAT。

### 4.1 Normal FedAvg-LoRA

```bash
GPU=1 run_sft_full baseline_normal normal
```

### 4.2 FFA

```bash
GPU=1 run_sft_full baseline_ffa ffa
```

### 4.3 FLoRA

```bash
GPU=1 run_sft_full baseline_flora flora
```

### 4.4 FlexLoRA

```bash
GPU=1 run_sft_full baseline_flexlora flexlora
```

### 4.5 EcoLoRA

```bash
GPU=1 run_sft_full baseline_ecolora ecolora --ecolora_keep_ratio 0.25
```

### 4.6 FedLEASE

```bash
GPU=1 run_sft_full baseline_fedlease fedlease
```

### 4.7 HiLoRA

```bash
GPU=1 run_sft_full baseline_hilora hilora
```

### 4.8 HydraLoRA

```bash
GPU=1 run_sft_full baseline_hydralora hydralora
```

### 4.9 FedALT

```bash
GPU=1 run_sft_full baseline_fedalt fedalt
```

### 4.10 FedSA-LoRA

```bash
GPU=1 run_sft_full baseline_fedsa_lora fedsa_lora
```

### 4.11 FedPLoRA-Oneshot

```bash
GPU=1 run_sft_full baseline_fedplora_oneshot fedplora_oneshot
```

### 4.12 YOCO

```bash
GPU=1 run_sft_full baseline_yoco yoco
```

### 4.13 FedDAT

```bash
GPU=1 run_sft_full baseline_feddat feddat
```

> 每条 `run_sft_full` 会 **立即返回** 并打印 `pid=...` 与 `log=...`（`nohup` 后台，不占当前终端）。并行示例：`GPU=1 run_sft_full ...` 与 `GPU=2 run_sft_full ...` 各开一条。

---

## 5. baseline × 域内 Dirichlet 三分布（α=0.1 / 0.5 / 1.0）

**前置**：先完成 §0 构建 `domain_benchmark_35c_dir01|dir05|dir10`，再 `source` 或粘贴 §1 前置块（含 `run_sft_full`、`use_domain_dirichlet`）。

切换分布（**必须**在每条 baseline 前调用，或写进循环）：

```bash
use_domain_dirichlet 0.5    # 或 0.1 / 1.0；第二参数可改 bench seed：use_domain_dirichlet 0.5 43
```

`method` 名建议带 `_${DIR_ALPHA_TAG}`，避免与跨域 `domain_benchmark_35c` 或另一 α 的 checkpoint 冲突。

### 5.1 单分布：α=0.5 示例（其余 α 只改 `use_domain_dirichlet` 与 method 后缀）

```bash
use_domain_dirichlet 0.5

GPU=1 run_sft_full baseline_normal_dir05 normal
GPU=1 run_sft_full baseline_ffa_dir05 ffa
GPU=1 run_sft_full baseline_flora_dir05 flora
GPU=1 run_sft_full baseline_flexlora_dir05 flexlora
GPU=1 run_sft_full baseline_ecolora_dir05 ecolora --ecolora_keep_ratio 0.25
GPU=1 run_sft_full baseline_fedlease_dir05 fedlease
GPU=1 run_sft_full baseline_hilora_dir05 hilora
GPU=1 run_sft_full baseline_hydralora_dir05 hydralora
GPU=1 run_sft_full baseline_fedalt_dir05 fedalt
GPU=1 run_sft_full baseline_fedsa_lora_dir05 fedsa_lora
GPU=1 run_sft_full baseline_fedplora_oneshot_dir05 fedplora_oneshot
GPU=1 run_sft_full baseline_yoco_dir05 yoco
GPU=1 run_sft_full baseline_feddat_dir05 feddat
```

α=0.1：`use_domain_dirichlet 0.1`，method 后缀 `_dir01`。α=1.0：`use_domain_dirichlet 1.0`，后缀 `_dir10`。

### 5.2 一次扫三分布（同一 baseline，多 GPU 并行）

```bash
for ALPHA in 0.1 0.5 1.0; do
  use_domain_dirichlet "$ALPHA"
  GPU=1 run_sft_full "baseline_normal_${DIR_ALPHA_TAG}" normal
done
# 三条命令会立刻打出三行 pid；若只有一张卡，请逐条跑或改 GPU=2/3
```

### 5.3 v10 主实验在三分布上（可选）

```bash
for ALPHA in 0.1 0.5 1.0; do
  use_domain_dirichlet "$ALPHA"
  GPU=1 run_sft_full "fedplora_v10_${DIR_ALPHA_TAG}" fedplora_v10 \
    --v10_a_sketch_rank 4 --v10_a_sketch_beta 0.5 --v10_a_sketch_mode row \
    --v10_a_rel_clip 0.1 --v10_a_row_cos_threshold 0.95 --v10_a_row_cos_mode clip \
    --v10_cache_shared_a_downlink
done
```

### 5.4 查看后台任务

```bash
tail -f "$RUN_ROOT/run_logs/test20260630_baseline_normal_dir05_r10_seed42.log"
ps -p <pid> -o pid,cmd
```

---

| 项 | order_20260630.md | order_gb.md |
|----|-------------------|-------------|
| conda | `FedRepo2` | `fedplora` |
| 代码 | `/home/minghao/code/FedPLoRA-main` | `/data/yaominghao/gb/FedPLoRA` |
| 模型 | `/data2/minghao/model/SmolLM2-135M` | `/data/yaominghao/gb/models/SmolLM2-135M` |
| 数据（跨域） | `.../domain_benchmark_35c/seed_42` | `.../domain_benchmark_35c/seed_42` |
| 数据（域内 Dirichlet） | — | `.../domain_benchmark_35c_dir{01,05,10}/seed_42` |
| 结果 | `/data2/minghao/result/FedPLoRA/` | `/data/yaominghao/gb/result/FedPLoRA/` |
| checkpoint | `/data2/minghao/model/trained_models_LW/` | `/data/yaominghao/gb/models/trained_models_LW/` |
| GPU 默认 | `0` | `1` |

【注意事项】

1. 如果 smoke 日志出现 `argument --max_train_samples_per_client: invalid int value: ''`，说明 shell 环境中传入了空变量；本文命令已在 smoke 中直接写死整数 `10`，正式实验则不传该参数。
2. 如果日志显示 `[resume] Run fully complete ... skipping training and evaluation`，说明同名 checkpoint 已完成。需要重跑时在对应命令末尾追加 `--force_retrain`，或更换 `RUN_ID`。
3. `fedplora_v10_sketch_a` 当前框架内部仍序列化重构后的 dense A 以兼容单 adapter PEFT 训练；metrics 中的 effective communication 按 rank-k A-delta sketch 计。若该点成为论文主结果，后续应再补一个真实 low-rank payload 的系统实现版本。
4. v10 是否成立不能只看 Macro。必须同时检查 run log 中 `[lora-expert] domain_nmi/domain_ari` 和 `[fedplora-v10] a_rel_update/a_row_cos/a_clipped_row_frac`，否则无法证明“追回 A 的 macro 且不塌 B 几何”。
5. 并行多任务时把 `GPU=1` 改成 `GPU=2`、`GPU=3` 等；前置块里 `export GPU=1` 为默认值。`run_sft_full` / `run_sft_smoke` 已 `nohup ... &` 并 **echo pid**，可连续敲多条 baseline。
6. 域内 Dirichlet 实验勿与 §2–§4 默认 `BENCHMARK_DIR=domain_benchmark_35c` 混用；先 `use_domain_dirichlet` 再跑。FedDAT **一条命令只跑一个 α**，不会自动扫三分布。
