# FedPLoRA-v11 分支归因与 v10 复核实验命令（gb 服务器）

> 由 `order/order_20260705.md` 适配。超参数、算法与实验设计不变；仅修改路径、conda、**默认 GPU=1**，并增强多 seed 下 `BENCHMARK_DIR` 自动同步、`run_sft_*` 打印 `pid`。

######### FedPLoRA-v11 A 放开 / B 诊断 / Global-B Mixing / 多 seed 复核命令 #########

【命令介绍】

本文档包含 8 组命令：

1. 前置检查与公共运行函数。
2. 所有待运行算法族的 smoke 测试。
3. E1：分支 A，放开 A-correction，判断 v10/v11 macro 缺口是否来自 A 被正则掐死。
4. E2：分支 B，建池纯度诊断，验证「混域池 = 隐式 global mixing」的归因假说。
5. E3：分支 C，显式 global-B mixing，验证 v11 主命题。
6. E4：多 seed 可信度复核（seed 43/44）。
7. E7：旧 split 机制复核（`domain_benchmark_35c`）。
8. E8：补缺 baseline（dir05 上 yoco / feddat）+ 汇总脚本。

【命令目的】

本轮代码新增 v11，不改 v10：

- `fedplora_v11a_relaxed_a`：v10-style routed B + 可放开的 A-correction + 真实 rank-k A sketch payload。
- `fedplora_v11c_gmix`：v11a 基础上显式加入 `B_client = μ·B_global + (1−μ)·B_routed`。
- 中间轮即使 `--eval_final_only` 跳过评测，也写入 `lora_expert_stats`，用于追踪 NMI/ARI、cluster hist、A 更新统计。
- 新增 `scripts/Analysis/summarize_fedplora_results.py`，自动汇总主表、几何、通信、域级表，并检查 split 是否混用。

【命令设置】

```text
代码目录: /data/yaominghao/gb/FedPLoRA
主数据: data/domain_benchmark_35c_dir05/seed_{42|43|44}
旧 split（E7）: data/domain_benchmark_35c/seed_42
场景: 7 domains × 5 clients = 35 clients, Dirichlet α=0.5 域内 non-IID + 跨域 task-shift
模型: SmolLM2-135M
LoRA: r=8, alpha=16, dropout=0.05
训练: 10 rounds, local epoch=1, lr=2e-4, batch size=2, max seq length=256
精度: bfloat16
正式评测: eval_max_batches=0, full eval, --eval_final_only, --eval_personalization_metrics
smoke: 1 round, 每客户端 1 train step, eval_max_batches=1
主 seed: 42；多 seed 复核: 43 / 44
GPU: 默认物理 1 号卡（CUDA_VISIBLE_DEVICES=1）
```

注意：个性化 `Gap` 的正确定义是 `Local - OffDom`，不是 `Local - Macro`。

【实验产物位置说明】

```text
run_logs:
/data/yaominghao/gb/result/FedPLoRA/${RUN_ID}/run_logs/test20260705_*.log

result_logs:
/data/yaominghao/gb/result/FedPLoRA/${RUN_ID}/result_logs/<method>/

result_files/client_states:
/data/yaominghao/gb/result/FedPLoRA/${RUN_ID}/result_files/client_states/<method>/

checkpoints:
/data/yaominghao/gb/models/trained_models_LW/${RUN_ID}/<method>_${RUN_TAG}_seed${SEED}

summary:
/data/yaominghao/gb/result/FedPLoRA/${RUN_ID}/result_logs/summary_*.md
```

【实验运行涉及场景】

35-client 跨域 SFT，7 个域分别为 code、education、finance、general、legal、math、medical。核心观察指标是 Macro、Worst、PPL、Local、OffDom、Gap、InDom、B-subspace NMI/ARI、cluster hist、A rel update / row cosine / clip fraction、raw/effective communication。

【实验前置命令】

```bash
set -euo pipefail

conda activate fedplora
cd /data/yaominghao/gb/FedPLoRA

export CODE_ROOT=/data/yaominghao/gb/FedPLoRA
export MODEL_ROOT=/data/yaominghao/gb/models
export DATA_ROOT="$CODE_ROOT/data"
export RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA

export MODEL_PATH="$MODEL_ROOT/SmolLM2-135M"
export EXPECTED_NUM_CLIENTS=35

# 主 split：dir05；E7 临时改为 old35c（见 §7）
export BENCHMARK_SPLIT=dir05
export RUN_ID_PREFIX=v11_20260705_35c_dir05_r10_finaleval

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

export GPU=1

check_benchmark () {
  local dir="$1"
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
if len(domain_counts) != 7 or set(domain_counts.values()) != {5}:
    raise SystemExit(f"[preflight][error] expected 7 domains x 5 clients, found {dict(domain_counts)}")
print(f"[preflight] benchmark_ok path={path.parent} clients={len(clients)} domains={dict(sorted(domain_counts.items()))}")
PY
}

# 按 seed 同步 BENCHMARK_DIR / RUN_ID / SEED / checkpoint 根目录
set_run_paths () {
  local seed="$1"
  export SEED="$seed"
  case "$BENCHMARK_SPLIT" in
    dir05)  export BENCHMARK_DIR="$DATA_ROOT/domain_benchmark_35c_dir05/seed_${seed}" ;;
    dir01)  export BENCHMARK_DIR="$DATA_ROOT/domain_benchmark_35c_dir01/seed_${seed}" ;;
    dir10)  export BENCHMARK_DIR="$DATA_ROOT/domain_benchmark_35c_dir10/seed_${seed}" ;;
    old35c) export BENCHMARK_DIR="$DATA_ROOT/domain_benchmark_35c/seed_${seed}" ;;
    *) echo "[set_run_paths] unknown BENCHMARK_SPLIT=$BENCHMARK_SPLIT" >&2; return 1 ;;
  esac
  export RUN_ID="${RUN_ID_PREFIX}_seed${SEED}"
  export RUN_ROOT="$RESULT_ROOT/$RUN_ID"
  export TRAINED_MODELS_ROOT="$MODEL_ROOT/trained_models_LW/$RUN_ID"
  export RUN_TAG=SmolLM2-135M_${BENCHMARK_SPLIT}_r${ROUNDS}_e${LOCAL_EPOCHS}_lr${LR}
  mkdir -p "$RUN_ROOT/run_logs" "$RUN_ROOT/result_logs" "$RUN_ROOT/result_files/client_states" "$TRAINED_MODELS_ROOT"
  printf '[run] SEED=%s BENCHMARK_SPLIT=%s\n[run] BENCHMARK_DIR=%s\n[run] RUN_ID=%s\n[run] RUN_ROOT=%s\n' \
    "$SEED" "$BENCHMARK_SPLIT" "$BENCHMARK_DIR" "$RUN_ID" "$RUN_ROOT"
}

run_sft_full () {
  local method="$1"
  local agg="$2"
  shift 2
  local gpu="${GPU:-1}"
  local log_path="$RUN_ROOT/run_logs/test20260705_${method}_${RUN_TAG}_seed${SEED}.log"
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

export SMOKE_RUN_ID=v11_20260705_35c_dir05_smoke_seed42
export SMOKE_ROOT="$RESULT_ROOT/$SMOKE_RUN_ID"
export SMOKE_TRAINED_MODELS_ROOT="$MODEL_ROOT/trained_models_LW/$SMOKE_RUN_ID"
mkdir -p "$SMOKE_ROOT/run_logs" "$SMOKE_ROOT/result_logs" "$SMOKE_ROOT/result_files/client_states" "$SMOKE_TRAINED_MODELS_ROOT"

run_sft_smoke () {
  local method="$1"
  local agg="$2"
  shift 2
  local gpu="${GPU:-1}"
  local log_path="$SMOKE_ROOT/run_logs/test20260705_smoke_${method}_seed42.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u tasks/fed_train_sft.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$DATA_ROOT/domain_benchmark_35c_dir05/seed_42" \
    --num_clients "$EXPECTED_NUM_CLIENTS" \
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
    > "$log_path" 2>&1 &
  echo "[run_sft_smoke] method=${method} agg=${agg} gpu=${gpu} pid=$! log=${log_path}"
}

set_run_paths 42
check_benchmark "$BENCHMARK_DIR"

python -m py_compile \
  tasks/fed_train_sft.py \
  utilities/utils.py \
  utilities/train_eval.py \
  methods/lora_expert_baselines.py \
  methods/v11/__init__.py \
  methods/v11/v11_common.py \
  methods/v11/v11a_relaxed_a.py \
  methods/v11/v11c_gmix.py \
  methods/v10/__init__.py \
  methods/v10/geom_a.py \
  methods/v9/__init__.py \
  methods/v9/mix_lora.py \
  methods/v8/__init__.py \
  methods/v8/bsim_lora.py \
  scripts/Analysis/summarize_fedplora_results.py
```

【路径对照（minghao → gb）】

| 项 | order_20260705.md | order_gb_0705.md |
|----|-------------------|------------------|
| conda | `FedRepo2` | `fedplora` |
| 代码 | `/home/minghao/code/FedPLoRA-main` | `/data/yaominghao/gb/FedPLoRA` |
| 模型 | `/data2/minghao/model/SmolLM2-135M` | `/data/yaominghao/gb/models/SmolLM2-135M` |
| 主数据 | `.../domain_benchmark_35c_dir05/seed_*` | `$DATA_ROOT/domain_benchmark_35c_dir05/seed_*` |
| 旧 split | `.../domain_benchmark_35c/seed_42` | `$DATA_ROOT/domain_benchmark_35c/seed_42` |
| 结果 | `/data2/minghao/result/FedPLoRA/` | `/data/yaominghao/gb/result/FedPLoRA/` |
| checkpoint | `/data2/minghao/model/trained_models_LW/` | `/data/yaominghao/gb/models/trained_models_LW/` |
| GPU 默认 | `0` | `1` |

---

【实验运行命令】

## 1. 所有待运行算法族的 smoke 测试

说明：先跑本节。每条命令只启动一个 1-round smoke，用于检查算法导入、训练、聚合、最终评估、metrics JSON 和 checkpoint 写盘。smoke 结果不能作为论文数值。

```bash
GPU=1 run_sft_smoke smoke_v11a_relaxed_a fedplora_v11a_relaxed_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.75 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_smoke smoke_v11a_oracle_domain fedplora_v11a_relaxed_a --expert_cluster_mode domain --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35
GPU=1 run_sft_smoke smoke_v11c_gmix_mu040 fedplora_v11c_gmix --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35
GPU=1 run_sft_smoke smoke_v10_sketch_rank2_alpha050 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.50 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_smoke smoke_v8 fedplora_v8
GPU=1 run_sft_smoke smoke_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5
GPU=1 run_sft_smoke smoke_normal normal
GPU=1 run_sft_smoke smoke_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
GPU=1 run_sft_smoke smoke_yoco yoco --yoco_aggregate_mode conflict --yoco_conflict_method avgm
GPU=1 run_sft_smoke smoke_feddat feddat --feddat_teacher_lambda 0.01
```

smoke 检查：

```bash
find "$SMOKE_ROOT/result_logs" -name '*.json' | sort
tail -n 40 "$SMOKE_ROOT"/run_logs/test20260705_smoke_v11c_gmix_mu040_seed42.log
python scripts/Analysis/summarize_fedplora_results.py "$SMOKE_ROOT" --output "$SMOKE_ROOT/result_logs/summary_smoke.md"
```

### 1.1 一键复制：本节全部 smoke

```bash
GPU=1 run_sft_smoke smoke_v11a_relaxed_a fedplora_v11a_relaxed_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.75 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_smoke smoke_v11a_oracle_domain fedplora_v11a_relaxed_a --expert_cluster_mode domain --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35
GPU=1 run_sft_smoke smoke_v11c_gmix_mu040 fedplora_v11c_gmix --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35
GPU=1 run_sft_smoke smoke_v10_sketch_rank2_alpha050 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.50 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_smoke smoke_v8 fedplora_v8
GPU=1 run_sft_smoke smoke_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5
GPU=1 run_sft_smoke smoke_normal normal
GPU=1 run_sft_smoke smoke_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
GPU=1 run_sft_smoke smoke_yoco yoco --yoco_aggregate_mode conflict --yoco_conflict_method avgm
GPU=1 run_sft_smoke smoke_feddat feddat --feddat_teacher_lambda 0.01
```

---

## 2. E1：分支 A，放开 A-correction

目的：回答「Macro 缺口是不是因为 A correction 太保守」。通过线参考：Macro ≥ 0.600 且 NMI ≥ 0.75。

### 2.1 E1_v11a_alpha075

```bash
set_run_paths 42
GPU=1 run_sft_full E1_v11a_alpha075 fedplora_v11a_relaxed_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.75 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

### 2.2 E1_v11a_alpha100

```bash
GPU=1 run_sft_full E1_v11a_alpha100 fedplora_v11a_relaxed_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 1.00 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

### 2.3 E1_v11a_alpha050_noAreg

```bash
GPU=1 run_sft_full E1_v11a_alpha050_noAreg fedplora_v11a_relaxed_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.50 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 0.0
```

### 2.4 E1_v11a_alpha100_noreg

```bash
GPU=1 run_sft_full E1_v11a_alpha100_noreg fedplora_v11a_relaxed_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 1.00 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0
```

### 2.5 一键复制：E1 全部（seed=42）

```bash
set_run_paths 42
GPU=1 run_sft_full E1_v11a_alpha075 fedplora_v11a_relaxed_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.75 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E1_v11a_alpha100 fedplora_v11a_relaxed_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 1.00 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E1_v11a_alpha050_noAreg fedplora_v11a_relaxed_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.50 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 0.0
GPU=1 run_sft_full E1_v11a_alpha100_noreg fedplora_v11a_relaxed_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 1.00 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0
```

---

## 3. E2：分支 B，建池纯度诊断

目的：验证「v10/v11 收益来自混域池的隐式 global mixing」还是「auto clustering 本身质量不足」。B 是诊断分支，不直接作为最终方法。

### 3.1 E2_v11a_oracle_domain

```bash
set_run_paths 42
GPU=1 run_sft_full E2_v11a_oracle_domain fedplora_v11a_relaxed_a --expert_cluster_mode domain --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

### 3.2 E2_v11a_fixed_k7

```bash
GPU=1 run_sft_full E2_v11a_fixed_k7 fedplora_v11a_relaxed_a --expert_cluster_k 7 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

判读：

- 若 oracle/fixed-K 让 NMI 上升但 Local 掉回 v8，说明混域池确实贡献了隐式 mixing。
- 若 oracle/fixed-K 让 Macro/Local 双升，说明 auto clustering 是瓶颈，后续再做 domain-balanced / fixed-K 主方法。

### 3.3 一键复制：E2 全部（seed=42）

```bash
set_run_paths 42
GPU=1 run_sft_full E2_v11a_oracle_domain fedplora_v11a_relaxed_a --expert_cluster_mode domain --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E2_v11a_fixed_k7 fedplora_v11a_relaxed_a --expert_cluster_k 7 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

---

## 4. E3：分支 C，显式 global-B mixing

目的：把 v10 的「意外混域收益」改成受控机制。通过线：Macro ≥ 0.600 且 Local ≥ 0.630。

### 4.1 E3_v11c_gmix_mu020

```bash
set_run_paths 42
GPU=1 run_sft_full E3_v11c_gmix_mu020 fedplora_v11c_gmix --v11_global_b_mix_mu 0.2 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

### 4.2 E3_v11c_gmix_mu040

```bash
GPU=1 run_sft_full E3_v11c_gmix_mu040 fedplora_v11c_gmix --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

### 4.3 E3_v11c_gmix_mu060

```bash
GPU=1 run_sft_full E3_v11c_gmix_mu060 fedplora_v11c_gmix --v11_global_b_mix_mu 0.6 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

### 4.4 E3_v11c_gmix_mu040_oracle

```bash
GPU=1 run_sft_full E3_v11c_gmix_mu040_oracle fedplora_v11c_gmix --expert_cluster_mode domain --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

如果 E1 中某个 A 配置明显通过，可追加同一 μ 的最佳 A 配置复跑：

### 4.5 E3b_v11c_gmix_mu040_bestA（占位，按 E1 结果改超参）

```bash
set_run_paths 42
GPU=1 run_sft_full E3b_v11c_gmix_mu040_bestA_placeholder fedplora_v11c_gmix --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 1.00 --v10_a_anchor_lambda 0.0 --v10_a_prox_lambda 0.0 --v10_b_prox_lambda 0.0 --v10_a_norm_clip_ratio 0.0
```

### 4.6 一键复制：E3 全部（seed=42，不含 E3b 占位）

```bash
set_run_paths 42
GPU=1 run_sft_full E3_v11c_gmix_mu020 fedplora_v11c_gmix --v11_global_b_mix_mu 0.2 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E3_v11c_gmix_mu040 fedplora_v11c_gmix --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E3_v11c_gmix_mu060 fedplora_v11c_gmix --v11_global_b_mix_mu 0.6 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E3_v11c_gmix_mu040_oracle fedplora_v11c_gmix --expert_cluster_mode domain --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

---

## 5. E4：多 seed 可信度复核

目的：判断当前 0.3–1.7 个点的差距是否稳定。优先 seed 43/44；`set_run_paths` 会自动切到 `dir05/seed_43` 等数据目录。

### 5.1 seed=43

```bash
set_run_paths 43
GPU=1 run_sft_full E4_seed43_normal normal
GPU=1 run_sft_full E4_seed43_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
GPU=1 run_sft_full E4_seed43_v8 fedplora_v8
GPU=1 run_sft_full E4_seed43_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5
GPU=1 run_sft_full E4_seed43_v10_rank1 fedplora_v10_sketch_a --v10_a_sketch_rank 1 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E4_seed43_v10_rank2_alpha050 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.50 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E4_seed43_v11c_gmix_mu040 fedplora_v11c_gmix --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

### 5.2 seed=44

```bash
set_run_paths 44
GPU=1 run_sft_full E4_seed44_normal normal
GPU=1 run_sft_full E4_seed44_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
GPU=1 run_sft_full E4_seed44_v8 fedplora_v8
GPU=1 run_sft_full E4_seed44_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5
GPU=1 run_sft_full E4_seed44_v10_rank1 fedplora_v10_sketch_a --v10_a_sketch_rank 1 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E4_seed44_v10_rank2_alpha050 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.50 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E4_seed44_v11c_gmix_mu040 fedplora_v11c_gmix --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

### 5.3 一键复制：E4 seed43 + seed44 全部

```bash
set_run_paths 43
GPU=1 run_sft_full E4_seed43_normal normal
GPU=1 run_sft_full E4_seed43_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
GPU=1 run_sft_full E4_seed43_v8 fedplora_v8
GPU=1 run_sft_full E4_seed43_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5
GPU=1 run_sft_full E4_seed43_v10_rank1 fedplora_v10_sketch_a --v10_a_sketch_rank 1 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E4_seed43_v10_rank2_alpha050 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.50 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E4_seed43_v11c_gmix_mu040 fedplora_v11c_gmix --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5

set_run_paths 44
GPU=1 run_sft_full E4_seed44_normal normal
GPU=1 run_sft_full E4_seed44_ecolora ecolora --ecolora_keep_ratio 0.25 --ecolora_mask_mode round_robin
GPU=1 run_sft_full E4_seed44_v8 fedplora_v8
GPU=1 run_sft_full E4_seed44_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5
GPU=1 run_sft_full E4_seed44_v10_rank1 fedplora_v10_sketch_a --v10_a_sketch_rank 1 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E4_seed44_v10_rank2_alpha050 fedplora_v10_sketch_a --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.50 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
GPU=1 run_sft_full E4_seed44_v11c_gmix_mu040 fedplora_v11c_gmix --v11_global_b_mix_mu 0.4 --v10_a_sketch_rank 2 --v10_a_correction_alpha 0.35 --v10_a_anchor_lambda 0.001 --v10_a_prox_lambda 0.0005 --v10_b_prox_lambda 0.0001 --v10_a_norm_clip_ratio 1.5
```

跑完 E4 后切回主 split seed=42：

```bash
export BENCHMARK_SPLIT=dir05
export RUN_ID_PREFIX=v11_20260705_35c_dir05_r10_finaleval
set_run_paths 42
```

---

## 6. E7：旧 35c split 机制复核

目的：用当前代码在旧 `domain_benchmark_35c/seed_42` 上复跑 `v9_mix_ab_lam05`，判断「A+B 塌 NMI」是 split 依赖还是代码版本依赖。

```bash
export BENCHMARK_SPLIT=old35c
export RUN_ID_PREFIX=v11_20260705_old35c_r10_finaleval
set_run_paths 42
check_benchmark "$BENCHMARK_DIR"
GPU=1 run_sft_full E7_old35c_v9_mix_ab_lam05 fedplora_v9_mix_ab --v9_mix_lambda 0.5

# 跑完 E7 后切回主 split，避免后续误用旧数据
export BENCHMARK_SPLIT=dir05
export RUN_ID_PREFIX=v11_20260705_35c_dir05_r10_finaleval
set_run_paths 42
```

---

## 7. E8：补缺 baseline（dir05）

目的：补齐 dir05 批次缺失的 `yoco` 和 `feddat`，让主表 baseline 完整。

```bash
set_run_paths 42
GPU=1 run_sft_full E8_yoco yoco --yoco_aggregate_mode conflict --yoco_conflict_method avgm --yoco_sign_lambda 0.01
GPU=1 run_sft_full E8_feddat feddat --feddat_teacher_lambda 0.01
```

### 7.1 一键复制：E8 全部

```bash
set_run_paths 42
GPU=1 run_sft_full E8_yoco yoco --yoco_aggregate_mode conflict --yoco_conflict_method avgm --yoco_sign_lambda 0.01
GPU=1 run_sft_full E8_feddat feddat --feddat_teacher_lambda 0.01
```

---

## 8. 汇总与结果检查

单个 RUN_ID 汇总：

```bash
python scripts/Analysis/summarize_fedplora_results.py "$RUN_ROOT" \
  --output "$RUN_ROOT/result_logs/summary_${RUN_ID}.md" \
  --compare baseline_normal_dir05,baseline_ecolora_dir05,fedplora_v8,E3_v11c_gmix_mu040
```

汇总主 seed E1/E2/E3/E8：

```bash
export BENCHMARK_SPLIT=dir05
export RUN_ID_PREFIX=v11_20260705_35c_dir05_r10_finaleval
set_run_paths 42
python scripts/Analysis/summarize_fedplora_results.py "$RUN_ROOT" \
  --output "$RUN_ROOT/result_logs/summary_seed42_E1_E2_E3_E8.md" \
  --compare E8_yoco,E8_feddat,E1_v11a_alpha100_noreg,E3_v11c_gmix_mu040,E3_v11c_gmix_mu060
```

检查 v11 true sketch payload 与逐轮 stats：

```bash
python - <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for p in sorted(root.rglob("*.json")):
    data = json.loads(p.read_text(encoding="utf-8"))
    comm = data.get("communication", {})
    agg = comm.get("agg_type", "")
    if "v11" not in agg:
        continue
    print("\n", p)
    print("agg", agg)
    print("raw_total", comm.get("down_bytes_per_client", 0) + comm.get("up_bytes_per_client", 0))
    print("eff_total", comm.get("effective_down_bytes_per_client", 0) + comm.get("effective_up_bytes_per_client", 0))
    print("mode", comm.get("v11_a_correction_mode"), "mu", comm.get("v11_global_b_mix_mu"))
    for r in data.get("rounds", []):
        stats = r.get("lora_expert_stats", {})
        print("round", r.get("round"), "eval_skipped", r.get("eval_skipped", False),
              "nmi", stats.get("domain_nmi"), "a_rel", stats.get("v11_a_mean_rel_update_norm"))
PY "$RUN_ROOT"
```

查看后台任务：

```bash
tail -f "$RUN_ROOT/run_logs/test20260705_E3_v11c_gmix_mu040_${RUN_TAG}_seed${SEED}.log"
ps -p <pid> -o pid,cmd
```

---

【注意事项】

1. 不要把 `35c` 与 `35c_dir05` 结果合并进同一论文主表；汇总脚本会列出 benchmark split，若出现多个 split 需分表。
2. 本轮不加 `--force_retrain`；每个 RUN_ID 唯一，保留 resume 能力。若改代码或改协议，递增 `RUN_ID_PREFIX`。
3. E1/E2/E3 建议先跑 seed42，只有 E1 或 E3 单独通过后再推进 E4 多 seed。
4. 如果 `eval_personalization_metrics` 后段长时间无输出，这是 personalization full eval 静默阶段，不是必然卡死；看 `nvidia-smi` 与 run log 尾部判断。
5. v11 的 raw/effective 通信应接近一致；若 summary 显示 v11 raw 仍等于完整 A+B，说明命令没有跑到新代码或 agg_type 写错。
6. `set_run_paths <seed>` 会同步 `BENCHMARK_DIR=.../dir05/seed_<seed>` 与训练 `--seed`；E4 跑 43/44 前确认对应数据已建好（见 `order_gb.md` §0）。
7. 并行多任务时把 `GPU=1` 改成 `GPU=2`、`GPU=3`；`run_sft_full` / `run_sft_smoke` 已 `nohup ... &` 并 echo `pid`，可连续敲多条。
