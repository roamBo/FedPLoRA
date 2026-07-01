# FedPLoRA 个性化评测 + v7 分层方法 运行命令

本文档用于在小模型上运行 **A/B 子空间诊断**、**per-domain 个性化评测** 与 **v7 非对称分层方法**。命令均按 `nohup python -u ... > log 2>&1 &` 风格给出，便于服务器后台运行。

当前代码状态：

- **诊断脚本**已实现（独立运行，不经 `agg_type`）：`scripts/Analysis/diag_subspace.py`（A 行空间几何）、`scripts/Analysis/diag_subspace_AB.py`（A 行空间 vs B 列空间）、`scripts/Analysis/diag_b_swap.py`（同域 B 互换）。
- **个性化评测 + v7** 已实现（独立运行）：`scripts/Analysis/eval_personalized.py`，内含 4 套方案对比（`local` / `fedsa` / `global` / `v7`）；v7 聚合器在 `methods/v7/hier_lora.py`。
- v7 当前为 **standalone 原型**：捕获各客户端 `(A_i,B_i)` → 在脚本内聚合 → per-domain 评测；**尚未**接入 `tasks/fed_train_sft.py` 成为 `agg_type`（见 §8 待办）。

> **重要前提**：诊断与 v7 都需要**每域 ≥2 客户端**（域内共识/池化）。**必须用 35c（每域 5 客户端）**，不要用 LW7c（每域 1 客户端）。

## 1. 推荐运行顺序

1. 准备模型与 35c 数据（与 `SMALL_MODEL_RUN_COMMANDS.md` §2/§3 相同）。
2. 跑三个诊断确认结构性结论（A 域通用、B 域特异、同域 B 可互换）：`diag_subspace_AB.py`、`diag_b_swap.py`。
3. 跑 `eval_personalized.py`，对比 `local/fedsa/global/v7`，**用 per-domain 个性化目标**判定 v7 是否反超通才。
4. `--v7_b_mode` 三种（`mean`/`rep`/`svd`）各跑一遍，定位最优 B 池化方式。
5. 若 v7 在 per-domain 上稳定 > global，再考虑接入训练框架 + 上更大模型（§8）。

两块 GPU 调度建议：

- GPU0：`diag_subspace_AB.py` + `eval_personalized.py`（`mean`）。
- GPU1：`diag_b_swap.py` + `eval_personalized.py`（`rep` / `svd`）。
- 每张 GPU 同时建议只跑一个进程；不要一次性复制本文所有 `nohup` 命令。

## 2. 环境变量与目录

先进入仓库根目录。下面命令需逐行或整段粘贴执行，不要把 `cd` 和 `mkdir -p` 合并到同一行。

```bash
export CODE_ROOT=/home/minghao/code/FedPLoRA-main
export MODEL_ROOT=/data2/minghao/model
export DATA_ROOT="$CODE_ROOT/data"

cd "$CODE_ROOT" || exit 1
mkdir -p log_diag artifacts_35c/diag artifacts_35c/eval_personalized
```

默认小模型配置（35c，每域 5 客户端）：

```bash
export MODEL_PATH="$MODEL_ROOT/SmolLM2-135M"
export BENCHMARK_DIR="$DATA_ROOT/domain_benchmark_35c/seed_42"

export LORA_R=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.05
export BATCH_SIZE=2
export MAX_SEQ_LENGTH=512
export TORCH_DTYPE=bfloat16
export TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export SEED=42
```

快筛可减小（更快，信号略弱）：

```bash
export MAX_SEQ_LENGTH=256
# 诊断/评测脚本用 --max_steps 限制每客户端训练步数；0 表示整个 epoch
```

检查数据与模型就绪：

```bash
ls "$BENCHMARK_DIR"/clients.json "$BENCHMARK_DIR"/train.jsonl "$BENCHMARK_DIR"/test_domain.jsonl
find "$MODEL_PATH" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) -print
python -c "import torch, transformers, peft; print('env ok', torch.__version__)"
```

## 3. 诊断命令

### 3.1 A 行空间几何（域内/域间/随机 null）

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/Analysis/diag_subspace.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --max_steps 0 --seed "$SEED" \
  --out artifacts_35c/diag/diag_subspace_A_seed${SEED}.json --save_figs \
  > log_diag/diag_subspace_A_seed${SEED}.log 2>&1 &
```

### 3.2 A 行空间 vs B 列空间（域信号占比对比）

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/Analysis/diag_subspace_AB.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --max_steps 0 --seed "$SEED" \
  --out artifacts_35c/diag/diag_AB_seed${SEED}.json --save_figs \
  > log_diag/diag_AB_seed${SEED}.log 2>&1 &
```

### 3.3 同域 B 互换（B 可共享性）

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/diag_b_swap.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --max_steps 0 --seed "$SEED" \
  --eval_max_batches 20 --n_peers 4 --n_cross 2 \
  --out artifacts_35c/diag/diag_b_swap_seed${SEED}.json \
  > log_diag/diag_b_swap_seed${SEED}.log 2>&1 &
```

## 4. 个性化评测 + v7 命令

`eval_personalized.py` 捕获各客户端 `(A_i,B_i)` 后，对比 4 套方案，全部用 **per-domain 个性化评测**（每客户端只在自己域测试集上评测）：

- `local` : `(A_i, B_i)` —— 各自本地，个性化上界。
- `fedsa` : `(A_global, B_i)` —— 共享 A、本地 B。
- `global`: `(A_global, B_global)` —— FedAvg 通才。
- `v7`    : `(A_global, B_domain)` —— v7：全局 A + 按域池化 B（跨域隔离）。

### 4.1 v7 B 池化 = mean（最简）

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --max_steps 0 --eval_max_batches 30 --seed "$SEED" \
  --schemes local,fedsa,global,v7 \
  --v7_b_mode mean \
  --out artifacts_35c/eval_personalized/eval_personalized_mean_seed${SEED}.json \
  > log_diag/eval_personalized_mean_seed${SEED}.log 2>&1 &
```

### 4.2 v7 B 池化 = rep（取同域代表 B，gauge 自洽）

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --max_steps 0 --eval_max_batches 30 --seed "$SEED" \
  --schemes local,fedsa,global,v7 \
  --v7_b_mode rep \
  --out artifacts_35c/eval_personalized/eval_personalized_rep_seed${SEED}.json \
  > log_diag/eval_personalized_rep_seed${SEED}.log 2>&1 &
```

### 4.3 v7 B 池化 = svd（列空间共识去噪，gauge 鲁棒）

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --max_steps 0 --eval_max_batches 30 --seed "$SEED" \
  --schemes local,fedsa,global,v7 \
  --v7_b_mode svd \
  --out artifacts_35c/eval_personalized/eval_personalized_svd_seed${SEED}.json \
  > log_diag/eval_personalized_svd_seed${SEED}.log 2>&1 &
```

### 4.4 更长本地训练（让 B 收敛后再评，验证 1-epoch 噪声影响）

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --local_epochs 2 --max_steps 0 --eval_max_batches 30 --seed "$SEED" \
  --schemes local,fedsa,global,v7 \
  --v7_b_mode mean \
  --out artifacts_35c/eval_personalized/eval_personalized_mean_e2_seed${SEED}.json \
  > log_diag/eval_personalized_mean_e2_seed${SEED}.log 2>&1 &
```

## 4b. 决定 v7 生死的两个实验（数据充足时联邦不如纯本地，必须验证联邦真正有价值的 regime）

> 背景：720/客户端数据充足时，per-domain 个性化下 `local(0.627) > v7≈fedsa(0.622) > global(0.604)`，即**纯本地最好、池化 B 无增益**。联邦只有在两种 regime 才有价值：①数据稀缺（去噪）②冷启动（新客户端零样本）。这两个实验最便宜、最决定性。

### 4b.1 数据稀缺：联邦去噪（看 v7 是否反超 local）

把每客户端训练样本压到 30 / 50 / 100，本地 B 变噪，预期 v7 同域池化反超 local：

```bash
for CAP in 30 50 100; do
  CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/Analysis/eval_personalized.py \
    --model "$MODEL_PATH" --benchmark_dir "$BENCHMARK_DIR" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
    --max_steps 0 --eval_max_batches 30 --seed "$SEED" \
    --max_train_samples_per_client "$CAP" \
    --schemes local,fedsa,global,v7 --v7_b_mode mean \
    --out artifacts_35c/eval_personalized/eval_personalized_mean_cap${CAP}_seed${SEED}.json \
    > log_diag/eval_personalized_mean_cap${CAP}_seed${SEED}.log 2>&1
done
```

> 判读（脚本自动打印）：若 `v7 − local ≥ +0.003` 在某个 cap 出现 → 联邦去噪生效，v7 在数据稀缺下有真实价值。`for` 循环串行（无 `&`）。

### 4b.2 冷启动：新客户端零样本（`--cold_start`）

模拟"新机构无本地数据加入"：`base`=无适配下界，`coldstart`=同域其它客户端留一池化的 B（新客户端能零样本拿到的）：

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" --benchmark_dir "$BENCHMARK_DIR" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --max_steps 0 --eval_max_batches 30 --seed "$SEED" \
  --schemes local,fedsa,global,v7 --v7_b_mode mean --cold_start \
  --out artifacts_35c/eval_personalized/eval_personalized_coldstart_seed${SEED}.json \
  > log_diag/eval_personalized_coldstart_seed${SEED}.log 2>&1 &
```

> 判读：若 `coldstart − base > +0.01` → 同域池化 B 让新客户端**零样本**获得显著域能力（local 永远做不到，需要本地数据），这是联邦最硬的价值点。

## 4c. 域内非 IID 重构 + 冷启动复验（决定 v7 生死，必做）

> **为什么必做**：现有冷启动结果（coldstart 0.6220 ≈ local 0.6274）在很大程度上是"域内 IID"的产物——同域客户端同分布，peers 池化的 B 自然泛化，接近平凡。**只有当域内非 IID（新客户端子分布与 peers 不同）时，coldstart 仍接近 local，才证明 domain-B 抓到了可迁移的"域骨架技能"，v7 才是真贡献。** 这是整个故事真伪的命门。

### 4c.1 构建三套基准：IID 对照 + 非 IID(α=0.5) + 非 IID(α=0.1)

用 leak-free 构建器（`scripts/DataProcessScripts/build_domain_benchmark_v2.py`，已 dedup + K 一致 + 域内非 IID）。三套都从同一公开语料 JSONL 派生，**test_domain 保持域级 IID 干净**（评测公平），只有 train 的域内分配从 IID 变 Dirichlet。

```bash
export RAW_JSONL="$DATA_ROOT/raw/domain_7_all.jsonl"
ls "$RAW_JSONL"   # 确认存在；每行含 domain/prompt/response

# (a) IID 对照（与现有结论可比，且去泄漏/均衡）
nohup python -u scripts/DataProcessScripts/build_domain_benchmark_v2.py \
  --input_jsonl "$RAW_JSONL" \
  --output_dir "$DATA_ROOT/domain_benchmark_v2_iid" \
  --num_clients_per_domain 5 --dedup prompt --target_per_domain 2000 \
  --partition iid --seed 42 \
  > /data2/minghao/result/FedPLoRA/logs/test0618_build_v2_iid.log 2>&1 &

# (b) 中等域内非 IID（α=0.5）
nohup python -u scripts/DataProcessScripts/build_domain_benchmark_v2.py \
  --input_jsonl "$RAW_JSONL" \
  --output_dir "$DATA_ROOT/domain_benchmark_v2_niid_a05" \
  --num_clients_per_domain 5 --dedup prompt --target_per_domain 2000 \
  --partition dirichlet --dirichlet_alpha 0.5 --subtopic length --n_subtopics 10 --seed 42 \
  > /data2/minghao/result/FedPLoRA/logs/test0618_build_v2_niid_a05.log 2>&1 &

# (c) 强域内非 IID（α=0.1）
nohup python -u scripts/DataProcessScripts/build_domain_benchmark_v2.py \
  --input_jsonl "$RAW_JSONL" \
  --output_dir "$DATA_ROOT/domain_benchmark_v2_niid_a01" \
  --num_clients_per_domain 5 --dedup prompt --target_per_domain 2000 \
  --partition dirichlet --dirichlet_alpha 0.1 --subtopic length --n_subtopics 10 --seed 42 \
  > /data2/minghao/result/FedPLoRA/logs/test0618_build_v2_niid_a01.log 2>&1 &
```

构建日志末尾应打印 `[leakcheck] PASS — zero prompt-level leakage`。检查产物：

```bash
for d in iid niid_a05 niid_a01; do
  echo "== $d =="
  ls "$DATA_ROOT/domain_benchmark_v2_${d}/seed_42"/{clients.json,train.jsonl,test_domain.jsonl}
done
```

### 4c.2 三套基准各跑一遍冷启动评测

```bash
for TAG in iid niid_a05 niid_a01; do
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/Analysis/eval_personalized.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$DATA_ROOT/domain_benchmark_v2_niid_a01/seed_42" \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --max_steps 0 --eval_max_batches 30 --seed "$SEED" \
  --schemes local,fedsa,global,v7 --v7_b_mode mean --cold_start \
  --out artifacts_35c/eval_personalized/coldstart_niid_a01_seed${SEED}.json \
  > /data2/minghao/result/FedPLoRA/logs/test0618_coldstart_niid_a01_seed${SEED}.log 2>&1 &
done
```

> `for` 循环串行（无 `&`），一张卡顺序跑三套。若两卡并行：把 iid 放 GPU0、两个 niid 放 GPU1，各自加 `&`。

### 4c.3 决定性判读

每个 log 末尾看 `coldstart − base` 和 `coldstart vs local`：

| 现象（随 α 从 IID→0.5→0.1） | 结论 |
|---|---|
| coldstart 始终接近 local，`coldstart − base` 始终 > +0.015 | ✅ **domain-B 抓到可迁移域技能 → v7 真贡献，上 8B 写论文** |
| coldstart 随非 IID 增强**显著塌向 base** | ❌ **冷启动优势是 IID 假象 → v7 垮，回退到纯分析贡献或换方向** |
| coldstart 仍 > global B、但与 local 拉开差距 | 🟡 部分成立：domain-B 比 FedAvg 强但非全能，需 few-shot 曲线补强 |

对照查看三套结果：

```bash
for TAG in iid niid_a05 niid_a01; do
  echo "===== $TAG ====="
  grep -E "^base |^coldstart|^local |^global |判读|参考" log_diag/coldstart_${TAG}_seed${SEED}.log
done
```

### 4c.4 （若 4c.3 通过）few-shot onboarding 曲线

回应"新机构现实里不会一条数据都没有"。在非 IID(α=0.5) 上，给新客户端 0/8/32/128 条本地数据（用 `--max_train_samples_per_client` 近似各客户端数据量），看 domain-B 起点优势随本地数据增多如何收敛：

```bash
for CAP in 8 32 128; do
  CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/Analysis/eval_personalized.py \
    --model "$MODEL_PATH" \
    --benchmark_dir "$DATA_ROOT/domain_benchmark_v2_niid_a05/seed_42" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
    --max_steps 0 --eval_max_batches 30 --seed "$SEED" \
    --max_train_samples_per_client "$CAP" \
    --schemes local,fedsa,global,v7 --v7_b_mode mean --cold_start \
    --out artifacts_35c/eval_personalized/coldstart_niid_a05_cap${CAP}_seed${SEED}.json \
    > log_diag/coldstart_niid_a05_cap${CAP}_seed${SEED}.log 2>&1
done
```

> 理想曲线：本地数据=0 时 v7/coldstart 远超 local；随本地数据增多 local 追上来。这正是"零/少样本 onboarding"的卖点图。

## 5. 多 seed（定稿前）

把 `SEED` 改为 `1234`、`9999` 各重跑 §4.1（mean）一遍，报 mean ± std。例如：

```bash
for S in 42 1234 9999; do
  CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/Analysis/eval_personalized.py \
    --model "$MODEL_PATH" --benchmark_dir "$BENCHMARK_DIR" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
    --max_steps 0 --eval_max_batches 30 --seed "$S" \
    --schemes local,fedsa,global,v7 --v7_b_mode mean \
    --out artifacts_35c/eval_personalized/eval_personalized_mean_seed${S}.json \
    > log_diag/eval_personalized_mean_seed${S}.log 2>&1
done
```

> 注意：上面 `for` 循环去掉了 `&`，串行跑（一张卡上顺序执行）。若要并行，分别放到不同 GPU 并加 `&`。

## 6. 如何看结果（判读）

脚本末尾会打印每方案的 per-domain `macro_acc` / `worst_acc` / 各域 acc，并自动比较：

- **`v7 − global` macro/worst 为正（≥ +0.003）** → 个性化目标下 v7（域特异 B）反超通才 → **方法方向成立**，可考虑接入训练框架。
- **`v7 ≈ local`** → 域内 B 池化无损（甚至去噪有益），v7 不牺牲个性化。
- **`v7 ≈ global`、`local` 也不高** → 域内 IID 下池化收益有限；主打"A/B 非对称"分析图，或换更异质数据 / 更长训练。
- **`mean` 明显劣于 `rep`** → 同域 B 平均出现 gauge 抵消 → 改用 `rep` 或 `svd`。

查看结果：

```bash
tail -n 30 log_diag/eval_personalized_mean_seed${SEED}.log
cat artifacts_35c/eval_personalized/eval_personalized_mean_seed${SEED}.json
```

## 7. 日志与进程检查

```bash
tail -f log_diag/eval_personalized_mean_seed${SEED}.log
ps -ef | grep -E "eval_personalized|diag_subspace|diag_b_swap" | grep -v grep
nvidia-smi
```

## 8. 待办：把 v7 接入训练框架（当前为 standalone）

当前 v7 只在 `eval_personalized.py` 内做"捕获→聚合→per-domain 评测"，**未**成为 `tasks/fed_train_sft.py` 的 `agg_type`。若 §6 判读显示 v7 在 per-domain 上稳定优于 global，再实现：

- `methods/v7/hier_lora.py`（已就位）的 `aggregate_per_domain_B` 接入聚合 dispatch；
- `utilities/utils.py` 增加 `is_v7_agg`，并走 per-domain 个性化下发（复用 v4 `_fedplora_personalized_shared_states`，但需把 **B**（而非 A）按域下发）；
- `tasks/fed_train_sft.py` 增加 dispatch + per-domain 个性化 eval；
- 预期运行形式（**当前不可运行**，待实现）：

```bash
# 当前不可运行：仓库尚未实现 agg_type=v7_hier。
CUDA_VISIBLE_DEVICES=0 nohup python -u tasks/fed_train_sft.py \
  --model "$MODEL_PATH" --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v7_hier \
  --rounds 1 --local_epochs 1 --lr 2e-4 \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --batch_size "$BATCH_SIZE" --max_seq_length "$MAX_SEQ_LENGTH" \
  --torch_dtype "$TORCH_DTYPE" --target_modules "$TARGET_MODULES" \
  --client_state_dir artifacts_35c/v7_client_states/v7_hier \
  --metrics_output_dir artifacts_35c/sft_metrics_v7 \
  --eval_max_batches 30 --seed "$SEED" \
  --gradient_checkpointing --save_client_state_to_disk \
  --v7_b_mode mean \
  --eval_personalization_metrics \
  > log_diag/v7_hier_seed${SEED}.log 2>&1 &
```

## 9. 从快筛切到主验证

- 快筛：`MAX_SEQ_LENGTH=256`、`--eval_max_batches 30`、`--max_steps 0`、单 seed。
- 主验证：`MAX_SEQ_LENGTH=512`、`--eval_max_batches` 提到 `50`+、3 seed（42/1234/9999）、`--v7_b_mode` 三种都报。
- 关键比较：per-domain 下 `v7` vs `global/flora`（通才）与 `local/fedsa`（个性化）；重点看 **worst/hard 域**（legal/finance/medical）。
- 8B 只在小模型确认 v7 有稳定增益后，做关键确认点（v7 vs global 一组），符合算力约束。
