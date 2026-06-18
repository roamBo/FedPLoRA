# 标准（非跨域）联邦 SFT 实验

为论文补充 **通用、经典、标准** 的对比实验：在广泛使用的公开指令微调数据集上，以 **Dirichlet non-IID** 联邦划分验证各类 baseline，与仓库主线的 **7 域跨域 benchmark** 形成互补。

---

## 数据集与任务

| 项 | 选择 | 说明 |
|----|------|------|
| **数据集** | [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (`tatsu-lab/alpaca`) | 指令微调领域最常被引用的公开集之一（≈52k 条） |
| **任务** | **Instruction-following SFT**（监督微调） | 输入：`instruction` + 可选 `input`；输出：`output` |
| **划分** | **Dirichlet non-IID**（默认 **α=0.5**） | 对 instruction 做 TF-IDF + KMeans（**纯 numpy**，构建阶段无需 sklearn）得到伪标签，再按 GLUE 同款 `Dir(α)` 做 label skew；默认 10 客户端 |
| **评测** | 全局 held-out test（10%） | token accuracy、perplexity |
| **模型（全量）** | Llama-3.1-8B base | `configs/standard_sft.env` |
| **模型（LW）** | SmolLM2-135M base | `configs/lw_standard_noniid.env` |

**Non-IID 含义（与跨域实验对比）**

| | 跨域主实验 | 标准实验（本文档） |
|--|-----------|-------------------|
| 数据 | 自构建 7 域 benchmark | 公开 Alpaca |
| Non-IID 类型 | **域偏移**（task shift across domains） | **伪标签 Dirichlet skew**（α=0.5，同数据集内） |
| 入口 | `tasks/fed_train_sft.py` | `tasks/fed_train_standard_sft.py` |
| 产物目录 | `artifacts_{Nc}c/` | `artifacts_standard/` / `artifacts_LW_standard/` |

划分元数据写入 `partition_info.json`（含 `dirichlet_alpha`、`num_pseudo_labels`）；各客户端伪标签直方图见 `clients.json` 的 `pseudo_label_hist`。

---

## 一、构建数据集

### 全量（Llama，10 客户端，α=0.5）

```bash
bash scripts/DataProcessScripts/build_alpaca_standard_benchmark_noniid.sh 10 0.5
```

产物：`data/standard_benchmark_alpaca_noniid_a0.5/seed_42/`

### LW 轻量（7 客户端，≈5200 条，α=0.5）

```bash
bash scripts/DataProcessScripts/build_alpaca_lw_standard_noniid_benchmark.sh
# 可选: bash .../build_alpaca_lw_standard_noniid_benchmark.sh 5200 0.5
```

产物：`data/standard_benchmark_alpaca_LW_noniid_a0.5/seed_42/`

### 分步命令（全量）

```bash
# 1) 导出 Alpaca
python scripts/DataProcessScripts/prepare_alpaca_standard_data.py \
  --dataset tatsu-lab/alpaca \
  --domain alpaca \
  --output data/standard_sources/alpaca/alpaca.jsonl \
  --prompt_template "{instruction}\n\n{input}" \
  --response_template "{output}" \
  --shuffle --seed 42

# 2) Dirichlet non-IID 划分
python scripts/DataProcessScripts/build_standard_sft_benchmark.py \
  --input_jsonl data/standard_sources/alpaca/alpaca.jsonl \
  --output_dir data/standard_benchmark_alpaca_noniid_a0.5 \
  --num_clients 10 \
  --partition dirichlet \
  --dirichlet_alpha 0.5 \
  --domain_label alpaca \
  --seed 42
```

**可选 IID**（旧设定，round-robin）：

```bash
bash scripts/DataProcessScripts/build_alpaca_standard_benchmark.sh 10
# 或 --partition iid
```

---

## 二、训练入口

`tasks/fed_train_standard_sft.py` — 复用 `fed_train_sft.py` 训练循环；**禁止** FedPLoRA 族（用于跑对比 baseline）。

```bash
set -a && source configs/standard_sft.env && set +a

python tasks/fed_train_standard_sft.py \
  --model "${MODEL_PATH}" \
  --benchmark_dir data/standard_benchmark_alpaca_noniid_a0.5/seed_42 \
  --agg_type normal \
  --gradient_checkpointing \
  --eval_max_batches 50
```

---

## 三、Llama baseline 批量脚本（除 FedPLoRA 外）

```bash
bash scripts/RunScripts/run_standard_sft_baselines.sh 0
```

8 个方法：`normal`, `flora`, `flexlora`, `feddat`, `fedalt`, `yoco`, `fedsa_lora`, `ffa`

环境：`configs/standard_sft.env`  
指标：`artifacts_standard/sft_metrics/`

---

## 四、LW baseline 批量脚本（Alpaca non-IID α=0.5）

```bash
bash scripts/DataProcessScripts/build_alpaca_lw_standard_noniid_benchmark.sh
bash scripts/RunScripts/LWv4/download_lw_model_modelscope.sh
bash scripts/RunScripts/LWv4/run_lw_standard_noniid_baseline.sh 0
```

环境：`configs/lw_standard_noniid.env`  
指标：`artifacts_LW_standard/sft_metrics/`  
同样 8 个 baseline，**不含** FedPLoRA。

**FedPLoRA v2 + v4（SmolLM LW）**：

```bash
bash scripts/RunScripts/LWv4/run_lw_standard_fedplora_all.sh 0
```

共 17 个 run（v2 oneshot + v4 支线 A–F）；指标：`artifacts_LW_standard/v4_sft_metrics/`。

---

## 五、推荐论文表述

- **跨域**：自构建 7 域 benchmark → 域偏移 non-IID
- **标准**：Stanford Alpaca + Dirichlet α=0.5 + 10-client FL → 经典指令微调设定下的 label-skew non-IID

两条实验共用 LoRA 配置与评测口径，便于主表并列汇报。
