# FedPLoRA：面向跨域个性化联邦大模型的 LoRA 因子解耦方法

本仓库实现了一个面向联邦个性化微调的 LoRA 方法：**FedPLoRA**。  
核心思想是将 LoRA 更新写为：

\[
\Delta W_i = B_i A_i
\]

其中：

- `A`：表示可跨客户端共享的低秩公共子空间
- `B_i`：表示客户端或领域私有的个性化映射

在联邦训练中：

- 服务端只维护并聚合 `A_global`
- 客户端保留私有 `B_i`
- 上传时只传 `A_i`、任务头以及轻量的重要性统计
- `B_i` 不上传、不聚合、不广播

这比直接联邦平均整套 `A+B` 更适合个性化场景，也更符合“共享知识”和“私有知识”分离的直觉。

---

## 20260504：FedPLoRA-Oneshot，一次聚合下的跨域冲突个性化

### 1. 为什么需要 FedPLoRA-Oneshot

新增的 **FedPLoRA-Oneshot** 面向更强的通信约束：所有客户端只本地训练一次，只向服务端上传一次，服务端只完成一次聚合。  
它的目标不是替代多轮 FedPLoRA，而是在极低通信、低暴露面的场景下完成跨域个性化联邦 SFT。

该设定用于和两类强相关工作拉开边界：

- FedSA-LoRA 已经提出 `A` 学 general knowledge、`B` 学 client-specific knowledge，并多轮只共享 `A`。
- YOCO 已经提出 true one-shot federated LoRA，但它面向 MLLM，全局聚合 `A+B`，并将 `B` 作为全局一致性、`A` 作为个性化适应。
- FedPLoRA-Oneshot 的新问题是：在 LLM 跨域 SFT 中，只通信一次时，如何只聚合共享 `A` 子空间，同时让每个领域保留本地私有 `B_i`，避免跨域强冲突方向被硬平均。

推荐论文叙事：

> FedPLoRA-Oneshot studies communication-constrained cross-domain personalized federated instruction tuning. It performs one-shot A-only aggregation with private B adapters, and uses conflict-aware row gating to prevent negative transfer across domains.

### 2. 方法定义

每个客户端从同一个 LoRA 初始化出发，本地训练一次：

\[
\Delta W_i = B_i A_i
\]

训练结束后：

- 客户端上传 `A_i`
- 客户端上传 task head
- 客户端上传轻量 row importance
- 客户端不上传 `B_i`
- 服务端只聚合一次得到 `A_global`
- 每个客户端最终使用 `B_i A_global` 作为个性化模型

### 3. 与多轮 FedPLoRA 的区别

| 维度 | 多轮 FedPLoRA `gp_lora` | FedPLoRA-Oneshot `fedplora_oneshot` |
|---|---|---|
| 通信轮数 | 多轮 | 强制一轮 |
| 服务端状态 | 每轮更新 `A_global` | 一次性得到 `A_global` |
| 本地 B | 始终私有 | 始终私有 |
| 聚合对象 | `A` + task head + row importance | `A` + task head + row importance |
| 冲突处理 | 共识加权 + momentum + QR | 初始共享基对齐 + row conflict gate，默认不旋转 row basis |
| 适用场景 | 性能优先 | 极低通信、隐私暴露更少、跨域一次协作 |

### 4. 当前代码实现

新增入口：

- `--agg_type fedplora_oneshot`

新增核心代码：

- [utils.py](/Users/hawaiii/codex/FedPLoRA/FedPLoRA/utils.py)：新增 `is_fedplora_oneshot_agg`
- [fed_agg.py](/Users/hawaiii/codex/FedPLoRA/FedPLoRA/fed_agg.py)：新增 `aggregate_models_fedplora_oneshot`
- [train_eval.py](/Users/hawaiii/codex/FedPLoRA/FedPLoRA/train_eval.py)：新增 one-shot 专用正则权重
- [fed_train_sft.py](/Users/hawaiii/codex/FedPLoRA/FedPLoRA/fed_train_sft.py)：接入跨域 SFT 主实验
- [fed_train_glue.py](/Users/hawaiii/codex/FedPLoRA/FedPLoRA/fed_train_glue.py)：接入 GLUE sanity check
- [fed_train_e2e.py](/Users/hawaiii/codex/FedPLoRA/FedPLoRA/fed_train_e2e.py)：接入 E2E-NLG sanity check

FedPLoRA-Oneshot 的聚合逻辑：

1. 服务端保存初始共享 `A_0` 作为唯一全局坐标系。
2. 客户端本地训练 `A_i` 和私有 `B_i`。
3. 上传包只包含 `A_i`、task head 和 row importance。
4. 服务端按 `A_i` 与 `A_0` 的方向一致性进行符号对齐。
5. 对每个 rank row 计算跨客户端 conflict score。
6. 低冲突 row 做重要性感知加权平均。
7. 高冲突 row 向初始共享 `A_0` 回退，避免强行合并跨域私有方向。
8. 默认保留 row 坐标和 row 尺度，避免一次聚合后 `B_i` 与 `A_global` 不兼容；可选打开 QR 正交化做消融。

### 5. 关键参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--agg_type fedplora_oneshot` | - | 启用 FedPLoRA-Oneshot |
| `--rounds` | 自动强制为 `1` | one-shot 只允许一次通信 |
| `--oneshot_align_lambda` | `0.02` | 约束 `B_i A_i` 与 `B_i A_0` 兼容 |
| `--oneshot_prox_lambda` | `0.002` | 约束 `A_i` 不过度偏离初始共享基 |
| `--oneshot_orth_lambda` | `1e-4` | 约束 `A_i` row basis 稳定 |
| `--oneshot_consensus_power` | `2.0` | row conflict gate 的锐度 |
| `--oneshot_conflict_threshold` | `0.35` | 超过该值的 row 视为高冲突方向 |
| `--oneshot_no_keep_init_on_conflict` | false | 关闭高冲突 row 向 `A_0` 回退 |
| `--oneshot_orthogonalize` | false | 打开服务端 QR 正交化消融，默认关闭以保持 `B_i/A` 坐标兼容 |

### 6. 最小可运行命令

先进入项目目录：

```bash
cd /Users/hawaiii/codex/FedPLoRA/FedPLoRA
```

如果已有统一七域数据 `data/raw/domain_7_all.jsonl`，先构建 benchmark：

```bash
python scripts/build_domain_benchmark.py \
  --input_jsonl data/raw/domain_7_all.jsonl \
  --output_dir data/domain_benchmark \
  --num_clients_per_domain 5 \
  --min_samples_per_client 50 \
  --seed 42
```

快速跑通 FedPLoRA-Oneshot：

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora_oneshot \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --save_client_state_to_disk
```

如果先做小模型 sanity check：

```bash
python fed_train_glue.py \
  --task rte \
  --model roberta-base \
  --agg_type fedplora_oneshot \
  --rounds 1 \
  --num_clients 3 \
  --partition dirichlet \
  --dirichlet_alpha 0.5 \
  --local_epochs 3 \
  --batch_size 32 \
  --lr 1e-3
```

### 7. 必须增加的对照实验

FedPLoRA-Oneshot 不能只和多轮方法比较，必须包含 one-shot 对照：

| 方法 | 命令中的 `--agg_type` | 目的 |
|---|---|---|
| Local LoRA | 后续可加 `local` 或用不聚合脚本 | 零通信下界 |
| One-shot FedAvg-LoRA | `normal --rounds 1` | 一次聚合完整 `A+B` |
| One-shot FedEx-LoRA | `fedex --rounds 1` | 一次聚合完整 LoRA 并吸收残差 |
| One-shot FedSA-LoRA | 后续建议加 `fedsa_lora` | 最强 A-only 直接基线 |
| YOCO-style | 后续建议加 `yoco` | true one-shot LoRA 强相关基线 |
| FedPLoRA-Oneshot | `fedplora_oneshot --rounds 1` | 本方法 |
| Multi-round FedPLoRA | `gp_lora --rounds 10/20` | 性能上界 |

推荐 baseline 命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type normal \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedex \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type gp_lora \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --save_client_state_to_disk
```

### 8. 消融实验命令

关闭冲突回退：

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora_oneshot \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --oneshot_no_keep_init_on_conflict
```

打开服务端正交化消融：

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora_oneshot \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --oneshot_orthogonalize
```

关闭 one-shot 本地兼容性正则：

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora_oneshot \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --oneshot_align_lambda 0 \
  --oneshot_prox_lambda 0 \
  --oneshot_orth_lambda 0
```

### 9. 当前方法的投稿边界

可以强调：

- 第一个系统研究 **one-shot cross-domain personalized federated LLM SFT** 的 LoRA 因子解耦方法。
- 不 claim 第一个 A-only FedLoRA，因为 FedSA-LoRA 已经覆盖。
- 不 claim 第一个 true one-shot FedLoRA，因为 YOCO 已经覆盖。
- 重点 claim：一次通信下，`A-only`、`B-private`、跨域 row conflict gating 三者结合，用于领域个性化而不是单一全局模型。

---

## 一、当前仓库已经支持什么

当前仓库包含三条能力线：

1. **GLUE 联邦分类**
   - 文件：[fed_train_glue.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/fed_train_glue.py)
   - 支持方法：`normal`、`ffa`、`fedex`、`gp_lora`
   - 支持 IID 和 Dirichlet non-IID 划分

2. **E2E-NLG 联邦生成**
   - 文件：[fed_train_e2e.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/fed_train_e2e.py)
   - 适合做小规模生成 sanity check

3. **面向 7 域 benchmark 的联邦 SFT 新链路**
   - 新增文件：[fed_train_sft.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/fed_train_sft.py)
   - 支持从统一 JSONL 自动构建 benchmark，再直接启动 causal LLM 联邦训练
   - 支持 `Qwen`、`Mistral`、`Llama`、`Gemma` 等 Hugging Face 因果语言模型
   - 已支持大模型场景下的 **sequential client training**
     即单次只在 GPU 上保留一份模型，客户端本地 `B` 状态保存在 CPU 或磁盘中，更适合 `2 x A100 80G`

---

## 二、方法概述

### 2.1 FedPLoRA 的目标

对每个客户端 `i`，个性化参数化形式为：

\[
W_i = W_0 + B_i A_{global}
\]

其中：

- `W_0`：冻结的预训练大模型参数
- `A_global`：跨客户端共享的全局低秩基
- `B_i`：客户端本地保留的私有映射

### 2.2 本地训练目标

为了避免 `A/B` 因子塌缩，FedPLoRA 在本地训练时加入三类正则：

1. **对齐损失 `align`**
   约束 `B_i A_i` 与 `B_i A_global` 的方向一致

2. **近端损失 `prox`**
   约束 `A_i` 不要偏离当前轮广播的 `A_global`

3. **正交损失 `orth`**
   保持 `A_i` 的行向量构成稳定基

当前实现位置：

- 本地正则：[train_eval.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/train_eval.py)
- 上传包构造：[fed_agg.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/fed_agg.py)
- 服务端聚合：[fed_agg.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/fed_agg.py)

### 2.3 服务端聚合

当前聚合逻辑为：

1. 客户端上传 `A_i`
2. 客户端上传每一行的轻量重要性统计
3. 服务端按：
   - 客户端样本量
   - 行重要性
   - 与上一轮 `A_global` 的共识程度
   进行加权融合
4. 服务端对融合结果做正交化，得到新一轮 `A_global`

---

## 三、论文定位

- **面向领域个性化联邦大模型的因子解耦 PEFT 框架**
- **显式分离跨域共享知识与领域私有知识的联邦 LoRA 方法**
- **在 7 个专业域上系统评估个性化联邦指令微调**

推荐主线：

1. 跨域个性化联邦大模型需要显式区分“共享知识”和“私有知识”
2. LoRA 天然提供了一个低秩因子分解视角
3. 将共享知识放在 `A` 中、私有知识保存在 `B_i` 中更合理
4. `A-only` 上传同时带来：
   - 更低通信开销
   - 更强个性化
   - 更低私有域泄露风险

---

## 四、推荐主实验：7 域个性化联邦 SFT

推荐将客户端按以下 7 个域划分：

- `general`
- `math`
- `code`
- `medical`
- `legal`
- `finance`
- `education`

推荐语义解释：

- `general`：通用指令跟随锚点域
- `math` / `code`：能力型专业域
- `medical` / `legal` / `finance`：高风险垂直域
- `education`：教学与反馈型个性化域

这个设定比普通 Dirichlet non-IID 更好讲，因为它更接近真实联邦场景：  
不同机构、不同业务线、不同专业群体共同微调一个基础大模型，但不愿共享完整私有数据与私有适配器。

---

## 五、推荐模型

咱有两张 A100 80G，可以用模型如下：

### 5.1 用于快速消融与调参

- `meta-llama/Meta-Llama-3.1-8B`
- `Qwen/Qwen3-14B`

### 5.2 用于主结果

- `Qwen/Qwen3-32B`
- `mistralai/Mistral-Small-24B-Instruct-2501`

### 5.3 可选扩展

- `google/gemma-3-27b-it`

建议：

- 先在 `8B/14B` 上完成调参和消融
- 再在 `24B/32B` 上做主结果
- 不建议一开始就上 `70B+`

---

## 六、推荐数据集

建议用“一个总 JSONL 接口 + 多领域原始数据集”的方式组织。

### 6.1 七个领域推荐训练集

| 领域 | 主训练集 | 备选/补充 |
|---|---|---|
| `general` | `allenai/tulu-3-sft-mixture` | `teknium/OpenHermes-2.5`, `Open-Orca/SlimOrca` |
| `math` | `AI-MO/NuminaMath-CoT` | `meta-math/MetaMathQA` |
| `code` | `OpenCoder-LLM/opc-sft-stage1` | `bigcode/self-oss-instruct-sc2-exec-filter-50k` |
| `medical` | `FreedomIntelligence/medical-o1-reasoning-SFT` | 领域 QA 指令数据 |
| `legal` | `lawinstruct/lawinstruct` | 法律 QA / StackExchange 风格数据 |
| `finance` | `gbharti/finance-alpaca` | `FinQA` 指令化版本 |
| `education` | `eth-nlped/mathdial` + `ScaleAI/TutorBench` | tutoring 对话数据 |

当前仓库已经提供对应的数据准备入口：

- [prepare_hf_domain_data.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/prepare_hf_domain_data.py)
- [prepare_general_data.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/prepare_general_data.py)
- [prepare_math_data.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/prepare_math_data.py)
- [prepare_code_data.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/prepare_code_data.py)
- [prepare_medical_data.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/prepare_medical_data.py)
- [prepare_legal_data.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/prepare_legal_data.py)
- [prepare_finance_data.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/prepare_finance_data.py)
- [prepare_education_data.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/prepare_education_data.py)
- [prepare_all_domains.sh](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/prepare_all_domains.sh)
- [domain_data_pilot.env](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/configs/domain_data_pilot.env)

### 6.2 推荐评测集

| 领域 | 推荐评测 |
|---|---|
| `general` | `IFEval`, `AlpacaEval 2` |
| `math` | `GSM8K`, `MATH-500` |
| `code` | `HumanEval`, `MBPP` |
| `medical` | `MedQA` |
| `legal` | `LegalBench` 子集 |
| `finance` | `FinQA`, `FiQA` |
| `education` | `TutorBench` 或 tutoring-style 测试集 |

---

## 七、统一数据接口

为了让“下载数据后直接运行”，本仓库统一采用 **JSONL benchmark 接口**。

### 7.1 原始统一 JSONL 格式

你需要先把不同来源的数据转成下面格式：

```json
{"domain":"math","prompt":"请解方程 x^2-5x+6=0","response":"方程可因式分解为 ...","source_id":"numina_0001","metadata":{"dataset":"AI-MO/NuminaMath-CoT"}}
{"domain":"code","prompt":"写一个 Python 函数反转链表","response":"```python\nclass Solution: ...\n```","source_id":"opc_0002","metadata":{"dataset":"OpenCoder-LLM/opc-sft-stage1"}}
```

必须字段：

- `domain`
- `prompt`
- `response`

可选字段：

- `source_id`
- `metadata`

### 7.2 自动构建 benchmark 的输出

执行 benchmark 构建后，会自动生成：

```text
data/
  domain_benchmark/
    seed_42/
      clients.json
      domain_stats.json
      train.jsonl
      val.jsonl
      test_local.jsonl
      test_domain.jsonl
      test_global.jsonl
```

其中：

- `train.jsonl`：所有客户端训练样本，带 `client_id`
- `val.jsonl`：客户端验证集
- `test_local.jsonl`：客户端本地测试集
- `test_domain.jsonl`：按域汇总的领域测试集
- `test_global.jsonl`：混合全局测试集

### 7.3 已提供的数据准备与 benchmark 脚本

仓库内已补充三个直接可用的脚本：

- 模板生成脚本：[scripts/prepare_domain_jsonl_template.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/prepare_domain_jsonl_template.py)
- 多域汇总脚本：[scripts/merge_domain_jsonl.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/merge_domain_jsonl.py)
- benchmark 构建脚本：[scripts/build_domain_benchmark.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/build_domain_benchmark.py)

先生成一个统一 JSONL 模板：

```bash
cd /Users/hawaiii/yao/FedPLoRA/FedPLoRA
python scripts/prepare_domain_jsonl_template.py \
  --output data/raw/domain_7_template.jsonl \
  --examples_per_domain 2
```

然后把你实际下载和清洗后的数据写入：

```text
data/raw/domain_7_all.jsonl
```

如果你已经把各领域样本按目录存放，也可以先自动汇总：

```bash
python scripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

如果你希望直接从 Hugging Face 数据集导出一个 pilot 版本，也可以执行：

```bash
cd /Users/hawaiii/yao/FedPLoRA/FedPLoRA
source configs/domain_data_pilot.env
bash scripts/prepare_all_domains.sh
python scripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

再独立构建 benchmark：

```bash
python scripts/build_domain_benchmark.py \
  --input_jsonl data/raw/domain_7_all.jsonl \
  --output_dir data/domain_benchmark \
  --num_clients_per_domain 5 \
  --min_samples_per_client 50 \
  --seed 42
```

---

## 八、环境安装

### 8.1 创建环境

```bash
cd /Users/hawaiii/yao/FedPLoRA/FedPLoRA
conda create -n fedplora python=3.10 -y
conda activate fedplora
pip install -r requirements.txt
```

### 8.2 建议额外安装

用于大模型训练和扩展实验：

```bash
pip install bitsandbytes trl sentence-transformers
```

### 8.3 语法检查

```bash
python -m py_compile \
  fed_train_glue.py \
  fed_train_e2e.py \
  fed_train_sft.py \
  train_eval.py \
  fed_agg.py \
  models.py \
  data_utils.py \
  run_script.py \
  scripts/prepare_domain_jsonl_template.py \
  scripts/build_domain_benchmark.py \
  scripts/merge_domain_jsonl.py \
  scripts/prepare_hf_domain_data.py \
  scripts/prepare_general_data.py \
  scripts/prepare_math_data.py \
  scripts/prepare_code_data.py \
  scripts/prepare_medical_data.py \
  scripts/prepare_legal_data.py \
  scripts/prepare_finance_data.py \
  scripts/prepare_education_data.py
```

---

## 九、当前可直接运行的 sanity 实验

### 9.1 GLUE sanity：IID

```bash
CUDA_VISIBLE_DEVICES=0 python fed_train_glue.py \
  --model roberta-base \
  --task cola \
  --agg_type gp_lora \
  --num_clients 3 \
  --lora_r 4 \
  --rounds 10 \
  --lr 1e-3 \
  --local_epochs 3 \
  --partition iid
```

### 9.2 GLUE sanity：Dirichlet non-IID

```bash
CUDA_VISIBLE_DEVICES=0 python fed_train_glue.py \
  --model roberta-base \
  --task cola \
  --agg_type gp_lora \
  --num_clients 3 \
  --lora_r 4 \
  --rounds 10 \
  --lr 1e-3 \
  --local_epochs 3 \
  --partition dirichlet \
  --dirichlet_alpha 0.1 \
  --print_partition_stats \
  --pfl_eval_split client_val
```

### 9.3 E2E-NLG 生成 sanity

```bash
CUDA_VISIBLE_DEVICES=0 python fed_train_e2e.py \
  --agg_type gp_lora \
  --rounds 6 \
  --num_clients 3 \
  --local_epochs 5 \
  --lr 2e-3 \
  --lora_r 4 \
  --batch_size 8 \
  --log
```

这些实验用于：

- 检查方法实现是否正常
- 检查日志、通信统计、聚合过程是否正常
- 不作为主实验

---

## 十、7 域 benchmark 的构建步骤

本仓库已经补上了最小 benchmark 构建接口，位于：

- [data_utils.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/data_utils.py)
- [fed_train_sft.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/fed_train_sft.py)
- [scripts/merge_domain_jsonl.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/merge_domain_jsonl.py)
- [scripts/build_domain_benchmark.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/build_domain_benchmark.py)

### 10.1 第一步：准备统一 JSONL

将各领域样本整理到一个文件，例如：

```text
data/raw/domain_7_all.jsonl
```

如果你还没有开始整理数据，可以先生成模板：

```bash
python scripts/prepare_domain_jsonl_template.py \
  --output data/raw/domain_7_template.jsonl \
  --examples_per_domain 2
```

如果你已经将每个域的文件分别放在 `data/domain_sources/<domain>/` 目录，也可以直接合并：

```bash
python scripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

推荐先人工检查每个领域至少 100 条样本，确保：

- `prompt/response` 非空
- 域标签正确
- 没有大量格式损坏

### 10.2 第二步：自动切成 benchmark

```bash
python scripts/build_domain_benchmark.py \
  --input_jsonl data/raw/domain_7_all.jsonl \
  --output_dir data/domain_benchmark \
  --num_clients_per_domain 5 \
  --min_samples_per_client 50 \
  --seed 42
```

推荐产物检查：

```bash
ls data/domain_benchmark/seed_42
cat data/domain_benchmark/seed_42/domain_stats.json
cat data/domain_benchmark/seed_42/clients.json | head
```

---

## 十一、7 域联邦 SFT 主实验入口

新增主入口：

- [fed_train_sft.py](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/fed_train_sft.py)

它支持两种模式：

1. **直接读取 benchmark**
2. **从原始 JSONL 自动构建 benchmark 再训练**

### 11.1 直接用准备好的 benchmark 训练

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type gp_lora \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
  --save_client_state_to_disk
```

说明：

- `fed_train_sft.py` 现在默认按 **顺序客户端训练** 工作
- 对 `gp_lora` 而言，GPU 上只保留一份模型；每个 client 的本地 `B` 保存在 CPU 或磁盘中
- `--save_client_state_to_disk` 更适合 `35~70 clients` 的主实验
- 如果模型需要自定义代码，可以加 `--trust_remote_code`

### 11.2 从原始 JSONL 直接构建并训练

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --build_benchmark \
  --benchmark_jsonl data/raw/domain_7_all.jsonl \
  --benchmark_output_dir data/domain_benchmark \
  --num_clients_per_domain 5 \
  --min_samples_per_client 50 \
  --agg_type gp_lora \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
  --save_client_state_to_disk
```

### 11.3 推荐的一键脚本

已补充主实验脚本：

- [scripts/run_domain_sft.sh](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/run_domain_sft.sh)
- [configs/domain_sft_pilot.env](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/configs/domain_sft_pilot.env)

使用方式：

```bash
cd /Users/hawaiii/yao/FedPLoRA/FedPLoRA
source configs/domain_sft_pilot.env
bash scripts/run_domain_sft.sh
```

训练结束后，每轮 `domain_macro_loss` 会自动保存到：

```text
artifacts/sft_metrics/
```

你也可以临时覆盖变量：

```bash
MODEL_PATH=Qwen/Qwen3-32B \
AGG_TYPE=gp_lora \
CUDA_DEVICES=0,1 \
ROUNDS=20 \
BATCH_SIZE=1 \
bash scripts/run_domain_sft.sh
```

### 11.4 运行 baseline

已补充批量 baseline 脚本：

- [scripts/run_domain_sft_baselines.sh](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/scripts/run_domain_sft_baselines.sh)
- [configs/domain_sft_baselines.env](/Users/hawaiii/yao/FedPLoRA/FedPLoRA/configs/domain_sft_baselines.env)

```bash
source configs/domain_sft_baselines.env
bash scripts/run_domain_sft_baselines.sh
```

#### `normal`

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type normal \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
```

#### `ffa`

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type ffa \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
```

#### `fedex`

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedex \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
```

---

## 十二、模型下载接口

本仓库默认直接使用 Hugging Face 模型名作为接口，因此只要模型已经登录并可访问，就可以直接运行。

例如：

```bash
--model Qwen/Qwen3-14B
--model Qwen/Qwen3-32B
--model mistralai/Mistral-Small-24B-Instruct-2501
--model meta-llama/Meta-Llama-3.1-8B
```

如果你已经把模型下载到本地，也可以直接传本地路径：

```bash
--model /path/to/local/model_dir
```

如需预先下载，可用：

```bash
huggingface-cli download Qwen/Qwen3-14B --local-dir /path/to/models/Qwen3-14B
```

如果模型需要认证：

```bash
huggingface-cli login
```

然后运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model /path/to/models/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type gp_lora \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --save_client_state_to_disk
```

---

## 十三、数据下载接口

本仓库对原始数据的要求只有一个：  
**最终整理成统一 JSONL 格式。**

你可以：

1. 直接从 Hugging Face 下载原始数据集
2. 自己写数据清洗脚本
3. 输出为 `data/raw/domain_7_all.jsonl`
4. 然后调用本仓库自动切 benchmark

建议目录：

```text
data/
  raw/
    domain_7_all.jsonl
  domain_benchmark/
    seed_42/
```

当前仓库已经留好了统一 JSONL 接口，因此你只需要把不同来源的数据清洗到：

```text
data/raw/domain_7_all.jsonl
```

推荐的数据组织方式：

```text
data/domain_sources/
  general/
  math/
  code/
  medical/
  legal/
  finance/
  education/
```

每个域目录下可以放多个 `.json` 或 `.jsonl` 文件，然后统一执行：

```bash
python scripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

现在仓库已经补上：

- `scripts/prepare_general_data.py`
- `scripts/prepare_math_data.py`
- `scripts/prepare_code_data.py`
- `scripts/prepare_medical_data.py`
- `scripts/prepare_legal_data.py`
- `scripts/prepare_finance_data.py`
- `scripts/prepare_education_data.py`
- `scripts/prepare_hf_domain_data.py`
- `scripts/prepare_all_domains.sh`

如果某个数据集字段与默认规则不一致，可以在对应脚本里增加：

- `--prompt_field`
- `--response_field`
- `--input_field`
- `--messages_field`
- `--prompt_template`
- `--response_template`

经验上：

- 单轮问答 / 指令数据优先用 `--prompt_field` + `--response_field`
- 带额外上下文的数据可加 `--input_field`
- 多轮对话数据优先用 `--messages_field`
- 字段比较散乱时，直接用 `--prompt_template` 和 `--response_template` 最稳
- `--prompt_template`
- `--response_template`

---

## 十四、推荐实验列表

为了支撑顶会论文，建议按如下顺序推进。

### E1. GLUE sanity

目标：

- 验证方法逻辑
- 验证通信统计
- 验证 `gp_lora` 行为正常

### E2. E2E-NLG sanity

目标：

- 验证生成侧可用
- 检查因果 LM 路线没有逻辑错误

### E3. 7 域主实验

目标：

- 作为主结果表
- 比较 `gp_lora` 与 `normal/ffa/fedex`

建议：

- `7 domains x 5 clients = 35 clients` 作为 pilot
- `7 domains x 10 clients = 70 clients` 作为主实验

### E4. 跨模型验证

模型建议：

- `Qwen3-32B`
- `Mistral-Small-24B`

### E5. 个性化收益分析

比较：

- 本域测试
- 跨域测试

目标：

- 证明个性化不是只提高平均分，而是提高 in-domain specialization

### E6. 通信-性能实验

比较：

- `A+B` 全上传
- `A-only`
- `B-only`
- local-only

目标：

- 证明 FedPLoRA 在通信-性能 Pareto 上更优

### E7. 机制消融

消融项：

- 去掉 `align`
- 去掉 `prox`
- 去掉 `orth`
- 去掉共识加权
- 去掉 server momentum

### E8. 6 域 hard setting

去掉 `general` 域，证明方法不是靠通用锚点域撑住全局

### E9. 分层异质性

在领域内部再分：

- `math`：按难度/题型
- `code`：按语言/任务
- `medical/legal/finance`：按子主题

### E10. 不均衡与参与率鲁棒性

扫描：

- `10% / 20% / 50%` partial participation
- client size long-tail
- 不同 `local_epochs`

### E11. Transfer matrix

构建 `7 x 7` 域迁移矩阵，分析负迁移

### E12. 上传载荷泄露代理实验

比较：

- 上传 `A+B`
- 上传 `A-only`

看看能否从上传参数推测 client/domain

---

## 十五、详细实验步骤与命令

### 阶段 0：环境准备

```bash
cd /Users/hawaiii/yao/FedPLoRA/FedPLoRA
conda activate fedplora
pip install -r requirements.txt
python -m py_compile \
  fed_train_glue.py fed_train_e2e.py fed_train_sft.py \
  train_eval.py fed_agg.py models.py data_utils.py run_script.py \
  scripts/prepare_domain_jsonl_template.py scripts/build_domain_benchmark.py \
  scripts/merge_domain_jsonl.py
```

### 阶段 1：GLUE sanity

```bash
CUDA_VISIBLE_DEVICES=0 python fed_train_glue.py \
  --model roberta-base \
  --task cola \
  --agg_type gp_lora \
  --num_clients 3 \
  --lora_r 4 \
  --rounds 10 \
  --lr 1e-3 \
  --local_epochs 3 \
  --partition dirichlet \
  --dirichlet_alpha 0.1 \
  --print_partition_stats
```

### 阶段 2：E2E sanity

```bash
CUDA_VISIBLE_DEVICES=0 python fed_train_e2e.py \
  --agg_type gp_lora \
  --rounds 6 \
  --num_clients 3 \
  --local_epochs 5 \
  --lr 2e-3 \
  --lora_r 4 \
  --batch_size 8 \
  --log
```

### 阶段 3：准备 7 域统一 JSONL

目标文件：

```text
data/raw/domain_7_all.jsonl
```

可先生成模板：

```bash
python scripts/prepare_domain_jsonl_template.py \
  --output data/raw/domain_7_template.jsonl \
  --examples_per_domain 2
```

或者从按域存放的文件自动汇总：

```bash
python scripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

如果要直接从 Hugging Face 数据集准备一个 pilot：

```bash
source configs/domain_data_pilot.env
bash scripts/prepare_all_domains.sh
python scripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

### 阶段 4：自动构建 benchmark

```bash
python scripts/build_domain_benchmark.py \
  --input_jsonl data/raw/domain_7_all.jsonl \
  --output_dir data/domain_benchmark \
  --num_clients_per_domain 5 \
  --min_samples_per_client 50 \
  --seed 42
```

### 阶段 5：Pilot 主实验

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type gp_lora \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
  --save_client_state_to_disk \
  --gp_align_lambda 0.01 \
  --gp_prox_lambda 0.001 \
  --gp_orth_lambda 1e-4 \
  --gp_consensus_power 2.0 \
  --gp_agg_momentum 0.5
```

或者直接使用脚本：

```bash
source configs/domain_sft_pilot.env
bash scripts/run_domain_sft.sh
```

训练指标会自动落到：

```text
artifacts/sft_metrics/
```

### 阶段 6：Pilot baseline

可直接批量运行：

```bash
source configs/domain_sft_baselines.env
bash scripts/run_domain_sft_baselines.sh
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type normal \
  --rounds 1 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type ffa \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedex \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16
```

### 阶段 7：主模型实验

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-32B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type gp_lora \
  --rounds 20 \
  --local_epochs 1 \
  --lr 1e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 1 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --save_client_state_to_disk
```

### 阶段 8：跨模型验证

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model mistralai/Mistral-Small-24B-Instruct-2501 \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type gp_lora \
  --rounds 20 \
  --local_epochs 1 \
  --lr 1e-4 \
  --lora_r 8 \
  --batch_size 1 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --save_client_state_to_disk
```

---

## 十六、建议汇报指标

主实验建议至少报告：

1. client-macro 平均
2. domain-macro 平均
3. worst-domain
4. worst-client
5. 总上传字节数
6. 总通信量
7. 达到目标性能所需轮数
8. 不同域的收敛曲线

---

## 十七、建议表格与图

### 表 1：7 域主结果

列：

- 方法
- client-macro
- domain-macro
- worst-domain
- 通信量

### 表 2：跨模型结果

列：

- 方法
- Qwen3-32B
- Mistral-24B

### 表 3：消融实验

行：

- full
- w/o align
- w/o prox
- w/o orth
- w/o consensus
- w/o momentum

### 表 4：通信-性能结果

列：

- 方法
- 每轮上传字节
- 总通信量
- 最终得分

### 图建议

- 图 1：方法框架图
- 图 2：7 域 benchmark 构建流程
- 图 3：通信-性能 Pareto 曲线
- 图 4：各域收敛曲线
- 图 5：7x7 transfer matrix

---

## 十八、当前代码与后续可扩展点

当前已经补上的关键功能：

- 面向 domain benchmark 的 JSONL 构建
- 面向 causal LLM 的联邦 SFT 训练入口
- 支持 `gp_lora`、`normal`、`ffa`、`fedex`

后续建议继续补：

1. `scripts/prepare_*.py`
   - 自动下载并清洗 7 个领域数据

2. 更细粒度评测脚本
   - `eval_math.py`
   - `eval_code.py`
   - `eval_medical.py`
   - `eval_legal.py`
   - `eval_finance.py`
   - `eval_education.py`

3. baseline 复现
   - `FDLoRA`
   - `FlexLoRA`
   - personalized FL baseline

4. payload leakage 分析脚本

---

## 十九、最小可运行路径

如果你希望今天就跑通一版，最短路径是：

1. 准备 `data/raw/domain_7_all.jsonl`
2. 执行 benchmark 自动构建
3. 用 `Qwen/Qwen3-14B` 跑 `gp_lora`
4. 再跑 `normal`、`ffa`、`fedex`

对应命令：

```bash
cd /Users/hawaiii/yao/FedPLoRA/FedPLoRA
conda activate fedplora
pip install -r requirements.txt
```

```bash
python scripts/build_domain_benchmark.py \
  --input_jsonl data/raw/domain_7_all.jsonl \
  --output_dir data/domain_benchmark \
  --num_clients_per_domain 5 \
  --min_samples_per_client 50 \
  --seed 42
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type gp_lora \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --save_client_state_to_disk
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type normal \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing
```

跑通这两条之后，你就已经有：

- benchmark 产物
- FedPLoRA 主实验入口
- baseline 对照入口
- 可继续扩展的完整骨架
