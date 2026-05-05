(base) random-bo@LAPTOP-NJ0CUBJA:/mnt/f/HuggingfaceModels/models/qwen3-14b$ find . -type f -print0 | sort -z | xargs -0 sha256sum > qwen3-14b.sha256
(base) random-bo@LAPTOP-NJ0CUBJA:/mnt/f/HuggingfaceModels/models/qwen3-14b$ diff qwen3-14b.sha256 ../../qwen3-14b.sha256 

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

## 一、当前仓库已经支持什么

### 工程目录说明（与 `data/` 同级）

- **`methods/`**：各 baseline 的服务端聚合与（部分）上传逻辑，**每个方法一个 `.py` 文件**；不含 `__init__.py`（使用隐式命名空间包，便于 `from methods.xxx import ...`）。
- **`tasks/`**：可执行入口：`fed_train_sft.py`（域 SFT）、`fed_train_glue.py`、`fed_train_e2e.py`。
- **`utilities/`**：`data_utils.py`、`models.py`、`utils.py`、训练/评估 `train_eval.py`、联邦状态 `state_dict_ops.py`。
- **`scripts/`**：数据准备、`build_domain_benchmark.py`、批量实验入口 **`run_script.py`**。
- **`__pycache__/`**：Python 解释器自动生成的字节码缓存，**可删**；运行后会再次出现，已写入 `.gitignore` 建议不要提交。

当前仓库包含三条能力线：

1. **GLUE 联邦分类**
   - 文件：[tasks/fed_train_glue.py](tasks/fed_train_glue.py)
   - 支持方法：`normal`、`ffa`、`fedex`、`fedplora`
   - 支持 IID 和 Dirichlet non-IID 划分

2. **E2E-NLG 联邦生成**
   - 文件：[tasks/fed_train_e2e.py](tasks/fed_train_e2e.py)
   - 适合做小规模生成 sanity check

3. **面向 7 域 benchmark 的联邦 SFT 新链路**
   - 新增文件：[tasks/fed_train_sft.py](tasks/fed_train_sft.py)
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

- 本地正则 / 训练步：[utilities/train_eval.py](utilities/train_eval.py)
- 上传包构造（FedP-LoRA 族）：[methods/fedp_lora.py](methods/fedp_lora.py)
- 各 baseline 服务端聚合：`methods/` 下对应方法文件（如 `fedavg_normal.py`、`fedsa_lora.py`、`yoco.py` 等）

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

- [prepare_hf_domain_data.py](scripts/prepare_hf_domain_data.py)
- [prepare_general_data.py](scripts/prepare_general_data.py)
- [prepare_math_data.py](scripts/prepare_math_data.py)
- [prepare_code_data.py](scripts/prepare_code_data.py)
- [prepare_medical_data.py](scripts/prepare_medical_data.py)
- [prepare_legal_data.py](scripts/prepare_legal_data.py)
- [prepare_finance_data.py](scripts/prepare_finance_data.py)
- [prepare_education_data.py](scripts/prepare_education_data.py)
- [prepare_all_domains.sh](scripts/prepare_all_domains.sh)
- [domain_data_pilot.env](configs/domain_data_pilot.env)

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

- 模板生成脚本：[scripts/prepare_domain_jsonl_template.py](scripts/prepare_domain_jsonl_template.py)
- 多域汇总脚本：[scripts/merge_domain_jsonl.py](scripts/merge_domain_jsonl.py)
- benchmark 构建脚本：[scripts/build_domain_benchmark.py](scripts/build_domain_benchmark.py)

先生成一个统一 JSONL 模板：

```bash
cd FedPLoRA  # 进入本包根目录（含 tasks、utilities、scripts）
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
cd FedPLoRA  # 进入本包根目录（含 tasks、utilities、scripts）
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
cd FedPLoRA  # 进入本包根目录（含 tasks、utilities、scripts）
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
  tasks/fed_train_glue.py \
  tasks/fed_train_e2e.py \
  tasks/fed_train_sft.py \
  utilities/train_eval.py \
  utilities/state_dict_ops.py \
  utilities/models.py \
  utilities/data_utils.py \
  utilities/utils.py \
  scripts/run_script.py \
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
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_glue.py \
  --model roberta-base \
  --task cola \
  --agg_type fedplora \
  --num_clients 3 \
  --lora_r 4 \
  --rounds 10 \
  --lr 1e-3 \
  --local_epochs 3 \
  --partition iid
```

### 9.2 GLUE sanity：Dirichlet non-IID

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_glue.py \
  --model roberta-base \
  --task cola \
  --agg_type fedplora \
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
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_e2e.py \
  --agg_type fedplora \
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

- [utilities/data_utils.py](utilities/data_utils.py)
- [tasks/fed_train_sft.py](tasks/fed_train_sft.py)
- [scripts/merge_domain_jsonl.py](scripts/merge_domain_jsonl.py)
- [scripts/build_domain_benchmark.py](scripts/build_domain_benchmark.py)

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

- [tasks/fed_train_sft.py](tasks/fed_train_sft.py)

它支持两种模式：

1. **直接读取 benchmark**
2. **从原始 JSONL 自动构建 benchmark 再训练**

### 11.1 直接用准备好的 benchmark 训练

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model ../../models/qwen3-14b \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora \
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
- 对 `fedplora` 而言，GPU 上只保留一份模型；每个 client 的本地 `B` 保存在 CPU 或磁盘中
- `--save_client_state_to_disk` 更适合 `35~70 clients` 的主实验
- 如果模型需要自定义代码，可以加 `--trust_remote_code`

### 11.2 从原始 JSONL 直接构建并训练

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model ../../models/qwen3-14b \
  --build_benchmark \
  --benchmark_jsonl data/raw/domain_7_all.jsonl \
  --benchmark_output_dir data/domain_benchmark \
  --num_clients_per_domain 5 \
  --min_samples_per_client 50 \
  --agg_type fedplora \
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

- [scripts/run_domain_sft.sh](scripts/run_domain_sft.sh)（单次跑一个 `AGG_TYPE`）
- [scripts/run_domain_sft_baselines.sh](scripts/run_domain_sft_baselines.sh)（按固定顺序自动串行多种 `AGG_TYPE`）
- [configs/domain_sft_pilot.env](configs/domain_sft_pilot.env)（单机 pilot 默认环境）
- [configs/domain_sft_baselines.env](configs/domain_sft_baselines.env)（批量 baseline 默认环境）

默认模型目录已配置为（可按机器修改 env 文件或命令行覆盖）：

`/data/yaominghao/gb/models/Meta-Llama-3.1-8B`

**聚合类型说明**：多轮 **`fedplora`** 为 **FedP-LoRA**（客户端按 FedP 协议上传 LoRA `A` 与可训练头、行统计；服务端 `aggregate_models_fedplora`；本地带对齐/近端/正交正则）。`fedplora-oneshot` 为 **FedP-LoRA 通信形态 + YOCO 单次聚合**：仍只上传 `A` 与头（`B` 留在客户端 / 磁盘），但联邦轮数强制为 1，服务端走 `aggregate_models_yoco`（PCWA 聚合 `A`），本地训练启用与独立 `yoco` 相同的 **LoRA `A` 稀疏先验**（`--yoco_sparse_lambda`）。二者不是同一条聚合代码路径。

#### 11.3.1 单次实验（pilot）

`scripts/run_domain_sft.sh` 会**自动** `cd` 到仓库根目录，并若存在则 **source `configs/domain_sft_pilot.env`**（无需先手动 `source`）。直接执行：

```bash
bash /path/to/FedPLoRA/scripts/run_domain_sft.sh
```

`domain_sft_pilot.env` 中已设置 `MODEL_PATH=/data/yaominghao/gb/models/Meta-Llama-3.1-8B`、`AGG_TYPE=fedplora` 等。若需临时覆盖环境变量，仍可在命令前导出（会覆盖 env 文件中的同名字段）：

```bash
MODEL_PATH=/data/yaominghao/gb/models/Meta-Llama-3.1-8B \
AGG_TYPE=fedplora \
CUDA_DEVICES=0,1 \
ROUNDS=20 \
BATCH_SIZE=1 \
bash /path/to/FedPLoRA/scripts/run_domain_sft.sh
```

#### 11.3.2 自动批量 baseline（推荐顺序）

[scripts/run_domain_sft_baselines.sh](scripts/run_domain_sft_baselines.sh) 同样会 **cd 到仓库根** 并自动 **source `configs/domain_sft_baselines.env`**。当前顺序为：**先 `fedplora-oneshot`，再 `fedplora`**，随后 `normal`、`ffa`、`fedex`（后三项在脚本里顺序可任意调整）。

```bash
bash /path/to/FedPLoRA/scripts/run_domain_sft_baselines.sh
```

`domain_sft_baselines.env` 中已写入 `MODEL_PATH`。需要临时改 GPU 或轮数时：

```bash
CUDA_DEVICES=0,1 ROUNDS=10 bash /path/to/FedPLoRA/scripts/run_domain_sft_baselines.sh
```

训练结束后，每轮 `domain_macro_loss` 等指标会写入：

```text
artifacts/sft_metrics/
```

### 11.4 运行 baseline

已补充批量 baseline 脚本：

- [scripts/run_domain_sft_baselines.sh](scripts/run_domain_sft_baselines.sh)
- [configs/domain_sft_baselines.env](configs/domain_sft_baselines.env)

```bash
source configs/domain_sft_baselines.env
bash scripts/run_domain_sft_baselines.sh
```

#### `normal`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model ../../models/qwen3-14b \
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model ../../models/qwen3-14b \
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model ../../models/qwen3-14b \
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /path/to/models/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora \
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
- 验证 `fedplora` 行为正常

### E2. E2E-NLG sanity

目标：

- 验证生成侧可用
- 检查因果 LM 路线没有逻辑错误

### E3. 7 域主实验

目标：

- 作为主结果表
- 比较 `fedplora` 与 `normal/ffa/fedex`

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
cd FedPLoRA  # 进入本包根目录（含 tasks、utilities、scripts）
conda activate fedplora
pip install -r requirements.txt
python -m py_compile \
  tasks/fed_train_glue.py tasks/fed_train_e2e.py tasks/fed_train_sft.py \
  utilities/train_eval.py utilities/state_dict_ops.py utilities/models.py \
  utilities/data_utils.py utilities/utils.py scripts/run_script.py \
  scripts/prepare_domain_jsonl_template.py scripts/build_domain_benchmark.py \
  scripts/merge_domain_jsonl.py
```

### 阶段 1：GLUE sanity

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_glue.py \
  --model roberta-base \
  --task cola \
  --agg_type fedplora \
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
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_e2e.py \
  --agg_type fedplora \
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora \
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model Qwen/Qwen3-32B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora \
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model mistralai/Mistral-Small-24B-Instruct-2501 \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora \
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
- 支持 `fedplora`、`normal`、`ffa`、`fedex`

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
3. 用 `Qwen/Qwen3-14B` 跑 `fedplora`
4. 再跑 `normal`、`ffa`、`fedex`

对应命令：

```bash
cd FedPLoRA  # 进入本包根目录（含 tasks、utilities、scripts）
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora \
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
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
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

---

## 附录：FedP-LoRA 多轮 vs FedP-LoRA–YOCO 单次（`fedplora-oneshot`）

### 本仓库中的实现对应关系

| 项目 | `fedplora`（多轮 FedP-LoRA） | `fedplora-oneshot`（FedP + YOCO 单次） |
|------|------------------------------------------|----------------------------------------|
| 联邦轮数 | 由 `--rounds` 指定，可多轮 | 入口强制 **1 轮**（真 one-shot 通信模式） |
| 客户端上传内容 | LoRA `A`、可训练任务头、由私有 `B` 推导的行重要度（与 `build_fedplora_upload_package` 一致） | **相同**（沿用 FedP-LoRA 的低通信上传协议） |
| LoRA `B` | 各客户端本地保留（域 SFT 下可落盘） | 同上 |
| 服务端聚合 | `methods/fedp_lora.py`：`aggregate_models_fedplora`（符号对齐、动量、QR 等） | `methods/yoco.py`：`aggregate_models_yoco`（对堆叠后的客户端 `A` 做 **PCWA**：主方向上的加权融合） |
| 本地训练额外项 | `utilities/train_eval.py`：FedP 对齐 / 近端 / 正交（依赖广播的全局 `A`） | **不**使用上述 FedP 多轮正则；启用 **YOCO 式 `A` 的 L1 稀疏项**（与 `agg_type=yoco` 共用 `--yoco_sparse_lambda`） |
| 独立 `yoco` 模式 | — | `agg_type=yoco` 与 `fedplora-oneshot` 在**聚合与稀疏先验**上一致；`fedplora-oneshot` 显式表达「在 FedP 上传协议下做 YOCO 单次」的研究设定 |

入口逻辑见 `tasks/fed_train_sft.py`（`is_fedplora_multiround_agg` vs `is_fedplora_oneshot_agg` / `is_yoco_agg`）、`utilities/utils.py`（`is_fedplora_oneshot_agg`）、`utilities/train_eval.py`（`_add_yoco_sparse` 对 `fedplora-oneshot` 生效）。

### 与 NeurIPS 2025 YOCO 论文的对应（目标算法 vs 当前代码）

论文 *You Only Communicate Once: One-shot Federated Low-rank Adaptation of MLLM*（YOCO）的核心脉络可概括为：

1. **动机**：真 one-shot FL 中各客户端从同一预训练权重出发，用预训练权重作为**隐式全局监督**，缓解异质数据带来的冲突；实证上**对 LoRA `B` 的方向（符号）约束**比对幅度约束更有效，且**仅约束 `B`** 优于同时约束 `A`、`B`。
2. **SVD 先验初始化**：对预训练权重矩阵 \(W\) 做 SVD，构造秩-\(r\) 的 \(A^0,B^0\) 使 \(A^0 B^0 \approx W\)，并取 \(B^s=\mathrm{sign}(B^0)\) 作为符号目标。
3. **本地总损失**：\(L = L_{\text{task}} + R_{\text{sign}} + R_{\text{sparse}}\)。其中 \(R_{\text{sign}}\) 用 \(\tanh(\gamma B)\) 光滑逼近符号，使 \(B\) 与 \(B^s\) 一致；\(R_{\text{sparse}}=\lambda\|A\|_1\) 鼓励客户端 **`A` 稀疏**、保留个性化。
4. **服务端**：对 **`B`** 做数据集规模加权平均；对 **`A`** 做 **PCWA**（将各客户端展平后的 `A` 堆叠，在主成分子空间内加权再重构），得到 \(A_g,B_g\)，全局增量 \(\Delta W_g=B_g A_g\)，与冻结的 \(W\) 组合用于推理；**仅一轮**上传与下发。

**本仓库当前已对齐的部分**：单轮协议；FedP 式「只上传 `A`+头、`B` 本地」与 YOCO 的「\(B\) 本地学习再参与全局」在**通信结构**上一致；**PCWA 聚合 `A`**（`aggregate_models_yoco`）；**本地对 `A` 的 L1 稀疏正则**（`--yoco_sparse_lambda`，`--yoco_pcwa_components` 控制主方向数）。

**尚未从论文全文落地的部分**（若要与论文完全一致，可作为后续工作）：基于 **SVD 的 LoRA 初始化**、仅作用于 **`B` 的符号一致性损失 \(R_{\text{sign}}\)**（及 \(\gamma,\beta\) 等超参）、服务端对 **`B` 的显式加权平均**（当前域 SFT 上传包不含 `B`，全局 `B` 不通过 FedAvg 式合并；与论文「上传 \(A_i,B_i\)」的叙述在实现细节上需按需扩展上传与聚合）。在扩展前，可将 `fedplora-oneshot` 理解为 **FedP 通信约束下的 YOCO 风格 one-shot：PCWA + `A` 稀疏先验**，并与多轮 `fedplora` 在代码上已明确分离。
