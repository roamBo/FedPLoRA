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
- **`scripts/`**：子目录 **`DataProcessScripts/`**（数据准备、benchmark 构建）、**`RunScripts/`**（`run_domain_sft*.sh`、`run_script.py` 等训练/批量实验入口）。
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

- [prepare_hf_domain_data.py](scripts/DataProcessScripts/prepare_hf_domain_data.py)
- [prepare_general_data.py](scripts/DataProcessScripts/prepare_general_data.py)
- [prepare_math_data.py](scripts/DataProcessScripts/prepare_math_data.py)
- [prepare_code_data.py](scripts/DataProcessScripts/prepare_code_data.py)
- [prepare_medical_data.py](scripts/DataProcessScripts/prepare_medical_data.py)
- [prepare_legal_data.py](scripts/DataProcessScripts/prepare_legal_data.py)
- [prepare_finance_data.py](scripts/DataProcessScripts/prepare_finance_data.py)
- [prepare_education_data.py](scripts/DataProcessScripts/prepare_education_data.py)
- [prepare_all_domains.sh](scripts/DataProcessScripts/prepare_all_domains.sh)
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

- 模板生成脚本：[scripts/DataProcessScripts/prepare_domain_jsonl_template.py](scripts/DataProcessScripts/prepare_domain_jsonl_template.py)
- 多域汇总脚本：[scripts/DataProcessScripts/merge_domain_jsonl.py](scripts/DataProcessScripts/merge_domain_jsonl.py)
- benchmark 构建脚本：[scripts/DataProcessScripts/build_domain_benchmark.py](scripts/DataProcessScripts/build_domain_benchmark.py)

先生成一个统一 JSONL 模板：

```bash
cd FedPLoRA  # 进入本包根目录（含 tasks、utilities、scripts）
python scripts/DataProcessScripts/prepare_domain_jsonl_template.py \
  --output data/raw/domain_7_template.jsonl \
  --examples_per_domain 2
```

然后把你实际下载和清洗后的数据写入：

```text
data/raw/domain_7_all.jsonl
```

如果你已经把各领域样本按目录存放，也可以先自动汇总：

```bash
python scripts/DataProcessScripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

如果你希望直接从 Hugging Face 数据集导出一个 pilot 版本，也可以执行：

```bash
cd FedPLoRA  # 进入本包根目录（含 tasks、utilities、scripts）
source configs/domain_data_pilot.env
bash scripts/DataProcessScripts/prepare_all_domains.sh
python scripts/DataProcessScripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

再独立构建 benchmark：

```bash
python scripts/DataProcessScripts/build_domain_benchmark.py \
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
  scripts/RunScripts/run_script.py \
  scripts/DataProcessScripts/prepare_domain_jsonl_template.py \
  scripts/DataProcessScripts/build_domain_benchmark.py \
  scripts/DataProcessScripts/merge_domain_jsonl.py \
  scripts/DataProcessScripts/prepare_hf_domain_data.py \
  scripts/DataProcessScripts/prepare_general_data.py \
  scripts/DataProcessScripts/prepare_math_data.py \
  scripts/DataProcessScripts/prepare_code_data.py \
  scripts/DataProcessScripts/prepare_medical_data.py \
  scripts/DataProcessScripts/prepare_legal_data.py \
  scripts/DataProcessScripts/prepare_finance_data.py \
  scripts/DataProcessScripts/prepare_education_data.py
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
- [scripts/DataProcessScripts/merge_domain_jsonl.py](scripts/DataProcessScripts/merge_domain_jsonl.py)
- [scripts/DataProcessScripts/build_domain_benchmark.py](scripts/DataProcessScripts/build_domain_benchmark.py)

### 10.1 第一步：准备统一 JSONL

将各领域样本整理到一个文件，例如：

```text
data/raw/domain_7_all.jsonl
```

如果你还没有开始整理数据，可以先生成模板：

```bash
python scripts/DataProcessScripts/prepare_domain_jsonl_template.py \
  --output data/raw/domain_7_template.jsonl \
  --examples_per_domain 2
```

如果你已经将每个域的文件分别放在 `data/domain_sources/<domain>/` 目录，也可以直接合并：

```bash
python scripts/DataProcessScripts/merge_domain_jsonl.py \
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
python scripts/DataProcessScripts/build_domain_benchmark.py \
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

以下命令均在**仓库根目录**执行；默认模型为 **Meta-Llama-3.1-8B** 本地目录（与 `configs/domain_sft_*.env` 一致），请按你机器上的实际路径替换 `--model`。

`fed_train_sft.py` 在 11.1 中给出 **12** 条可整段复制的命令（对应 **12** 个 `agg_type`）。其中 **`fedplora-oneshot`** 与 **`yoco`** 在服务端均调用 `aggregate_models_yoco`（PCWA），并均强制单轮通信；差别在于上传协议与命名：`fedplora-oneshot` 走 FedP 式「只上传 A+头」，`yoco` 为独立 `agg_type` 标签。别名：`fedsa`≡`fedsa_lora`，`fd_lora`≡`fdlora`，`het_lora`≡`hetlora`，`loraa2`≡`lora_a2`。

公共说明：

- 顺序客户端训练；需 **A/B 分离磁盘协议** 的方法（`fedplora`、`fedplora-oneshot`、`yoco`、`fedsa_lora`、`fedalt`）务必加 `--save_client_state_to_disk`。
- 若 Transformers 加载 Llama 报错，可在命令末尾追加 `--trust_remote_code`。

#### 1) `normal`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type normal \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 2) `fedex`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedex \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 3) `ffa`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type ffa \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 4) `fedplora`（多轮 FedP-LoRA）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk
```

#### 5) `fedplora-oneshot`（FedP 通信 + YOCO 单轮；入口会强制 `--rounds 1`）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedplora-oneshot \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --yoco_sparse_lambda 1e-4 \
  --yoco_pcwa_components 3 \
  --save_client_state_to_disk
```

#### 6) `yoco`（单轮 PCWA；入口会强制 `--rounds 1`）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type yoco \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --yoco_sparse_lambda 1e-4 \
  --yoco_pcwa_components 3 \
  --save_client_state_to_disk
```

#### 7) `fedsa_lora`（可用 `--agg_type fedsa` 等价）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedsa_lora \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk
```

#### 8) `fedalt`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fedalt \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk
```

#### 9) `hetlora`（可用 `--agg_type het_lora` 等价）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type hetlora \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 10) `flora`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type flora \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 11) `lora_a2`（可用 `--agg_type loraa2` 等价）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type lora_a2 \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 12) `fdlora`（可用 `--agg_type fd_lora` 等价）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark/seed_42 \
  --agg_type fdlora \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

### 11.1.1 七客户端 benchmark（`data/domain_benchmark_7c/seed_42`）

已由 `clients.json` / `domain_stats.json` 校验：**7 个域、每域 1 个客户端（共 7 个 `client_id`）**，与「每域 5 客户端」的 `data/domain_benchmark/seed_42` **目录不同，互不覆盖**。

**总数据量与每客户端数据量（7c / 14c / 21c / 35c）**：在同一套原始 JSONL 与相同的 `val_ratio` / `test_ratio` / `min_samples_per_client` 规则下构建时，各域在 `domain_stats.json` 里的 **`n_total` 不变**——只是把域内训练样本**切成不同数量的客户端 shard**。例如 **35c → 7c** 时，每域由 5 个客户端合并为 1 个，**每个客户端上的训练样本数约为原来的约 5 倍**；**全库参与训练的总样本量仍与 35 客户端版相同**。因此**单轮联邦的总本地更新步数（各客户端 batch 之和）与总数据量仍大致同阶**，训练墙钟**不会**仅仅因为「客户端数变少」就按 5 倍变快；变快主要来自**更少的客户端顺序调度开销**与**更少的评测前向次数**（`7 域 × N 客户端`）。若需要真正缩短训练时间，需减少数据（如 `MAX_SAMPLES`）、减小 `local_epochs` / `max_seq_length`、或增大 `batch_size`（在显存允许时）等。

**生成该划分**（在仓库根目录；输入 JSONL 按你实际路径调整）：

```bash
python scripts/DataProcessScripts/build_domain_benchmark.py \
  --input_jsonl data/raw/domain_7_all.jsonl \
  --output_dir data/domain_benchmark_7c \
  --num_clients_per_domain 1 \
  --min_samples_per_client 50 \
  --seed 42
```

**训练代码无需修改**：与 11.1 相同，仅将 **`--benchmark_dir` 改为 `data/domain_benchmark_7c/seed_42`**。评估前向次数为 **7 域 × 7 客户端 = 49**（约为 35 客户端版的约 1/5）。若仍嫌慢，可在任一条命令中追加 **`--eval_max_batches 50`**（或其它正整数）以截断每轮 eval 的 batch 数。

以下为 **12** 种 `agg_type` 的完整命令（与 11.1 一一对应，仅 `benchmark_dir` 不同）。

#### 1) `normal`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type normal \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 2) `fedex`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type fedex \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 3) `ffa`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type ffa \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 4) `fedplora`（多轮 FedP-LoRA）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk
```

#### 5) `fedplora-oneshot`（FedP 通信 + YOCO 单轮；入口会强制 `--rounds 1`）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type fedplora-oneshot \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --yoco_sparse_lambda 1e-4 \
  --yoco_pcwa_components 3 \
  --save_client_state_to_disk
```

#### 6) `yoco`（单轮 PCWA；入口会强制 `--rounds 1`）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type yoco \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --yoco_sparse_lambda 1e-4 \
  --yoco_pcwa_components 3 \
  --save_client_state_to_disk
```

#### 7) `fedsa_lora`（可用 `--agg_type fedsa` 等价）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type fedsa_lora \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk
```

#### 8) `fedalt`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type fedalt \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk
```

#### 9) `hetlora`（可用 `--agg_type het_lora` 等价）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type hetlora \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 10) `flora`

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type flora \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 11) `lora_a2`（可用 `--agg_type loraa2` 等价）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type lora_a2 \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

#### 12) `fdlora`（可用 `--agg_type fd_lora` 等价）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_7c/seed_42 \
  --agg_type fdlora \
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
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

### 11.2 从原始 JSONL 直接构建并训练

在首次没有 benchmark 时，可用下面命令**先构建再训练**（同样使用 Meta-Llama；`--agg_type` 可换成 11.1 中任意一种）。开启 `--build_benchmark` 时，训练使用的 `split_dir` 由构建结果决定（控制台会打印 `[benchmark] loaded from ...`），**不要**再单独传 `--benchmark_dir`。

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
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
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk
```

### 11.3 推荐的一键脚本

已补充主实验脚本：

- [scripts/RunScripts/run_domain_sft.sh](scripts/RunScripts/run_domain_sft.sh)（单次跑一个 `AGG_TYPE`）
- [scripts/RunScripts/run_domain_sft_baselines.sh](scripts/RunScripts/run_domain_sft_baselines.sh)（按固定顺序自动串行多种 `AGG_TYPE`）
- [scripts/RunScripts/run_domain_sft_baselines1.sh](scripts/RunScripts/run_domain_sft_baselines1.sh)（分组 baseline ①：FedP-LoRA-oneshot、FedALT、Flora、FedAvg-Normal）
- [scripts/RunScripts/run_domain_sft_baselines2.sh](scripts/RunScripts/run_domain_sft_baselines2.sh)（分组 baseline ②：YOCO、FedSA-LoRA、FFA）
- 扩展实验脚本（详见 **§十四**）：[`run_exp_personalization.sh`](scripts/RunScripts/run_exp_personalization.sh)、[`run_exp_comm_profile.sh`](scripts/RunScripts/run_exp_comm_profile.sh)、[`run_exp_ablation_fedplora.sh`](scripts/RunScripts/run_exp_ablation_fedplora.sh)
- [configs/domain_sft_pilot.env](configs/domain_sft_pilot.env)（单机 pilot 默认环境）
- [configs/domain_sft_baselines.env](configs/domain_sft_baselines.env)（批量 baseline 默认环境）

默认模型目录已配置为（可按机器修改 env 文件或命令行覆盖）：

`/data/yaominghao/gb/models/Meta-Llama-3.1-8B`

**聚合类型说明**：多轮 **`fedplora`** 为 **FedP-LoRA**（客户端按 FedP 协议上传 LoRA `A` 与可训练头、行统计；服务端 `aggregate_models_fedplora`；本地带对齐/近端/正交正则）。`fedplora-oneshot` 为 **FedP-LoRA 通信形态 + YOCO 单次聚合**：仍只上传 `A` 与头（`B` 留在客户端 / 磁盘），但联邦轮数强制为 1，服务端走 `aggregate_models_yoco`（PCWA 聚合 `A`），本地训练启用与独立 `yoco` 相同的 **LoRA `A` 稀疏先验**（`--yoco_sparse_lambda`）。二者不是同一条聚合代码路径。

#### 11.3.1 单次实验（pilot）

`scripts/RunScripts/run_domain_sft.sh` 会**自动** `cd` 到仓库根目录，并若存在则 **source `configs/domain_sft_pilot.env`**（无需先手动 `source`）。直接执行：

```bash
bash /path/to/FedPLoRA/scripts/RunScripts/run_domain_sft.sh

不加gpu参数就会自动选卡
bash scripts/RunScripts/run_domain_sft_baselines2.sh 35 1
bash scripts/RunScripts/run_exp_personalization.sh 35 1
bash scripts/RunScripts/run_exp_ablation_fedplora.sh 35 1
bash scripts/RunScripts/run_exp_comm_profile.sh 1
bash scripts/RunScripts/run_domain_sft_baselines.sh 1
```

可选**第一个参数**指定 GPU（单卡或逗号列表，如 `1` 或 `0,1`）。不写且未在环境里设置 `CUDA_DEVICES` 时，由 [`configs/cuda_resolve.inc.sh`](configs/cuda_resolve.inc.sh) 在本机有 `nvidia-smi` 的情况下**自动选用空闲显存最大的 GPU**；无 GPU 工具时回退为 `0`。若要强制不用自动选卡：`AUTO_CUDA_PICK=0 bash ...`（此时使用 `CUDA_DEVICES_FALLBACK`，默认 `0`）。

`domain_sft_pilot.env` 中已设置 `MODEL_PATH=/data/yaominghao/gb/models/Meta-Llama-3.1-8B`、`AGG_TYPE=fedplora` 等。若需临时覆盖环境变量，仍可在命令前导出（会覆盖 env 文件中的同名字段）；其中 **`CUDA_DEVICES` 仍可通过前缀指定**，优先级高于自动选卡与第一个参数：

```bash
MODEL_PATH=/data/yaominghao/gb/models/Meta-Llama-3.1-8B \
AGG_TYPE=fedplora \
CUDA_DEVICES=1 \
ROUNDS=20 \
BATCH_SIZE=1 \
bash /path/to/FedPLoRA/scripts/RunScripts/run_domain_sft.sh
```

#### 11.3.2 自动批量 baseline（推荐顺序）

[scripts/RunScripts/run_domain_sft_baselines.sh](scripts/RunScripts/run_domain_sft_baselines.sh) 同样会 **cd 到仓库根** 并自动 **source `configs/domain_sft_baselines.env`**。当前顺序为：**先 `fedplora-oneshot`，再 `fedplora`**，随后 `normal`、`ffa`、`fedex`（后三项在脚本里顺序可任意调整）。

```bash
bash /path/to/FedPLoRA/scripts/RunScripts/run_domain_sft_baselines.sh
# 指定仅用 1 号物理卡：
bash /path/to/FedPLoRA/scripts/RunScripts/run_domain_sft_baselines.sh 1
```

`domain_sft_baselines.env` 中已写入 `MODEL_PATH`（**不再**写入 `CUDA_DEVICES`，避免覆盖命令行前缀）。需要临时改 GPU 或轮数时：

```bash
CUDA_DEVICES=1 ROUNDS=10 bash /path/to/FedPLoRA/scripts/RunScripts/run_domain_sft_baselines.sh
```

#### 11.3.3 分组 baseline 脚本（可选客户端数 7 / 14 / 21 / 35）

以下两个脚本会 **cd 到仓库根**、自动 **source `configs/domain_sft_baselines.env`**（若存在），并根据**第一个命令行参数**选择 benchmark；**第二个可选参数**为 GPU（与 `run_domain_sft_baselines.sh` 的首参同理，见 `configs/cuda_resolve.inc.sh`）。

| 参数 | `benchmark_dir` |
|------|-----------------|
| `7` | `data/domain_benchmark_7c/seed_42` |
| `14` | `data/domain_benchmark_14c/seed_42` |
| `21` | `data/domain_benchmark_21c/seed_42` |
| `35`（默认） | `data/domain_benchmark_35c/seed_42` |

**脚本 1**（`run_domain_sft_baselines1.sh`）串行方法及与代码中 `agg_type` 的对应：

| 论文/口头名称 | `agg_type` |
|---------------|------------|
| FedP-LoRA-oneshot | `fedplora-oneshot` |
| FedALT | `fedalt` |
| Flora | `flora` |
| FedAvg-Normal | `normal` |

**脚本 2**（`run_domain_sft_baselines2.sh`）：

| 论文/口头名称 | `agg_type` |
|---------------|------------|
| YOCO | `yoco` |
| FedSA-LoRA | `fedsa_lora` |
| FFA | `ffa` |

**示例**（与 `domain_sft_baselines.env` 联用）：

```bash
source configs/domain_sft_baselines.env
bash scripts/RunScripts/run_domain_sft_baselines1.sh 7
bash scripts/RunScripts/run_domain_sft_baselines2.sh 35
# 35 客户端 + 指定 GPU 1：
bash scripts/RunScripts/run_domain_sft_baselines2.sh 35 1
```

不传第一个参数时等价于末尾的 `35`。其它超参仍可通过环境变量覆盖（与 `run_domain_sft_baselines.sh` 相同，如前缀 `CUDA_DEVICES=`、`ROUNDS`）。

训练结束后，每轮 **`domain_macro_token_accuracy` / `domain_macro_perplexity` / `domain_macro_loss`** 等会写入（默认路径在加载 benchmark 后按客户端数 **自动** 带上后缀，子目录名不变）：

```text
artifacts_{num_clients}c/sft_metrics/
```

例如 35 个客户端时为 `artifacts_35c/sft_metrics/`；客户端状态目录同理为 `artifacts_35c/domain_client_states/`。若你在命令行 **显式** 传入非默认的 `--metrics_output_dir` / `--client_state_dir`，则不会改写。控制台会打印 `[setup] client_state_dir=... metrics_output_dir=...`。

### 11.4 运行 baseline

已补充批量 baseline 脚本：

- [scripts/RunScripts/run_domain_sft_baselines.sh](scripts/RunScripts/run_domain_sft_baselines.sh)（全量默认顺序）
- [scripts/RunScripts/run_domain_sft_baselines1.sh](scripts/RunScripts/run_domain_sft_baselines1.sh)、[scripts/RunScripts/run_domain_sft_baselines2.sh](scripts/RunScripts/run_domain_sft_baselines2.sh)（分组 + `7|14|21|35` 选 benchmark，见 **§11.3.3**）
- [configs/domain_sft_baselines.env](configs/domain_sft_baselines.env)

```bash
source configs/domain_sft_baselines.env
bash scripts/RunScripts/run_domain_sft_baselines.sh
# 或（示例：7 客户端数据）
bash scripts/RunScripts/run_domain_sft_baselines1.sh 7
bash scripts/RunScripts/run_domain_sft_baselines2.sh 7
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
python scripts/DataProcessScripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

现在仓库已经补上：

- `scripts/DataProcessScripts/prepare_general_data.py`
- `scripts/DataProcessScripts/prepare_math_data.py`
- `scripts/DataProcessScripts/prepare_code_data.py`
- `scripts/DataProcessScripts/prepare_medical_data.py`
- `scripts/DataProcessScripts/prepare_legal_data.py`
- `scripts/DataProcessScripts/prepare_finance_data.py`
- `scripts/DataProcessScripts/prepare_education_data.py`
- `scripts/DataProcessScripts/prepare_hf_domain_data.py`
- `scripts/DataProcessScripts/prepare_all_domains.sh`

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

论文级实验按下面 **四条主线** 组织；每条主线前用 **【】** 标出。其余条目（GLUE / E2E / 跨模型 / hard setting 等）作为 **扩展实验**。

---

### 各实验在做什么、要看哪些指标（总览）

下面四条主线回答的问题不同；**跑完脚本后应优先打开的路径**与 **JSON 里要对的字段**一并列出，便于一次把代码/设定固定后直接出终稿结果。

| 主线 | 要回答的问题 | 是否训练 | 典型脚本入口 | 产物位置（默认） | **优先阅读的指标 / 字段** |
|------|----------------|----------|----------------|-------------------|---------------------------|
| **【7域主实验】** | 各 `agg_type` 在 7 域上的整体 LM 表现谁更好 | 是 | §11.3 `run_domain_sft_baselines*.sh` 等 | `artifacts_{N}c/sft_metrics/<agg>_<model>_*_r*_e*_seed*.json` | **主指标（推荐写进主表）**：每轮 **`domain_macro_token_accuracy`、`domain_macro_perplexity`** 及 **`worst_domain_token_accuracy`、`worst_domain_perplexity`**（均有 `best_*` 追踪最优轮）。**辅助**：`domain_macro_loss` / `worst_domain_loss`（与 NLL 一致，仍保留）。JSON 顶层含 **`recommended_primary_metrics`** 字段名列表。分域见 `rounds[].domain_metrics[<domain>]`。**通信**：顶层 `communication.*`。日志 `[eval]` 以 `primary_*` 打头、`aux_*loss` 为辅助。 |
| **【个性化收益分析】** | 仅看 domain-macro 不够：客户端是否在本域更优、离开本域是否变差 | 是 | `run_exp_personalization.sh` | 同上 `sft_metrics` | **主读**与主实验同口径的 **`token_accuracy` + `perplexity`**（`client_local_*`、`in_domain_domain_test_*`、`off_domain_*`）；**`personalization_gap_token_accuracy`（local−off）**、**`personalization_gap_perplexity`（off−local，越大表示跨域更难）**。**`loss` 仅辅助**。JSON 含 **`recommended_primary_personalization_metrics`**。 |
| **【通信-性能实验】** | 在**同一套 PEFT 形状**下，各方法的**单客户端单轮**上下行字节（便于画 Pareto / 方法表） | **否**（只加载模型算字节） | `run_exp_comm_profile.sh` → `print_sft_comm_profile.py` | 终端表格；可加 `--json` 管道保存 | 每个 `agg_type` 的 **`down_bytes_per_client`、`up_bytes_per_client`**（及脚本打印的 MB 列）。**不写入** `sft_metrics`；与训练是否跑完无关。 |
| **【机制消融】** | `fedplora-oneshot` 路径上，YOCO 稀疏与 PCWA 超参是否敏感 | 是 | `run_exp_ablation_fedplora.sh` | `artifacts_{N}c/sft_metrics_oneshot_ablation/<full|no_sparse|pcwa_k1>/` | 与主实验相同的 **domain-macro / worst / domain_metrics**；按消融 tag **分文件**对比即可。 |

**总通信量（直觉）**：`utilities/utils.py` 中 `estimate_round_communication_bytes` 给出的是 **每名客户端、每一轮** 的下行/上行字节；单轮全集群流量量级约为 **`num_clients × (down_bytes_per_client + up_bytes_per_client)`**（与函数注释一致）。论文里若写「总上传」，请自行用 **`up_bytes_per_client × num_clients × rounds`**（若各轮相同）或按实现说明截取。

---

#### 主实验 JSON 里的 `communication` 与「通信-性能」脚本是否重复？要不要删？

- **公式相同**：两处都调用 **`estimate_round_communication_bytes`**（`fed_train_sft.py` 启动时写入 metrics；`print_sft_comm_profile.py` 批量打印）。数值在相同 `agg_type`、相同模型与 LoRA 配置下应对齐。
- **用途不同（建议都保留）**：
  - **主实验 JSON**：与该次 run 的 **`domain_macro_loss` 等写在同一文件**，可复查「这次实验到底用的哪套通信假设」，便于归档与答辩溯源。
  - **`run_exp_comm_profile`**：**零训练**，一次列出**多种** `agg_type`，专门用来做通信对比表 / Pareto 轴，省 GPU 时间。

因此 **不建议从主实验 metrics 中删除 `communication` 块**；论文主表若以通信章节脚本为准，只需注明数字来源与 `estimate_round_communication_bytes` 一致即可。

---

### 【7域主实验】

对应原 **E3**，主结果表与 baseline 对比（`fedplora`、`fedplora-oneshot`、`normal`、`ffa`、`fedex` 及 README §11 中其它 `agg_type`）。

- **数据**：`data/domain_benchmark_35c/seed_42`（或 `7c` / `14c` / `21c` 做客户端规模扫描）。
- **脚本**：§11.3 的 `run_domain_sft.sh`、`run_domain_sft_baselines.sh`、`run_domain_sft_baselines1.sh`、`run_domain_sft_baselines2.sh`。`run_domain_sft.sh` / `run_domain_sft_baselines.sh` 可选首参指定 GPU；后二者第一个参数 `7|14|21|35` 选择 `domain_benchmark_<N>c`，第二个可选参数指定 GPU（见 §11.3 与 `configs/cuda_resolve.inc.sh`）。
- **指标（汇总）**：见上表「7域主实验」行。**论文主表建议以 `domain_macro_token_accuracy`（越高越好）与 `domain_macro_perplexity`（越低越好）为主**；`worst_domain_*` 对应「最难点域」的稳健性。**`domain_macro_loss` 仅作辅助**（与交叉熵一致）。`rounds[]` 内字段顺序已把 token 准确率与 PPL 放在 loss 之前；metrics JSON 顶层 **`recommended_primary_metrics`** 列出推荐主字段名。**`domain_metrics`** 仍为按域对象：`loss`、`token_accuracy`、`perplexity`。顶层 **`communication`** 为单客户端单轮上下行字节估计。

**前置 sanity（非主线必排进主表，但建议先做）**

- **E1. GLUE sanity**：验证分类联邦与通信统计。
- **E2. E2E-NLG sanity**：验证生成侧因果 LM 链路。

---

### 【个性化收益分析】

对应原 **E5**：比较 **客户端本地 held-out（本域）** 与 **非本域 held-out（跨域）** 上的 LM loss，避免只看 domain-macro 平均而忽略 specialization。  
脚本侧与主线一致：先跑 **`fedplora-oneshot`**（单轮），再跑 **`normal`** 作对照。

- **指标（汇总）**：见 §十四总览表「个性化收益分析」行。**建议主表/主图用 `token_accuracy` 与 `perplexity`（及 gap）**，与 7 域主实验口径一致；**loss 仅作辅助**。

**代码**

- 训练入口增加 `--eval_personalization_metrics`：在原有 domain-macro eval 之外，写入并打印（均进入 `rounds[]`）；metrics 顶层含 **`recommended_primary_personalization_metrics`**。  
  - **本地 held-out**：`client_local_macro_{token_accuracy,perplexity,loss}`（`test_local` 按客户端 eval 再 macro）；  
  - **本域 `test_domain`**：`in_domain_domain_test_macro_{token_accuracy,perplexity,loss}`；  
  - **非本域 `test_domain`**：`off_domain_macro_{token_accuracy,perplexity,loss}`（每个 (客户端 × 非本域) eval 后再 macro）；  
  - **派生**：`personalization_gap_token_accuracy = local − off`（越大越好）；`personalization_gap_perplexity = off − local`（越大表示跨域相对本地越「更困惑」）；`personalization_gap_loss = off − local`（辅助）。  
  另保留主线 **`domain_macro_*`** 等与全文对照。

**一键脚本**

```bash
source configs/domain_sft_baselines.env
bash scripts/RunScripts/run_exp_personalization.sh 35
# 第二个参数可指定 GPU，与 §11.3.3 一致，例如 35 客户端、1 号卡：
bash scripts/RunScripts/run_exp_personalization.sh 35 1
```

（第一个参数 `7|14|21|35` 与 `run_domain_sft_baselines1.sh` 相同；第二个可选参数为 GPU。未指定时见 `configs/cuda_resolve.inc.sh`：有 `nvidia-smi` 则自动选空闲显存最大的卡。）

---

### 【通信-性能实验】

对应原 **E6**：在同一套 PEFT 结构下对比各 `agg_type` 的 **每轮下行/上行字节**（与训练时 `estimate_round_communication_bytes` 一致），再与最终 `domain_macro_loss` 等联合作 Pareto 图。

- **指标（汇总）**：见 §十四总览表「通信-性能实验」行；输出为各 **`agg_type` 的 `down_bytes_per_client` / `up_bytes_per_client`**。需要落盘时：`python scripts/RunScripts/print_sft_comm_profile.py ... --json > comm.json`。

**代码含义（与实现对照）**

- **近似 `A+B` 全量可训练 LoRA 上传**：`normal`、`fedex`（见 `utilities/utils.py` 中 `estimate_round_communication_bytes`）。
- **FedPLoRA / YOCO 等「仅 A + 头」上传**：`fedplora`（多轮）、`fedplora-oneshot`（单轮 + PCWA，与 §十四 三条扩展脚本主线一致）、`yoco`、`fedsa_lora`、`fedalt`。
- **`ffa`**：下行全量、上行为 `B+head` 的专用规则（同文件）。
- **「B-only / local-only」**：当前仓库 **未**实现为独立 `agg_type`；若论文需要，需在方法层增加协议后再接统计。

**一键脚本（只打通信表，不训练）**

```bash
source configs/domain_sft_baselines.env
bash scripts/RunScripts/run_exp_comm_profile.sh
bash scripts/RunScripts/run_exp_comm_profile.sh 1
```

可选：`AGG_LIST=normal,fedplora-oneshot,ffa bash scripts/RunScripts/run_exp_comm_profile.sh 1`（默认列表已不含多轮 `fedplora`；末尾可选参数为 GPU）。  
底层：`python scripts/RunScripts/print_sft_comm_profile.py --model "$MODEL_PATH"`（会 **加载一次** 基座+LoRA 以统计参数规模，首次较慢）。

---

### 【机制消融】

对应原 **E7**，针对 **`fedplora-oneshot`**（单轮；服务端为 `aggregate_models_yoco` / PCWA，**不是**多轮 `aggregate_models_fedplora`）。  
多轮 `fedplora` 的 `gp_*` 与 `--fedplora_ablation_no_consensus` / `no_momentum` 仅在多轮聚合路径生效；oneshot 的脚本消融改为对 **YOCO 侧可开关项**：

**和 YOCO 是什么关系？你明明是 `fedplora-oneshot`？**  
在本仓库里，`fedplora-oneshot` **名字里带 FedPLoRA**，实现上是 **两段拼在一起**（见 `utilities/utils.py` 里 `is_fedplora_oneshot_agg` 的注释）：

1. **FedP 通信形态**：客户端只上传 LoRA **`A` + 可训练头**（及 FedP 需要的行统计等），**`B` 留在本地**——和独立 `agg_type=yoco` 的上传集合同类（都是「A+头」侧），但协议细节按 FedP 打包。
2. **YOCO 服务端**：联邦 **只跑 1 轮**，聚合函数是 **`aggregate_models_yoco`**（`methods/yoco.py`），即对堆叠的 **A 做 PCWA**；`--yoco_pcwa_components` 控制主方向个数 k。
3. **YOCO 客户端正则**：本地训练时对 **`lora_A` 加 L1 风格稀疏项 **`--yoco_sparse_lambda`**（与独立 `yoco` 共用 `train_eval._add_yoco_sparse`）。

因此：**消融表里出现 `yoco_*` 超参是正常的**——它们消融的是「oneshot 里借用的 YOCO 聚合 + YOCO 稀疏先验」，**不是**把实验偷偷改成 `agg_type=yoco`。若论文叙事要强调「我们的方法是 FedP 上传 + 单次 YOCO 式聚合」，可直接引用上述分工。

- **指标（汇总）**：见 §十四总览表「机制消融」行；与主实验相同的 **`rounds[]` 性能字段**，按 **`sft_metrics_oneshot_ablation/<tag>/`** 对比各 tag。

---

#### 消融要消融什么？（清单，便于你决定写进论文哪几行）

下面按 **「机制块 → 代码位置 → 怎么关掉/扫参」** 列全。当前 **`run_exp_ablation_fedplora.sh` 只自动跑 A 类**；B 类需 **`--agg_type fedplora`** 自行组命令；C 类需你扩展脚本或手写多次运行。

**A. `fedplora-oneshot` 上真的会生效的机制（与 `run_exp_ablation_fedplora.sh` 一致）**

| 机制块 | 作用（一句话） | 实现位置 | CLI / 环境变量 | 脚本预设 `ABLATION_MODE` | 你可选是否做 |
|--------|----------------|----------|----------------|--------------------------|--------------|
| **本地 A 稀疏先验** | 本地训练时对 `lora_A` 加 L1 风格惩罚，与独立 `yoco` 相同 | `utilities/train_eval.py` → `_add_yoco_sparse` | `--yoco_sparse_lambda`（默认 `1e-4`）；环境变量 **`YOCO_SPARSE_LAMBDA`** | `no_sparse`：置 `0` | 验证「没有稀疏先验」对单轮 FedP+PCWA 的影响 |
| **服务端 PCWA 主方向数 k** | 堆叠各客户端 A 后 SVD，用前 k 个主方向能量给客户端加权聚合 | `methods/yoco.py` → `aggregate_models_yoco` | `--yoco_pcwa_components`（默认 `3`）；**`YOCO_PCWA_COMPONENTS`** | `pcwa_k1`：`k=1` | 验证「只保留 1 个主方向」vs 多方向加权 |
| **full 基线** | 上述两项均用默认/环境覆盖 | — | `full` 传入脚本默认值 | `full` | 消融表里的「完整」一行 |

说明：**oneshot 固定 1 轮**，`--rounds` 会被 `fed_train_sft.py` 强制为 1；**不存在**多轮里的「上一轮全局 A」「共识符号对齐」等，因此 **A 表就是 oneshot 论文里「机制」的全部开关**（除非你再做 C 类扩展）。

**B. 仅当 `agg_type=fedplora`（多轮 FedPLoRA）才生效——当前 oneshot 脚本不跑**

若主文只写 **fedplora-oneshot**，下列项 **不会** 出现在 `run_exp_ablation_fedplora.sh` 里；若你**另开一节多轮 FedPLoRA** 或对比「多轮里的正则/聚合」，再按需选。

| 机制块 | 作用（一句话） | 实现位置 | CLI | 典型消融做法 |
|--------|----------------|----------|-----|----------------|
| **对齐项 R_align** | 约束本地更新与参考在 dW 空间一致 | `train_eval._add_fedplora_regularization` | `--gp_align_lambda` | 置 `0`（w/o align） |
| **近端项 R_prox** | FedP 风格近端 | 同上 | `--gp_prox_lambda` | 置 `0`（w/o prox） |
| **正交项 R_orth** | 正则化 A/B 结构 | 同上 | `--gp_orth_lambda` | 置 `0`（w/o orth） |
| **共识加权（行符号 + 幂次）** | 服务端按行与上一轮 A 对齐并加权 | `methods/fedp_lora.py` → `aggregate_models_fedplora` | `--fedplora_ablation_no_consensus` | 打开 flag（w/o consensus） |
| **服务端 A 的动量 EMA** | 与上一轮全局 A 混合 | 同上 | `--fedplora_ablation_no_momentum` | 打开 flag（w/o momentum） |
| **共识幂次、动量系数** | 调服务端形状 | `fedp_lora` + argparse | `--gp_consensus_power`、`--gp_agg_momentum` | 扫参或极端值，属**超参消融**而非二元开关 |

命令入口：直接 `python tasks/fed_train_sft.py --agg_type fedplora --rounds <大于 1 的整数> ...` 并按上表加减 flag/λ。

**C. 论文可写、但需你自编实验（本仓库未写死成消融行）**

| 方向 | 说明 |
|------|------|
| **PCWA 的 k 扫描** | 除 `k=1` 外可试 `k=2,5,min(3,n-1)` 等，复制 `run_exp_ablation_fedplora.sh` 里 `run_one` 模式即可 |
| **稀疏强度扫描** | 扫 `yoco_sparse_lambda`（如 `0,1e-5,1e-4,1e-3`） |
| **与 `agg_type=yoco` 对比** | 上传/打包与 FedP 略有不同，属「协议级」对比，不是同一脚本里的二元消融 |
| **训练配方** | `local_epochs`、`lr`、`lora_r`：算**训练敏感性**，一般单独一小表，不挤进「机制消融」主表 |
| **FedP 行统计是否参与** | 多轮聚合里行统计进上传；oneshot 的 YOCO 聚合**当前未**单独提供「关掉行统计」开关，要做得改 `methods/` 或加 flag |

**一键脚本（仅覆盖 A 类）**

```bash
source configs/domain_sft_baselines.env
bash scripts/RunScripts/run_exp_ablation_fedplora.sh 35
bash scripts/RunScripts/run_exp_ablation_fedplora.sh 35 1
```

单组：`ABLATION_MODE=no_sparse bash scripts/RunScripts/run_exp_ablation_fedplora.sh 35 1`。  
子集：`ABLATION_MODE="full pcwa_k1" bash ...`（空格分隔，与脚本 `for mode in ${MODES}` 一致）。  
脚本为每组消融写入 `artifacts_{N}c/sft_metrics_oneshot_ablation/<tag>/` 与 `domain_client_states_oneshot_ablation/<tag>/`，避免多组同名 `fedplora-oneshot` metrics 或磁盘 client 状态互相串。  
若需 **多轮 `fedplora`** 上表 5 类消融，请直接对 `tasks/fed_train_sft.py` 使用 `--agg_type fedplora` 与对应 CLI，而非本脚本。

---

### 扩展实验（未归入四条主线）

- **E4. 跨模型验证**：`Qwen3-32B`、`Mistral-Small-24B` 等，见 §十五阶段 7–8。
- **E8. 6 域 hard setting**：去掉 `general` 域，需重新构建 benchmark JSONL。
- **E9. 分层异质性**：域内再划分，需数据与构建脚本扩展。
- **E10. 不均衡与参与率**：需采样客户端子集或改 `create_domain_client_dataloaders` 逻辑。
- **E11. Transfer matrix**：由每轮各域 loss 组装 `7×7` 矩阵（可作后处理脚本）。
- **E12. 上传载荷泄露代理**：对比上传 `A+B` vs `A-only` 的可推断性，属安全向扩展。

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
  utilities/data_utils.py utilities/utils.py scripts/RunScripts/run_script.py \
  scripts/DataProcessScripts/prepare_domain_jsonl_template.py scripts/DataProcessScripts/build_domain_benchmark.py \
  scripts/DataProcessScripts/merge_domain_jsonl.py scripts/RunScripts/print_sft_comm_profile.py
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
python scripts/DataProcessScripts/prepare_domain_jsonl_template.py \
  --output data/raw/domain_7_template.jsonl \
  --examples_per_domain 2
```

或者从按域存放的文件自动汇总：

```bash
python scripts/DataProcessScripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

如果要直接从 Hugging Face 数据集准备一个 pilot：

```bash
source configs/domain_data_pilot.env
bash scripts/DataProcessScripts/prepare_all_domains.sh
python scripts/DataProcessScripts/merge_domain_jsonl.py \
  --input_root data/domain_sources \
  --output data/raw/domain_7_all.jsonl \
  --recursive
```

### 阶段 4：自动构建 benchmark

```bash
python scripts/DataProcessScripts/build_domain_benchmark.py \
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
bash scripts/RunScripts/run_domain_sft.sh
```

训练指标会自动落到 `artifacts_{num_clients}c/sft_metrics/`（与 §11.3.2 一致）。

### 阶段 6：Pilot baseline

可直接批量运行：

```bash
source configs/domain_sft_baselines.env
bash scripts/RunScripts/run_domain_sft_baselines.sh
# 分组 baseline + 客户端数（7/14/21/35）见 §11.3.3
bash scripts/RunScripts/run_domain_sft_baselines1.sh 7
bash scripts/RunScripts/run_domain_sft_baselines2.sh 7
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

**域 SFT（`fed_train_sft.py`）日志与 `artifacts_*c/sft_metrics/*.json` 中每轮包含（推荐主指标在前）：**

- **`domain_macro_token_accuracy` / `worst_domain_token_accuracy`** 及 **`best_*`**：在 **非 `-100` 的 label 位置**（即 **response 段**）上，**下一词预测** 的 micro **token 准确率**（先对每个客户端在该域测试集上算 micro-acc，再在客户端上取平均，再对域取 macro / worst）。这是因果 LM SFT 下与「分类 Accuracy」最接近、可自动批量计算的指标，**建议主表主列用它（越高越好）**。
- **`domain_macro_perplexity` / `worst_domain_perplexity`** 及 **`best_*`**：各域 `perplexity` 再 macro / max-worst；**建议主表并列 PPL（越低越好）**，与 LM 论文习惯一致。
- **`domain_macro_loss` / `worst_domain_loss`**：**辅助**（交叉熵，与 NLL 一致）；需要与旧文或 loss 曲线对比时保留。
- **`perplexity`（每域 `domain_metrics` 字典内）**：与上同源；轮级 macro PPL 由域级 PPL 聚合得到。
- **`recommended_primary_metrics`**（JSON 顶层）：字段名列表，标明推荐写进论文主结果的键。
- **`communication`**（JSON 顶层）：单轮上下行字节估计。

**与 YOCO 原论文表格的对应关系（你截图中的 Table 1–4）**：YOCO 主表是 **Hateful-Memes / CrisisMMD / VQA-RAD / SLAKE** 等任务上的 **分类或 VQA 准确率（Accuracy）** 与 **通信代价（rounds×layers×LoRAs）**。本仓库主线是 **7 域文本 SFT**，没有多模态分类头，因此 **不会直接产出「整张表那种 task Accuracy」**；若要对齐论文叙事，可选扩展包括：**(a)** 在固定测试集上做 **生成式 Exact Match / ROUGE-L**（需额外解码与参考串）；**(b)** 对带选项的医学/安全子集改 **MCQ 分类头** 再算 Acc；**(c)** 单独接 **GLUE** 子实验走分类 Acc（已有 `fed_train_glue.py`）。当前默认可写进论文主表的是 **token-level acc + PPL（主）+ comm + loss（辅）**。

**进一步可选指标（按需实现）**：`eval_max_batches` 截断下的 **校准性**（ECE）、**长度惩罚后的 BLEU**、**域间迁移矩阵**（每对域 loss）、**训练步 token 吞吐**、**显存峰值**。

**缩短评估时间**：`--eval_max_batches N`（每客户端×每域只跑前 N 个 batch）；pilot 用较小 `--max_seq_length`；保证评测时 **GPU 独占**；同一次进程内模型已在显存中，**「保存 checkpoint」不会减少当前这次 domain-macro 顺序 eval 的前向次数**；若师兄指的是把 **训练结束后的权重落盘**，下次 **单独写评测脚本只加载模型做 eval**，可避免重复训练，但总 eval 算子数不变。

---

## 十七、建议表格与图

### 表 1：7 域主结果

列（与 `sft_metrics` 中 `recommended_primary_metrics` 一致，**不以 loss 为主列**）：

- 方法
- **domain-macro token accuracy**（`domain_macro_token_accuracy`，越高越好）
- **domain-macro PPL**（`domain_macro_perplexity`，越低越好）
- **worst-domain token accuracy / PPL**（稳健性）
- 通信量（字节或等价汇总）
- （可选附录列）`domain_macro_loss`（辅助）

### 表 2：跨模型结果

列：

- 方法
- Qwen3-32B
- Mistral-24B

### 表 3：消融实验（fedplora-oneshot）

行（与 §十四【机制消融】**A 表**、`run_exp_ablation_fedplora.sh` 预设一致；多轮 FedPLoRA 的 B 类消融另见该节 **B 表**）：

- full
- w/o L1-on-A（`yoco_sparse_lambda=0`）
- PCWA k=1（`yoco_pcwa_components=1`）

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

1. `scripts/DataProcessScripts/prepare_*.py`
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
python scripts/DataProcessScripts/build_domain_benchmark.py \
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
