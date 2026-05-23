# FedPLoRA：面向跨域个性化联邦大模型的 LoRA 因子解耦方法

本仓库实现了一个面向联邦个性化微调的 LoRA 方法：**FedPLoRA**。  
核心思想是将 LoRA 更新写为：

\Delta W_i = B_i A_i

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

- `**methods/`**：各 baseline 的服务端聚合与（部分）上传逻辑，**每个方法一个 `.py` 文件**；不含 `__init__.py`（使用隐式命名空间包，便于 `from methods.xxx import ...`）。
- `**tasks/`**：可执行入口：`fed_train_sft.py`（域 SFT）、`fed_train_glue.py`、`fed_train_e2e.py`。
- `**utilities/`**：`data_utils.py`、`models.py`、`utils.py`、训练/评估 `train_eval.py`、联邦状态 `state_dict_ops.py`。
- `**scripts/`**：子目录 `**DataProcessScripts/`**（数据准备、benchmark 构建）、`**RunScripts/**`（`run_domain_sft*.sh`、`run_script.py` 等训练/批量实验入口）。
- `**__pycache__/**`：Python 解释器自动生成的字节码缓存，**可删**；运行后会再次出现，已写入 `.gitignore` 建议不要提交。

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

W_i = W_0 + B_i A_{global}

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


| 领域          | 主训练集                                           | 备选/补充                                           |
| ----------- | ---------------------------------------------- | ----------------------------------------------- |
| `general`   | `allenai/tulu-3-sft-mixture`                   | `teknium/OpenHermes-2.5`, `Open-Orca/SlimOrca`  |
| `math`      | `AI-MO/NuminaMath-CoT`                         | `meta-math/MetaMathQA`                          |
| `code`      | `OpenCoder-LLM/opc-sft-stage1`                 | `bigcode/self-oss-instruct-sc2-exec-filter-50k` |
| `medical`   | `FreedomIntelligence/medical-o1-reasoning-SFT` | 领域 QA 指令数据                                      |
| `legal`     | `lawinstruct/lawinstruct`                      | 法律 QA / StackExchange 风格数据                      |
| `finance`   | `gbharti/finance-alpaca`                       | `FinQA` 指令化版本                                   |
| `education` | `eth-nlped/mathdial` + `ScaleAI/TutorBench`    | tutoring 对话数据                                   |


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


| 领域          | 推荐评测                              |
| ----------- | --------------------------------- |
| `general`   | `IFEval`, `AlpacaEval 2`          |
| `math`      | `GSM8K`, `MATH-500`               |
| `code`      | `HumanEval`, `MBPP`               |
| `medical`   | `MedQA`                           |
| `legal`     | `LegalBench` 子集                   |
| `finance`   | `FinQA`, `FiQA`                   |
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
  domain_benchmark_35c/   # 与 build 的 --output_dir 一致；亦可为 domain_benchmark_7c 等
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
  --output_dir data/domain_benchmark_35c \
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
  utilities/sft_checkpoint_paths.py \
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
  --output_dir data/domain_benchmark_35c \
  --num_clients_per_domain 5 \
  --min_samples_per_client 50 \
  --seed 42
```

推荐产物检查（与 **§11.1** 默认 `--benchmark_dir` 对齐）：

```bash
ls data/domain_benchmark_35c/seed_42
cat data/domain_benchmark_35c/seed_42/domain_stats.json
cat data/domain_benchmark_35c/seed_42/clients.json | head
```

---

## 十一、7 域联邦 SFT 主实验入口

新增主入口：

- [tasks/fed_train_sft.py](tasks/fed_train_sft.py)

它支持两种模式：

1. **直接读取 benchmark**
2. **从原始 JSONL 自动构建 benchmark 再训练**

### 11.0 终版约定（35c、指标、GPU、一次训练多次 eval）

- **数据**：主实验与各训练类脚本默认 `**data/domain_benchmark_35c/seed_42`**（7 域 × 每域 5 客户端）。`configs/domain_sft.env` / `domain_sft_pilot.env` 已对齐。
- **7 域主实验**：**§11.1** 保留「每个 `agg_type` 一条」的手敲命令；指标 JSON 顶层含 `**recommended_kpis`**（索引 **token 准确率、PPL、`communication`、FedPLoRA-Oneshot 的 `fedplora_oneshot_conflict`**），与 `**recommended_primary_metrics**` / 各轮 `**rounds[]**` 字段一致。
- **个性化实验**：脚本 `**scripts/RunScripts/run_exp_personalization.sh`**；指标与主实验同一套 acc / PPL / communication / conflict（oneshot 有），并额外有个性化字段；读同一 JSON 中的 `**recommended_kpis`**（含 `personalization_gaps`）。
- **通信–性能实验**：脚本 `**scripts/RunScripts/run_exp_comm_profile.sh`**，指标仍为各 `**agg_type` 的 `down_bytes_per_client` / `up_bytes_per_client`**（与现实现一致）。
- **消融**：`**run_exp_ablation_oneshot_35c.sh`** / `run_exp_ablation_fedplora.sh` 默认跑 **wo_sparse / wo_conflict / wo_anchor** 三组（full 对照主实验 §11.1）；已默认带 `**EVAL_MAX_BATCHES`**。
- **GPU**：上述脚本均 `**source configs/cuda_resolve.inc.sh`**：优先环境变量 `**CUDA_DEVICES`**，否则可选 **命令行第二参数（或 comm 脚本的第一参数）** 传 `0`、`1`、`0,1`；再否则 **nvidia-smi 选空闲显存最大的单卡**。
- **加快 eval**：全局环境变量 `**EVAL_MAX_BATCHES=50`**（已在 `domain_sft_*.env` 与各 RunScripts 默认）；主表全量测评前设为 `**0`** 或 **unset** 并去掉命令行中的 `--eval_max_batches`。
- **一次训练，多次实验**：未手填 `--save_run_checkpoint_dir` 时，Python 默认把 bundle 写到 **仓库同级** `**../trained_models/<stem>/`**（无时间戳）。两类落盘与自动恢复：① eval 前（每轮聚合后、`snapshots/round_XXX_post_agg/`）：与紧接着要跑的 `_sft_eval_phase` 同一份 `global_shared.pt` / `full_clients.pt` + `clients/`** + meta，仅缺当轮 eval 指标；若再次启动且 meta 与当前 CLI 一致，则 跳过训练、只做 eval-only（防 eval 阶段崩溃白训）。② eval 后（全部 rounds 跑完且已写 metrics JSON 之后、根目录最终保存）：根下 `**checkpoint_ok.json**` 的 `**checkpoint_phase: final**`；若检测到，则 **训练与评估均跳过**（本方法本次配置已完整跑完）。失败落盘为同级 `**…_failed/`**。防白训靠 `save_run_checkpoint_dir`（默认自动启用）与 `artifacts_{N}c/sft_metrics/`；`**--save_client_state_to_disk`** 仅用于 **FedP / FedSA / FedALT 磁盘顺序协议**（见下），与 YOCO 无关。手动复评：`**python tasks/fed_train_sft.py --eval_only_from_checkpoint <子目录或根> ...`**。强制重训：`**--force_retrain`**。**  
**哪些实验能复用该目录见下表（个性化补评、改 eval 截断等 能；换方法、通信 profiling 不能 / 不必）。Oneshot 的 conflict 摘要会写入 `**run_checkpoint_meta.json`**，eval-only 时会写回 `**rounds[].fedplora_oneshot_conflict**`。

### 11.1 直接用准备好的 benchmark 训练（推荐：逐方法手敲命令）

以下命令均在**仓库根目录**执行。主表默认使用 **35 客户端** benchmark：`--benchmark_dir data/domain_benchmark_35c/seed_42`（与 `build_domain_benchmark.py` 里 `num_clients_per_domain=5` × 7 域一致）。**当前一轮对比实验统一 `--rounds 1`**（`configs/domain_sft.env` 与 `fed_train_sft.py` 入口均会强制为 1）。不写 `--save_run_checkpoint_dir` 时，权重默认落到 `**../trained_models/<stem>/**`，并按上节规则 **自动跳过已完成的训练 / 或仅补 eval**（防白训）；需要固定到其它目录时再显式传 `--save_run_checkpoint_dir`。

**复制下面任一条 `python` 命令之前，先加载环境（与本地数据 / 本地模型一致）**

单条手敲命令**不会**自动 `source` env。请先 `**source configs/domain_sft.env`**：其中 `**MODEL_PATH`** 指向本地基座目录（默认与 Meta-Llama 示例一致，可按机器修改该文件）。若 `--model` 仍写成 Hub 上的 repo id、或本地路径不存在，在无外网时会触发 `AutoTokenizer.from_pretrained` 连 `huggingface.co` 失败。

```bash
cd /path/to/FedPLoRA   # 本仓库根目录（含 tasks、configs）
set -a
source configs/domain_sft.env
set +a
# 下列 python 命令中 --model 已写为与 configs/domain_sft.env 相同的本地路径；换机器请改该路径或先改 env 再 sed/手改。
```

`configs/domain_sft.env` 与批量脚本 `run_domain_sft_baselines*.sh`、`run_exp_*.sh` 自动加载的是**同一份**默认（含 `BENCHMARK_DIR`、`EVAL_MAX_BATCHES` 等）。单机一键跑单次实验仍可用 `configs/domain_sft_pilot.env`（`run_domain_sft.sh` 会自行 source）。若仍需 `trust_remote_code`，在 `domain_sft.env` 中设 `TRUST_REMOTE_CODE=1` 或命令行加 `--trust_remote_code`。

**手敲 `python tasks/fed_train_sft.py ...` 与 `run_domain_sft_batch_*.sh` 是否等价？** 等价于「同一入口 + 同一组 CLI 参数」。批量脚本只是把 `domain_sft.env` 里的 `MODEL_PATH`、`ROUNDS`、`BENCHMARK_DIR`、`--eval_max_batches` 等展开成数组；**未**传 `--save_run_checkpoint_dir` 时，Python 仍会默认写到 `../trained_models/<stem>/`，每轮后写 `snapshots/`，训完写 `checkpoint_ok.json` + `artifacts_{N}c/sft_metrics/*.json`。你只要在跑手敲命令前 `source configs/domain_sft.env`（或把 env 里的值抄进命令行），能力就与脚本一致。

**若 shell 只打印 `Killed`（无 Python traceback）**：多为 **Linux OOM killer**（系统内存或交换空间耗尽）。此前 `normal` / `yoco` / `fedex` / `ffa` 等在每轮会把每个客户端的 **整份** `state_dict()`（含冻结基座）克隆进 RAM 列表，客户端一多极易 OOM。当前实现与 `fedplora` 侧一致：这些「内存聚合」路径只缓存 `**requires_grad` 的可训练快照**（LoRA + 任务头），聚合与评测仍通过 `load_partial_state_dict` 与当轮全局基座合并，**训练仍只在 GPU 上保留一份基座**。若仍内存紧张，可再调低 `DATALOADER_NUM_WORKERS` / `batch_size` / `max_seq_length`，或改用带 `--save_client_state_to_disk` 的磁盘顺序协议（`fedplora` / `fedplora-oneshot` / `fedsa_lora` / `fedalt`）。

**加快每轮「评估 / 测评」阶段（只影响 eval，不影响训练步）**

- 下面每条命令都带有 `**--eval_max_batches 50`**：每个 eval 子循环最多跑 50 个 batch（按「域 × 客户端 × dataloader」截断）。调参或排队时可保留；写论文主表前可改为更大、`0`（或不写该参数）表示**全量 eval**。
- 客户端数更少时 eval 也更短：可把 `--benchmark_dir` 换成 `data/domain_benchmark_7c/seed_42` 等（见 **§11.1.1**）。

**保存训练结果（run checkpoint，默认启用）**

- 默认 bundle 根：`**<仓库>/../trained_models/<stem>/`**。可用 `**TRAINED_MODELS_ROOT`** / `**--trained_models_root**` 改父目录；完全不要自动落盘：`**--no_auto_save_run_checkpoint**`。
- **eval 前 vs eval 后（内容是否一样？）**：**权重张量与 eval 前一刻一致**。eval 前快照里已是聚合后的 `global_shared`（或 `full_clients`）及各客户端磁盘状态拷贝，**就是**紧接着 `_sft_eval_phase` 会加载做前向的那套；eval 本身不反传、不改权重。eval 后在根目录再写一遍最终 bundle，**权重与当轮聚合结果相同**（最后一轮），额外多的是已写入磁盘的 **metrics JSON 路径**记在 meta 里。因此用 eval 前快照做 `--eval_only_from_checkpoint`，与「当时若 eval 没挂跑出来的前向」一致，**不会白训聚合阶段**。
- **自动恢复顺序**（`--force_retrain` 时均不启用）：若根目录 `checkpoint_ok.json` 为 `**final`** → 本方法本次配置 **训练 + eval 均跳过**；否则若存在最新的 `snapshots/round_XXX_post_agg/` 且其中 `checkpoint_ok` + meta 与当前 CLI 一致 → **只跳过训练、跑 eval-only**；否则正常训练。
- 每条目录内有 `**run_checkpoint_meta.json`** + `**full_clients.pt`**（内存聚合方法下为各客户端 可训练快照 列表，`meta.memory_agg_client_payload` 为 `trainable_only`）或 `**global_shared.pt` + `clients/`****（磁盘协议）。每轮 eval 前另有 `**snapshots/round_XXX_post_agg/`**（除非 `--skip_post_agg_snapshots`）。失败为 `**…_failed/`** + `checkpoint_failed.json`。
- 需要磁盘协议的方法（`**fedplora`**、`**fedplora-oneshot`**、`**fedsa_lora**` / `**fedsa**`、`**fedalt**`）：必须保留 `**--save_client_state_to_disk**`，否则写 checkpoint 时可能缺 `client_*.pt`。`**yoco`** / `**normal`** 等为内存聚合、**不需要**该 flag；`full_clients.pt` 存的是可训练张量而非整模的 N 份重复。

`**../trained_models/<stem>/` 在哪些实验里能重复用？**


| 实验                                                          | 能否直接用该 checkpoint？  | 说明                                                                                                                                                                                                                                       |
| ----------------------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **同方法、补开 `--eval_personalization_metrics`**                 | **能**               | `python tasks/fed_train_sft.py --eval_only_from_checkpoint <目录> --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B --benchmark_dir ... --eval_personalization_metrics --eval_max_batches 50`（路径与 checkpoint 一致；`--agg_type` 以 meta 为准） |
| **同方法、只刷新 domain-macro eval（改 `eval_max_batches`、全量 eval）** | **能**               | 同上，不加 `--eval_personalization_metrics` 即可                                                                                                                                                                                                |
| **通信–性能脚本 `run_exp_comm_profile.sh`**                       | **不依赖此 checkpoint** | 只按 PEFT 形状估字节，**不需要**你训好的权重                                                                                                                                                                                                              |
| **换 `agg_type`、换聚合、换方法对比**                                  | **不能**              | 每种方法各训各的目录；**不能**把 `normal` 的 checkpoint 当成 `fedplora` 用                                                                                                                                                                                 |
| **消融（改 λ / 冲突门控等训练超参）**                                     | **一般要重训**           | 权重变了就不是同一次 run；eval-only **不重训**                                                                                                                                                                                                         |


`**fedplora-oneshot`、`yoco`、`fedalt`（与官方实现对齐后的协议）**


| 方法                 | 上行（客户端→服务端）                        | 下行（服务端→客户端）                        | 服务端聚合                                                                                    | 磁盘协议                                              |
| ------------------ | ---------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `fedplora-oneshot` | LoRA **A** + 头 + 行统计               | 全局 **A** + 头                       | 冲突门控（`aggregate_models_fedplora_oneshot`）                                                | 需要 `--save_client_state_to_disk`（B 本地）            |
| `yoco`             | 全量可训练 LoRA **A+B** + 头（通信统计不含冻结基座） | 聚合后全局 LoRA（单轮）                     | **FedMLLM `conflict` → `aggregate_lora_weights`**（默认）；`--yoco_aggregate_mode fedavg` 为旧版 | **不需要**；checkpoint 用 `full_clients.pt`            |
| `fedalt`           | Individual LoRA **A+B** + 头        | 个性化 RoTW **A+B**（leave-one-out 均值） | `aggregate_models_fedalt`（[FedALT/server.py](https://github.com/jmbian/FedALT)）          | 需要 `--save_client_state_to_disk`（local + RoTW 快照） |


- `fedplora-oneshot` 与 `yoco` 均为单轮（入口强制 `--rounds 1`），但上传形态与服务端聚合**已分叉**（上表）。
- `yoco`：默认 `--yoco_aggregate_mode conflict`（FedMLLM B 相似度加权聚合）；本地 `--yoco_sparse_lambda`（A）+ `--yoco_sign_lambda`（B 符号约束）。**所有方法通信统计均不含冻结基座**；内存聚合且服务端产出全局 LoRA 的方法（`normal` / `ffa` / `flora` / `flexlora` / `feddat` / `yoco`）**评测默认用聚合后的全局 LoRA**。复现旧「各客户端本地 LoRA 评测」加 `--memory_agg_eval_use_local_clients`（或 YOCO 专用别名 `--yoco_eval_use_local_clients`）。
- `fedalt`：RoTW 已计算并落盘（`client_*_rotw.pt`）；双分支 forward + mixer（论文 `lora_route`）**尚未**接入，评测仍用各客户端 local A+B。
- 别名：`fedsa`≡`fedsa_lora`；v3 亦可用 `v3_lite` / `v3_cluster` / `v3_rpca`。
- 已移除域 SFT 主线的 `fedex`、`hetlora`、`fdlora`、`lora_a2`（见 §11.1 更新列表）。

公共说明：

- 顺序客户端训练；需 **磁盘顺序协议** 的方法（`fedplora`、`fedplora-oneshot`、`fedsa_lora`、`fedalt`）务必加 `**--save_client_state_to_disk`**（且与上表 保存 checkpoint 联用）。`**yoco`** / `**normal**` 等全量协议方法不必加，但仍依赖默认 `save_run_checkpoint_dir` 防白训。
- 若 Transformers 加载 Llama 报错，可在命令末尾追加 `**--trust_remote_code`**。
- 需要 **个性化指标** 时：训练阶段可加 `--eval_personalization_metrics`，或训完用 `**--eval_only_from_checkpoint`** 再开（见上表）。

**公共参数（下列每条命令均包含）**：`--rounds 1`、`--benchmark_dir data/domain_benchmark_35c/seed_42`、`--eval_max_batches 50`、`--gradient_checkpointing`；未写 `--save_run_checkpoint_dir` 时自动落盘到 `../trained_models/<stem>/` 并启用防白训。将 `CUDA_VISIBLE_DEVICES=0` 换成你的 GPU。

#### 1) `normal`

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type normal \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --seed 42
```

#### 2) `ffa`

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type ffa \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --seed 42
```

#### 3) `flora`（NeurIPS 2024；均值 ΔW + SVD）

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type flora \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --seed 42
```

#### 4) `flexlora`（[FlexLoRA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1a134b50202088aa8c595cc99b310e5a-Abstract-Conference.html)；样本加权 ΔW + SVD）

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type flexlora \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --seed 42
```

#### 5) `feddat`（[FedDAT](https://ojs.aaai.org/index.php/AAAI/article/view/29007)；样本加权 FedAvg + 教师近端正则）

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type feddat \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --feddat_teacher_lambda 0.01 \
  --seed 42
```

#### 6) `yoco`（单轮；FedMLLM `conflict` 聚合 + B 符号约束）

```bash
CUDA_VISIBLE_DEVICES=1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type yoco \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --yoco_sparse_lambda 1e-4 \
  --yoco_aggregate_mode conflict \
  --yoco_conflict_method avgm \
  --yoco_sign_lambda 0.01 \
  --seed 42
```

**复用旧 checkpoint（不重训本地）**：`--eval_only_from_checkpoint <bundle>`（内存聚合方法会从 `full_clients.pt` 重聚合）。旧评测协议：加 `--memory_agg_eval_use_local_clients`。要对齐新 YOCO（`conflict` + sign 损失）本地需重训。

#### 7) `fedsa_lora`（可用 `--agg_type fedsa`）

```bash
CUDA_VISIBLE_DEVICES=1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedsa_lora \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk \
  --seed 42
```

#### 8) `fedalt`

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedalt \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk \
  --seed 42
```

#### 9) `fedplora-oneshot`（v2 冲突门控；需 `--save_client_state_to_disk`）

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedplora-oneshot \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_conflict_threshold 0.35 \
  --oneshot_conflict_blend 1.0 \
  --seed 42
```

#### 10) `fedplora_v3_lite`（v3 残差冲突；单全局 A，见 [FedPLoRAOSv3_README.md](FedPLoRAOSv3_README.md)）

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedplora_v3_lite \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --v3_conflict_quantile 0.8 \
  --v3_gate_temperature 0.05 \
  --v3_conflict_blend 1.0 \
  --seed 42
```

#### 11) `fedplora_v3_cluster`（v3 域簇个性化 A 下发）

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedplora_v3_cluster \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --v3_conflict_quantile 0.8 \
  --v3_gate_temperature 0.05 \
  --v3_cluster_mode domain_prior \
  --v3_cluster_lambda_min 0.2 \
  --v3_cluster_lambda_max 1.0 \
  --seed 42
```

#### 12) `fedplora_v3_rpca`（v3 低秩公共 + 稀疏簇残差）

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedplora_v3_rpca \
  --rounds 1 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --v3_conflict_quantile 0.8 \
  --v3_gate_temperature 0.05 \
  --v3_cluster_mode domain_prior \
  --v3_rpca_rank 1 \
  --v3_sparse_quantile 0.8 \
  --seed 42
```

**v3 三版对比建议**：三条命令除 `--agg_type` 与 cluster/rpca 专有参数外保持一致；跑完后对比各方法 metrics JSON 中的 `domain_macro_token_accuracy`、`worst_domain_token_accuracy` 与 `fedplora_v3_stats`。也可一键：`bash scripts/RunScripts/run_domain_sft_batch_group5_fedplora_v3.sh 35 0`。

### 11.1.1 换一套 benchmark（7c / 14c / 21c）


| 划分               | `--benchmark_dir`                   |
| ---------------- | ----------------------------------- |
| 7 客户端（每域 1 客户端）  | `data/domain_benchmark_7c/seed_42`  |
| 14 客户端           | `data/domain_benchmark_14c/seed_42` |
| 21 客户端           | `data/domain_benchmark_21c/seed_42` |
| 35 客户端（§11.1 默认） | `data/domain_benchmark_35c/seed_42` |


在同一套原始 JSONL 与相同 `val_ratio` / `test_ratio` / `min_samples_per_client` 下，各域 `domain_stats.json` 里的 `**n_total` 不变**，只是域内 shard 数量不同；**7c** 时 eval 前向组合更少（约 **7 域 × 7 客户端**），通常比 35c 更省墙钟。将 §11.1 各命令中的 `**--benchmark_dir`** 换成上表路径即可，其余参数不变。

**生成 7c 示例**（输入 JSONL 按你实际路径调整）：

```bash
python scripts/DataProcessScripts/build_domain_benchmark.py \
  --input_jsonl data/raw/domain_7_all.jsonl \
  --output_dir data/domain_benchmark_7c \
  --num_clients_per_domain 1 \
  --min_samples_per_client 50 \
  --seed 42
```

### 11.2 从原始 JSONL 直接构建并训练

在首次没有 benchmark 时，可用下面命令**先构建再训练**（`--model` 换成你的基座；`--agg_type` 可换成 §11.1 中任意一种）。开启 `--build_benchmark` 时，训练使用的 `split_dir` 由构建结果决定（控制台会打印 `[benchmark] loaded from ...`），**不要**再单独传 `--benchmark_dir`。下面示例把产物写到 `**data/domain_benchmark_35c`**，与 §11.1 默认 `--benchmark_dir` 对齐（`seed_42` 子目录由脚本生成）。

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model /data/yaominghao/gb/models/Meta-Llama-3.1-8B \
  --build_benchmark \
  --benchmark_jsonl data/raw/domain_7_all.jsonl \
  --benchmark_output_dir data/domain_benchmark_35c \
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
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_client_state_to_disk \
```

### 11.3 可选：一键脚本（无人值守时再开）

主实验推荐直接复制 **§11.1** 的命令逐条跑；下列脚本适合**固定顺序串行**或 CI，不必作为主路径：

- [scripts/RunScripts/run_domain_sft.sh](scripts/RunScripts/run_domain_sft.sh)（单次跑一个 `AGG_TYPE`）
- [scripts/RunScripts/run_domain_sft_baselines.sh](scripts/RunScripts/run_domain_sft_baselines.sh)（按固定顺序自动串行多种 `AGG_TYPE`）
- **主实验推荐（四脚本，全量训练 + 每方法保存 checkpoint，见 §11.3.3）**  
  - [run_domain_sft_batch_group1_oneshot_fedalt.sh](scripts/RunScripts/run_domain_sft_batch_group1_oneshot_fedalt.sh)：`fedplora-oneshot`、`fedalt`  
  - [run_domain_sft_batch_group2_yoco_fedsa.sh](scripts/RunScripts/run_domain_sft_batch_group2_yoco_fedsa.sh)：`yoco`、`fedsa_lora`  
  - [run_domain_sft_batch_group3_normal.sh](scripts/RunScripts/run_domain_sft_batch_group3_normal.sh)：`normal`  
  - [run_domain_sft_batch_group4_flora_ffa.sh](scripts/RunScripts/run_domain_sft_batch_group4_flora_ffa.sh)：`flora`、`ffa`  
  - 兼容旧路径（等价于依次执行 group3 + group4）：[run_domain_sft_batch_group3_normal_flora_ffa.sh](scripts/RunScripts/run_domain_sft_batch_group3_normal_flora_ffa.sh)  
  - 共享逻辑：`[_run_domain_sft_batch.inc.sh](scripts/RunScripts/_run_domain_sft_batch.inc.sh)`、`[_fed_train_speed.inc.sh](scripts/RunScripts/_fed_train_speed.inc.sh)`
- [scripts/RunScripts/run_domain_sft_baselines1.sh](scripts/RunScripts/run_domain_sft_baselines1.sh)（旧版四合一分组：oneshot、fedalt、flora、normal；无统一 checkpoint 命名时可改用 §11.3.3 四脚本）
- [scripts/RunScripts/run_domain_sft_baselines2.sh](scripts/RunScripts/run_domain_sft_baselines2.sh)（旧版：yoco、fedsa_lora、ffa）
- 扩展实验脚本（详见 **§十四**）：`[run_exp_post_main_35c.sh](scripts/RunScripts/run_exp_post_main_35c.sh)`（通信+个性化）、`[run_exp_ablation_oneshot_35c.sh](scripts/RunScripts/run_exp_ablation_oneshot_35c.sh)`（oneshot v2 三模块消融）、`[run_exp_personalization.sh](scripts/RunScripts/run_exp_personalization.sh)`、`[run_exp_comm_profile.sh](scripts/RunScripts/run_exp_comm_profile.sh)`
- [configs/domain_sft_pilot.env](configs/domain_sft_pilot.env)（单机 pilot 默认环境）
- [configs/domain_sft.env](configs/domain_sft.env)（批量 baseline 默认环境）

默认模型目录已配置为（可按机器修改 env 文件或命令行覆盖）：

`/data/yaominghao/gb/models/Meta-Llama-3.1-8B`

**聚合类型说明**：多轮 `**fedplora`** 为 **FedP-LoRA**（上传 `A`+头+行统计；`aggregate_models_fedplora`；本地对齐/近端/正交正则）。`**fedplora-oneshot`**：仍只上传 `**A`+头**（`B` 本地），单轮 **冲突门控**（`aggregate_models_fedplora_oneshot`）。`**yoco`**：上传全量 LoRA A+B+头，单轮 FedMLLM `conflict` 聚合（`aggregate_models_yoco`，默认 `aggregate_lora_weights`）；与 oneshot 不是同一路径。`**fedalt`**：上传 Individual **A+B**，下行个性化 RoTW **A+B**（`aggregate_models_fedalt`）。批量脚本 `_run_domain_sft_batch.inc.sh` 仅对 FedP/FedSA/FedALT 自动加 `--save_client_state_to_disk`，**不含** `yoco`。

#### 11.3.1 单次实验（pilot）

`scripts/RunScripts/run_domain_sft.sh` 会**自动** `cd` 到仓库根目录，并若存在则 **source `configs/domain_sft_pilot.env`**（无需先手动 `source`）。直接执行：

```bash
bash /path/to/FedPLoRA/scripts/RunScripts/run_domain_sft.sh
```

不写第一个参数且未在环境里设置 `CUDA_DEVICES` 时，会由 `configs/cuda_resolve.inc.sh` 尝试**自动选空闲显存最大的 GPU**（见下文）。其它扩展脚本示例：

```bash
bash scripts/RunScripts/run_domain_sft_baselines2.sh 35 1
bash scripts/RunScripts/run_exp_personalization.sh 35 1
bash scripts/RunScripts/run_exp_ablation_fedplora.sh 35 1
bash scripts/RunScripts/run_exp_comm_profile.sh 1
bash scripts/RunScripts/run_domain_sft_baselines.sh 1
```

可选**第一个参数**指定 GPU（单卡或逗号列表，如 `1` 或 `0,1`）。不写且未在环境里设置 `CUDA_DEVICES` 时，由 `[configs/cuda_resolve.inc.sh](configs/cuda_resolve.inc.sh)` 在本机有 `nvidia-smi` 的情况下**自动选用空闲显存最大的 GPU**；无 GPU 工具时回退为 `0`。若要强制不用自动选卡：`AUTO_CUDA_PICK=0 bash ...`（此时使用 `CUDA_DEVICES_FALLBACK`，默认 `0`）。

`domain_sft_pilot.env` 中已设置 `MODEL_PATH=/data/yaominghao/gb/models/Meta-Llama-3.1-8B`、`AGG_TYPE=fedplora` 等。若需临时覆盖环境变量，仍可在命令前导出（会覆盖 env 文件中的同名字段）；其中 `**CUDA_DEVICES` 仍可通过前缀指定**，优先级高于自动选卡与第一个参数：

```bash
MODEL_PATH=/data/yaominghao/gb/models/Meta-Llama-3.1-8B \
AGG_TYPE=fedplora \
CUDA_DEVICES=1 \
ROUNDS=20 \
BATCH_SIZE=1 \
bash /path/to/FedPLoRA/scripts/RunScripts/run_domain_sft.sh
```

#### 11.3.2 自动批量 baseline（默认顺序）

[scripts/RunScripts/run_domain_sft_baselines.sh](scripts/RunScripts/run_domain_sft_baselines.sh) 同样会 **cd 到仓库根** 并自动 **source `configs/domain_sft.env`**。当前顺序为：**先 `fedplora-oneshot`，再 `fedplora`**，随后 `normal`、`ffa`、`fedex`（后三项在脚本里顺序可任意调整）。

```bash
bash /path/to/FedPLoRA/scripts/RunScripts/run_domain_sft_baselines.sh
# 指定仅用 1 号物理卡：
bash /path/to/FedPLoRA/scripts/RunScripts/run_domain_sft_baselines.sh 1
```

`domain_sft.env` 中已写入 `MODEL_PATH`（**不再**写入 `CUDA_DEVICES`，避免覆盖命令行前缀）。需要临时改 GPU 或轮数时：

```bash
CUDA_DEVICES=1 ROUNDS=10 bash /path/to/FedPLoRA/scripts/RunScripts/run_domain_sft_baselines.sh
```

#### 11.3.3 分组 baseline 脚本（可选客户端数 7 / 14 / 21 / 35）

以下脚本均会 **cd 到仓库根**、自动 **source `configs/domain_sft.env`**（若存在），并按**第一个命令行参数**选择 benchmark；**第二个可选参数**为 GPU 索引或列表（如 `0`、`1`、`0,1`）；若省略则由 `configs/cuda_resolve.inc.sh` 在未设置 `CUDA_DEVICES` 时根据 `nvidia-smi` 选空闲显存最大的 GPU。


| 第一个参数    | `benchmark_dir`                     |
| -------- | ----------------------------------- |
| `7`      | `data/domain_benchmark_7c/seed_42`  |
| `14`     | `data/domain_benchmark_14c/seed_42` |
| `21`     | `data/domain_benchmark_21c/seed_42` |
| `35`（默认） | `data/domain_benchmark_35c/seed_42` |


**推荐：四脚本（全量本地 epoch；checkpoint 由 Python 默认写入 `../trained_models/<stem>/`）**

批量入口 `_run_domain_sft_batch.inc.sh` 不再传 `--save_run_checkpoint_dir`；每种 `agg_type` 对应 **不同 `<stem>`**（方法 + 模型 + benchmark 尾 + r/e/seed），与手敲命令默认一致。可选 `**export TRAINED_MODELS_ROOT=/绝对路径**` 改父目录。

另会通过 `_fed_train_speed.inc.sh` 传入 `domain_sft.env` 中的加速项（如 `DATALOADER_NUM_WORKERS`、`DATALOADER_PERSISTENT_WORKERS`、`ATTN_IMPLEMENTATION=sdpa`）；**不设** `TRAIN_MAX_STEPS_PER_CLIENT` / `MAX_TRAIN_SAMPLES_PER_CLIENT` 即为全量训练步数。


| 脚本                                                                                                                | 串行 `agg_type`                                                   |
| ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [run_domain_sft_batch_group1_oneshot_fedalt.sh](scripts/RunScripts/run_domain_sft_batch_group1_oneshot_fedalt.sh) | `fedplora-oneshot` → `fedalt`                                   |
| [run_domain_sft_batch_group2_yoco_fedsa.sh](scripts/RunScripts/run_domain_sft_batch_group2_yoco_fedsa.sh)         | `yoco` → `fedsa_lora`                                           |
| [run_domain_sft_batch_group3_normal.sh](scripts/RunScripts/run_domain_sft_batch_group3_normal.sh)                 | `normal`                                                        |
| [run_domain_sft_batch_group4_flora_ffa.sh](scripts/RunScripts/run_domain_sft_batch_group4_flora_ffa.sh)           | `flora` → `flexlora` → `ffa` → `feddat`                         |
| [run_domain_sft_batch_group5_fedplora_v3.sh](scripts/RunScripts/run_domain_sft_batch_group5_fedplora_v3.sh)       | `fedplora_v3_lite` → `fedplora_v3_cluster` → `fedplora_v3_rpca` |


**运行示例**（在仓库根目录，或任意路径用 bash 调用下列绝对/相对路径均可；脚本内部会 `cd` 到仓库根）：

```bash
set -a && source configs/domain_sft.env && set +a

# 35 客户端，自动选 GPU
bash scripts/RunScripts/run_domain_sft_batch_group1_oneshot_fedalt.sh 35 0
bash scripts/RunScripts/run_domain_sft_batch_group2_yoco_fedsa.sh 35 1
bash scripts/RunScripts/run_domain_sft_batch_group3_normal.sh 35 0 
bash scripts/RunScripts/run_domain_sft_batch_group4_flora_ffa.sh 35 1
bash scripts/RunScripts/run_domain_sft_batch_group5_fedplora_v3.sh 35 0

# 7 客户端 + 指定物理 GPU 0
bash scripts/RunScripts/run_domain_sft_batch_group1_oneshot_fedalt.sh 7 0

# 覆盖 trained_models 父目录或轮数（批量脚本会传给 Python：`TRAINED_MODELS_ROOT` → `--trained_models_root`）
TRAINED_MODELS_ROOT=/data/trained_models ROUNDS=10 \
  bash scripts/RunScripts/run_domain_sft_batch_group2_yoco_fedsa.sh 35
```

**旧版两脚本**（仍可用；默认**不**写 `run_domain_sft_batch_`* 那种结构化 checkpoint 目录，除非你自行改脚本）：`run_domain_sft_baselines1.sh`（oneshot、fedalt、flora、normal）、`run_domain_sft_baselines2.sh`（yoco、fedsa_lora、ffa）。

```bash
bash scripts/RunScripts/run_domain_sft_baselines1.sh 35
bash scripts/RunScripts/run_domain_sft_baselines2.sh 35 1
```

不传第一个参数时等价于 `35`。其它超参仍可通过环境变量覆盖（如前缀 `CUDA_DEVICES=`、`ROUNDS`、`EVAL_MAX_BATCHES`；`EVAL_MAX_BATCHES=0` 为评测全量 batch，仅影响 eval 时长）。

训练结束后，每轮 `**domain_macro_token_accuracy` / `domain_macro_perplexity` / `domain_macro_loss**` 等会写入（默认路径在加载 benchmark 后按客户端数 **自动** 带上后缀，子目录名不变）：

```text
artifacts_{num_clients}c/sft_metrics/
```

例如 35 个客户端时为 `artifacts_35c/sft_metrics/`；客户端状态目录同理为 `artifacts_35c/domain_client_states/`。若你在命令行 **显式** 传入非默认的 `--metrics_output_dir` / `--client_state_dir`，则不会改写。控制台会打印 `[setup] client_state_dir=... metrics_output_dir=...`。

### 11.4 其它模型（Qwen 等）上的 baseline

逐方法命令仍以 **§11.1** 为准：其中 `--model` 已写为 Meta-Llama 本地目录；若在 Qwen 等上跑，请改为对应**本地**权重目录（例如 `../../models/qwen3-14b`），按需调整 `**--target_modules`**（Qwen 常用 `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj`），并保留 `**--eval_max_batches`** 以加快 eval。批量脚本用法见 **§11.3**。

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
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedplora \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
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
  domain_benchmark_35c/
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


| 主线            | 要回答的问题                                                   | 是否训练             | 典型脚本入口                                                                                         | 产物位置（默认）                                                      | **优先阅读的指标 / 字段**                                                                                                                                                                                                                                                                                                                                                                              |
| ------------- | -------------------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **【7域主实验】**   | 各 `agg_type` 在 7 域上的整体 LM 表现谁更好                          | 是                | **§11.1** 手敲命令为主；可选 §11.3 `run_domain_sft_batch_group*.sh`（推荐）或 `run_domain_sft_baselines*.sh` | `artifacts_{N}c/sft_metrics/<agg>_<model>_*_r*_e*_seed*.json` | **主指标（推荐写进主表）**：每轮 `**domain_macro_token_accuracy`、`domain_macro_perplexity`** 及 `**worst_domain_token_accuracy`、`worst_domain_perplexity`**（均有 `best_*` 追踪最优轮）。辅助：`domain_macro_loss` / `worst_domain_loss`（与 NLL 一致，仍保留）。JSON 顶层含 `**recommended_primary_metrics`** 字段名列表。分域见 `rounds[].domain_metrics[<domain>]`。**通信**：顶层 `communication.*`。日志 `[eval]` 以 `primary_*` 打头、`aux_*loss` 为辅助。 |
| **【个性化收益分析】** | 本域 vs 跨域 gap（时间紧：§十四 **仅 fedplora-oneshot + yoco** 手敲命令） | **否**（eval-only） | §十四最小集命令，或 `PERSONALIZATION_AGG_LIST=... run_exp_personalization.sh`                           | `artifacts_35c/sft_metrics/*_eval_ckpt_*.json`                | `personalization_gap_*`；**不需重训**。                                                                                                                                                                                                                                                                                                                                                             |
| **【通信-性能实验】** | 各方法**单客户端单轮**上下行字节表（**不必** 12 个都跑；见 §十四）                 | **否**            | `run_exp_comm_profile.sh`                                                                      | `artifacts_35c/comm_profile/sft_comm_35c.json`                | 画 Pareto / 方法表；**不读 checkpoint**、**不训练**。                                                                                                                                                                                                                                                                                                                                                     |
| **【机制消融】**    | FedPLoRA-Oneshot v2 **3 模块**（服务端冲突门控 / 本地稀疏 / A0 锚定）                         | 是（默认 3 组）      | `run_exp_ablation_oneshot_35c.sh` 或 `run_exp_ablation_fedplora.sh 35 [gpu]`                   | `artifacts_{N}c/sft_metrics_oneshot_ablation/<tag>/`          | **full** 用主实验 §11.1；本脚本默认只跑 `wo_*` 三组；见 §十四。                                                                                                                                                                                                                                                                                                                                                  |


**总通信量（直觉）**：`utilities/utils.py` 中 `estimate_round_communication_bytes` 给出的是 **每名客户端、每一轮** 的下行/上行字节；单轮全集群流量量级约为 `**num_clients × (down_bytes_per_client + up_bytes_per_client)`**（与函数注释一致）。论文里若写「总上传」，请自行用 `**up_bytes_per_client × num_clients × rounds`**（若各轮相同）或按实现说明截取。

---

#### 主实验 JSON 里的 `communication` 与「通信-性能」脚本是否重复？要不要删？

- **公式相同**：两处都调用 `**estimate_round_communication_bytes`**（`fed_train_sft.py` 启动时写入 metrics；`print_sft_comm_profile.py` 批量打印）。数值在相同 `agg_type`、相同模型与 LoRA 配置下应对齐。
- **用途不同（建议都保留）**：
  - **主实验 JSON**：与该次 run 的 `**domain_macro_loss` 等写在同一文件**，可复查「这次实验到底用的哪套通信假设」，便于归档与答辩溯源。
  - `**run_exp_comm_profile`**：零训练，一次列出**多种** `agg_type`，专门用来做通信对比表 / Pareto 轴，省 GPU 时间。

因此 **不建议从主实验 metrics 中删除 `communication` 块**；论文主表若以通信章节脚本为准，只需注明数字来源与 `estimate_round_communication_bytes` 一致即可。

---

### 【7域主实验】

对应原 **E3**，主结果表与 baseline 对比（`fedplora`、`fedplora-oneshot`、`normal`、`ffa`、`fedex` 及 README §11 中其它 `agg_type`）。

- **数据**：`data/domain_benchmark_35c/seed_42`（或 `7c` / `14c` / `21c` 做客户端规模扫描）。
- **命令**：**§11.1** 逐条 `fed_train_sft.py`；可选 **§11.3** 批量脚本（`run_domain_sft*.sh` 等，首参/次参与 GPU 见该节与 `configs/cuda_resolve.inc.sh`）。
- **指标（汇总）**：见上表「7域主实验」行。**论文主表建议以 `domain_macro_token_accuracy`（越高越好）与 `domain_macro_perplexity`（越低越好）为主**；`worst_domain_`* 对应「最难点域」的稳健性。`**domain_macro_loss` 仅作辅助**（与交叉熵一致）。`rounds[]` 内字段顺序已把 token 准确率与 PPL 放在 loss 之前；metrics JSON 顶层 `**recommended_primary_metrics`** 列出推荐主字段名。`**domain_metrics`** 仍为按域对象：`loss`、`token_accuracy`、`perplexity`。顶层 `**communication`** 为单客户端单轮上下行字节估计。

**前置 sanity（非主线必排进主表，但建议先做）**

- **E1. GLUE sanity**：验证分类联邦与通信统计。
- **E2. E2E-NLG sanity**：验证生成侧因果 LM 链路。

---

### 【通信-性能实验】是做什么的？要跑满 12 个方法吗？

**做什么**

- **不训练**、**不加载**你训好的 checkpoint；只按与 §11.1 相同的 **Llama-8B + LoRA 形状**（`lora_r=8`、相同 `target_modules`）统计：每个 `agg_type` 在**一轮联邦**里，**每个客户端**大概要传多少字节。
- 统计的是 **可训练 LoRA + 头** 的上下行，**不含**冻结的 8B 基座（已与「全模型交互」修正对齐）。
- 用途：论文里画 **通信–性能 Pareto**（横轴 `down+up` 或总流量，纵轴用主实验的 `domain_macro_token_accuracy` / PPL）；或写「FedPLoRA-oneshot 上行约为 YOCO 的一半」这类对比。

**和主实验 JSON 里 `communication` 的关系**

- **公式相同**（都是 `estimate_round_communication_bytes`）。
- 主实验 JSON 里已有每个方法的 `communication` 块 → **若你只关心已跑完的几种方法，可以直接从主实验 metrics 抄数字，不必再跑本实验**。
- 本脚本的价值：**一次打印多张方法的对比表**，并落盘 `artifacts_35c/comm_profile/sft_comm_35c.json`，方便排版。

**要不要 12 个都测？**

- **不必。** 通信只取决于 **协议设计**（传 A 还是 A+B、FedALT 是否个性化下行等），与某次训练是否完成无关。
- **不必跑满**也可只设子集；与 §11.3.3 主线一致的 **12 个方法** 如下（脚本默认同此列表）。
- 一次脚本（**约 5–15 分钟**，只加载模型一次）：

```bash
cd /path/to/FedPLoRA
set -a && source configs/domain_sft.env && set +a

export AGG_LIST=fedplora-oneshot,fedalt,yoco,fedsa_lora,normal,flora,flexlora,ffa,feddat
bash scripts/RunScripts/run_exp_comm_profile.sh 0
```

产物：`artifacts_35c/comm_profile/sft_comm_35c.json`。集群单轮总流量粗算：`35 × (down_bytes_per_client + up_bytes_per_client)`（见 JSON 内 `note`）。

---

### 【个性化收益分析】是做什么的？和什么相关？为何仍费时？

**回答什么问题（论文叙事）**

主实验 **domain-macro** 只在各域的 `test_domain` 上、按「域 × 客户端」平均，得到 **全局/域级** 好坏，**看不出**联邦个性化是否成立，例如：

- 客户端在自己 **本域** 的 held-out（`test_local`）是否更好？
- 换到 **其它域** 的 `test_domain`（off-domain）是否明显变差？
- **gap** 有多大？（`personalization_gap_token_accuracy = local − off` 等）

因此本实验与 **§四「跨域个性化联邦 SFT」**、**FedPLoRA「B 本地 / A 共享」**、以及 **FedALT / YOCO 等方法的「是否真有个性化」** 直接相关；主表仍用 domain-macro，本实验作 **E5 补充图/表**（ specialization 证据）。

**和主实验的关系**


| 项目  | 主实验（§11.1）                        | 个性化收益                                                      |
| --- | --------------------------------- | ---------------------------------------------------------- |
| 训练  | 已做完                               | **不做**（读同一 checkpoint）                                     |
| 权重  | `../trained_models/<stem>/`       | 同上                                                         |
| 指标  | `domain_macro_`*、`worst_domain_*` | 额外 `client_local_*`、`off_domain_*`、`personalization_gap_*` |
| 协议  | 各方法自己的聚合/评测                       | 相同 checkpoint + `--eval_personalization_metrics`           |


**为何 35 客户端仍要很久（即使不重训）**

实现上在 **原有 domain-macro 评测之后** 再跑三块前向（`fed_train_sft.py` → `_evaluate_personalization_metrics`）：

1. **本域 local**：35 次（每客户端 `test_local`）
2. **跨域 off-domain**：约 **35 × 6 ≈ 210** 次（每客户端在其 **非本域** 的每个域上 eval）
3. **本域 domain_test**：35 次（ sanity，与 macro 对照）

再加上 **domain-macro 本身** 已是 **7 域 × 35 客户端 ≈ 245** 次前向（与主实验 eval 相同，eval-only 会 **再跑一遍**）。  
合计前向量级约为「只跑 domain-macro」的 **约 2–2.5 倍**，且 **客户端数从 7 增到 35 时近似线性变慢** → 单方法 **数小时**、12 方法 **数天** 是正常的。

**时间紧时：不必跑满 12 个方法**；下面给出 **FedPLoRA-Oneshot（v2，`fedplora-oneshot`）** 与 **YOCO** 的 **35 客户端、eval-only、个性化指标** 手敲命令（复用 §11.1 主实验 checkpoint，**不重训**）。

前置（仓库根、与训练时相同 env）：

```bash
cd /path/to/FedPLoRA
set -a && source configs/domain_sft.env && set +a
```

解析 checkpoint 目录（默认 `../trained_models/<stem>/`，需含 `checkpoint_ok.json` 且 `phase=final`）：

```bash
# fedplora-oneshot（v2）
CKPT_ONESHOT=$(python utilities/sft_checkpoint_paths.py \
  --repo_root "$(pwd)" \
  --agg_type fedplora-oneshot \
  --model "${MODEL_PATH}" \
  --benchmark_dir "${BENCHMARK_DIR:-data/domain_benchmark_35c/seed_42}" \
  --rounds 1 --local_epochs 1 --seed 42)
echo "oneshot ckpt: ${CKPT_ONESHOT}"

# yoco
CKPT_YOCO=$(python utilities/sft_checkpoint_paths.py \
  --repo_root "$(pwd)" \
  --agg_type yoco \
  --model "${MODEL_PATH}" \
  --benchmark_dir "${BENCHMARK_DIR:-data/domain_benchmark_35c/seed_42}" \
  --rounds 1 --local_epochs 1 --seed 42)
echo "yoco ckpt: ${CKPT_YOCO}"
```

#### 个性化 eval-only：`fedplora-oneshot`（v2）

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model "${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}" \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedplora-oneshot \
  --rounds 1 \
  --local_epochs 1 \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --client_state_dir "${CLIENT_STATE_DIR:-artifacts/domain_client_states}" \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --eval_only_from_checkpoint "${CKPT_ONESHOT}" \
  --eval_personalization_metrics \
  --yoco_sparse_lambda 1e-4 \
  --oneshot_anchor_lambda 1e-4 \
  --oneshot_conflict_threshold 0.35 \
  --oneshot_conflict_blend 1.0 \
  --seed 42
```

#### 个性化 eval-only：`yoco`

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/fed_train_sft.py \
  --model "${MODEL_PATH:-/data/yaominghao/gb/models/Meta-Llama-3.1-8B}" \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type yoco \
  --rounds 1 \
  --local_epochs 1 \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --client_state_dir "${CLIENT_STATE_DIR:-artifacts/domain_client_states}" \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --eval_only_from_checkpoint "${CKPT_YOCO}" \
  --eval_personalization_metrics \
  --yoco_sparse_lambda 1e-4 \
  --yoco_aggregate_mode conflict \
  --yoco_conflict_method avgm \
  --yoco_sign_lambda 0.01 \
  --seed 42
```

**产物**：`artifacts_35c/sft_metrics/` 下新增  
`*_eval_ckpt_<stem>_r1_e1_seed42.json`（文件名含 `eval_ckpt`）。  
**优先读** `rounds[0].personalization_gap_token_accuracy`、`personalization_gap_perplexity`，以及 `client_local_macro_`* / `off_domain_macro_*`（见 JSON 顶层 `recommended_primary_personalization_metrics`）。

**墙钟（A100、35 客户端、`eval_max_batches=50`）**：每个方法约 **4–10 小时**（不重训，但会重做 domain-macro + 跨域 off-domain 前向）；**两条合计约 8–20 小时**。可与通信表 **并行**（通信用另一张卡或先跑 10 分钟通信表）。

**等价脚本（少打字）**：

```bash
export PERSONALIZATION_AGG_LIST=fedplora-oneshot,yoco
bash scripts/RunScripts/run_exp_personalization.sh 35 0
```

**可选：12 方法一键**（时间充裕时）：`bash scripts/RunScripts/run_exp_post_main_35c.sh 0`（通信 + 全部默认列表）。

---

**代码含义（通信档位，与 `utilities/utils.py` → `estimate_round_communication_bytes` 对照）**


| 通信档位                           | `agg_type`                                  | 上行（约）             | 下行（约）                      |
| ------------------------------ | ------------------------------------------- | ----------------- | -------------------------- |
| **低：仅 A + 头**（+ FedPLoRA 行统计）  | `fedplora`、`fedplora-oneshot`               | A + head（+ stats） | A + head                   |
| **低：仅 A + 头**                  | `fedsa_lora` / `fedsa`                      | A + head          | A + head                   |
| **中：全 LoRA A+B + 头**           | `normal`、`yoco`、`flora`、`flexlora`、`feddat` | A + B + head      | A + B + head（无冻结基座）        |
| **高：Individual / RoTW 双套 A+B** | `fedalt`                                    | A + B + head      | **个性化** A + B + head（RoTW） |
| **低：仅 B + 头（A 冻结）**            | `ffa`                                       | B + head          | B + head                   |


- **FedPLoRA 族**（`fedplora` / `fedplora-oneshot` / `fedsa_lora`）通信约为 **YOCO / FedALT 全 LoRA 路径的一半量级**（只传 A，不传 B）。
- **「B-only / local-only」**：当前仓库 **未**实现为独立 `agg_type`。

可选：`AGG_LIST=normal,yoco,fedsa_lora bash scripts/RunScripts/run_exp_comm_profile.sh 0`。

---

### 【机制消融】FedPLoRA-Oneshot v2：三个核心模块（35c）

主文方法为 **`fedplora-oneshot`（v2，`methods/fedplora_oneshotv2.py`）**。`FedPLoRAOSv3_README.md` 中的残差冲突 / 簇 / RPCA 消融属于 **v3 变体**，与 v2 主线不重复；时间紧时 **只跑下面 3 组 + 主实验 full 对照**即可。

#### 为何只选这 3 个？（去冗余）

| 模块 | 论文主张 | 关掉后（tag） | 与其它两项关系 |
|------|----------|---------------|----------------|
| **M1 服务端冲突门控** | 跨域上行 `A` 行方向冲突大，高冲突行回退 **A0**，否则按共识+行重要度融合 | `wo_conflict` → 样本量加权 **FedAvg**（`--oneshot_ablation_plain_fedavg`） | **服务端核心**；与 M2/M3 正交 |
| **M2 本地 A 稀疏** | 共享 `A` 应稀疏、可解释（与 FedSA/YOCO 叙事一致） | `wo_sparse` → `--yoco_sparse_lambda 0` | 本地训练；不碰服务端公式 |
| **M3 本地 A0 锚定** | one-shot 用初始 **A0** 约束行方向，与本地 **B** 兼容 | `wo_anchor` → `oneshot_anchor/prox_lambda=0` | 本地 one-shot 监督；不碰服务端公式 |

**不纳入本批的原因**：`oneshot_orthogonalize`（次要）、扫 `conflict_threshold`（超参）、多轮 `fedplora` 的 `gp_align/prox/orth`（另一套方法）、v3 cluster/RPCA（另一套方法）、`pcwa_k1`（与 oneshot 服务端无关）。

**full 对照**：直接用 §11.1 已训好的 **`fedplora-oneshot` 主实验**（`artifacts_35c/sft_metrics/...`），**不必**在消融脚本里重跑 `full`，除非 `ABLATION_RUN_FULL=1`。

#### 一键脚本（35 客户端，可指定 GPU）

```bash
cd /path/to/FedPLoRA
set -a && source configs/domain_sft.env && set +a

# 默认串行：wo_sparse → wo_conflict → wo_anchor
bash scripts/RunScripts/run_exp_ablation_oneshot_35c.sh 0
# 或：bash scripts/RunScripts/run_exp_ablation_fedplora.sh 35 0
```

只跑一组：`ABLATION_MODE=wo_conflict bash scripts/RunScripts/run_exp_ablation_fedplora.sh 35 1`

**产物**：`artifacts_35c/sft_metrics_oneshot_ablation/<tag>/`；client 状态：`domain_client_states_oneshot_ablation/<tag>/`；checkpoint：`../trained_models/<stem>_ablation_<tag>/`（与主实验 stem 隔离，避免白训跳过）。对比主表：`rounds[].domain_macro_token_accuracy`、`worst_domain_*`；机制：`rounds[].fedplora_oneshot_conflict`。

**A100 墙钟（35c、`EVAL_MAX_BATCHES=50`）**：每组约 **8–12 h**（与 yoco 训练+测评同量级）；**3 组串行约 1–1.5 天**。多卡可把 `ABLATION_MODE` 拆到不同 GPU 并行。

**多轮 `fedplora` 消融**（`gp_align` / `fedplora_ablation_no_consensus` 等）见 `FedPLoRAOSv2_README.md`；需 `--agg_type fedplora --rounds>1`，不是 oneshot v2 本批三模块。

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
  utilities/data_utils.py utilities/utils.py utilities/sft_checkpoint_paths.py \
  scripts/RunScripts/run_script.py \
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
  --output_dir data/domain_benchmark_35c \
  --num_clients_per_domain 5 \
  --min_samples_per_client 50 \
  --seed 42
```

### 阶段 5：Pilot 主实验

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedplora \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
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
source configs/domain_sft.env
bash scripts/RunScripts/run_domain_sft_baselines.sh
# 分组 + 客户端数（7/14/21/35）见 §11.3.3：推荐四脚本（checkpoint + 全量）
bash scripts/RunScripts/run_domain_sft_batch_group1_oneshot_fedalt.sh 35
bash scripts/RunScripts/run_domain_sft_batch_group2_yoco_fedsa.sh 35
bash scripts/RunScripts/run_domain_sft_batch_group3_normal.sh 35
bash scripts/RunScripts/run_domain_sft_batch_group4_flora_ffa.sh 35
# 旧版分组（可选）
bash scripts/RunScripts/run_domain_sft_baselines1.sh 7
bash scripts/RunScripts/run_domain_sft_baselines2.sh 7
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type normal \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type ffa \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedex \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
```

### 阶段 7：主模型实验

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model Qwen/Qwen3-32B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedplora \
  --rounds 20 \
  --local_epochs 1 \
  --lr 1e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 1 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
  --save_client_state_to_disk
```

### 阶段 8：跨模型验证

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model mistralai/Mistral-Small-24B-Instruct-2501 \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedplora \
  --rounds 20 \
  --local_epochs 1 \
  --lr 1e-4 \
  --lora_r 8 \
  --batch_size 1 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
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

- `**domain_macro_token_accuracy` / `worst_domain_token_accuracy`** 及 `**best_`***：在 **非 `-100` 的 label 位置**（即 **response 段**）上，**下一词预测** 的 micro **token 准确率**（先对每个客户端在该域测试集上算 micro-acc，再在客户端上取平均，再对域取 macro / worst）。这是因果 LM SFT 下与「分类 Accuracy」最接近、可自动批量计算的指标，**建议主表主列用它（越高越好）**。
- `**domain_macro_perplexity` / `worst_domain_perplexity`** 及 `**best_`***：各域 `perplexity` 再 macro / max-worst；**建议主表并列 PPL（越低越好）**，与 LM 论文习惯一致。
- `**domain_macro_loss` / `worst_domain_loss`**：**辅助**（交叉熵，与 NLL 一致）；需要与旧文或 loss 曲线对比时保留。
- `**perplexity`（每域 `domain_metrics` 字典内）**：与上同源；轮级 macro PPL 由域级 PPL 聚合得到。
- `**recommended_primary_metrics`**（JSON 顶层）：字段名列表，标明推荐写进论文主结果的键。
- `**communication`**（JSON 顶层）：单轮上下行字节估计。

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

行（**建议与代码同步后定稿**；脚本里 `pcwa_k1` 对当前冲突门控服务端**无对应机制**）：

- full（默认 `yoco_sparse_lambda` + 默认 `oneshot_`*）
- w/o L1-on-A（`yoco_sparse_lambda=0`）
- （手写）冲突门控：例如扫 `--oneshot_conflict_threshold` / `--oneshot_conflict_blend` 等

多轮 `**fedplora`** 的 B 类消融见 §十四 **B 表**。

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
  --output_dir data/domain_benchmark_35c \
  --num_clients_per_domain 5 \
  --min_samples_per_client 50 \
  --seed 42
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type fedplora \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
  --save_client_state_to_disk
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft.py \
  --model Qwen/Qwen3-14B \
  --benchmark_dir data/domain_benchmark_35c/seed_42 \
  --agg_type normal \
  --rounds 10 \
  --local_epochs 1 \
  --lr 2e-4 \
  --lora_r 8 \
  --batch_size 2 \
  --max_seq_length 2048 \
  --eval_max_batches 50 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
```

跑通这两条之后，你就已经有：

- benchmark 产物
- FedPLoRA 主实验入口
- baseline 对照入口
- 可继续扩展的完整骨架

---

## 附录：FedP-LoRA vs `fedplora-oneshot` vs `yoco` vs `fedalt`

### 本仓库中的实现对应关系


| 项目                            | `fedplora`                  | `fedplora-oneshot`                  | `yoco`                                                                     | `fedalt`                                                      |
| ----------------------------- | --------------------------- | ----------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 联邦轮数                          | 多轮（`--rounds`）              | 强制 **1**                            | 强制 **1**                                                                   | 多轮                                                            |
| 客户端上传                         | A + 头 + 行统计                 | A + 头 + 行统计                         | **A + B + 头**（全量）                                                          | **Individual A + B + 头**                                      |
| 客户端本地保留                       | B（磁盘/内存）                    | B                                   | —（单轮内存收集）                                                                  | Individual A+B；RoTW 另存                                        |
| 服务端下行                         | 全局 A + 头                    | 全局 A + 头                            | 聚合后全局 LoRA                                                                 | **每客户端个性化 RoTW A+B**                                          |
| 服务端聚合                         | `aggregate_models_fedplora` | `aggregate_models_fedplora_oneshot` | `aggregate_models_yoco`（**conflict** / `fedavg` 可选）                        | `aggregate_models_fedalt`（leave-one-out）                      |
| `--save_client_state_to_disk` | **需要**                      | **需要**                              | **不需要**                                                                    | **需要**                                                        |
| 参考实现                          | 本仓库 FedP                    | 本仓库 oneshot                         | FedMLLM `fed_global.py`（见 `methods/references/fedmllm_fed_global_yoco.py`） | [jmbian/FedALT](https://github.com/jmbian/FedALT) `server.py` |


入口分支见 `tasks/fed_train_sft.py`。

### 与 NeurIPS 2025 YOCO 论文的对应

- **已对齐**：单轮联邦；客户端上传 **完整可训练 LoRA（A+B）**；服务端默认 **按样本量 FedAvg**（FedMLLM `global_aggregate` 的 `else` 分支，非 `conflict`/PCWA）。
- **已对齐**：本地 `lora_A` 稀疏（`--yoco_sparse_lambda`）；B 符号约束（`--yoco_sign_lambda`）；服务端 FedMLLM `conflict` → `aggregate_lora_weights`；评测默认用聚合后全局 LoRA。
- **未接线 / 可选**：`--yoco_pcwa_components`；SVD 式 LoRA 初始化等。旧 checkpoint 可用 `--eval_only_from_checkpoint` + `fedavg` 复现旧聚合/评测，不必重训本地。

`**fedplora-oneshot`** 仍是 **FedP 低通信（仅 A）+ 冲突门控** 的研究路径，**不是** YOCO 论文复现。

### FedALT 说明

- **已对齐**：上行 Individual LoRA A+B；服务端 leave-one-out RoTW；通信统计见上表。
- **尚未落地**：论文中 **双 LoRA 分支 + mixer（`lora_route`）** 的前向；当前训练/评测仍用单 PEFT 适配器上的 local A+B。

