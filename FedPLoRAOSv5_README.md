# FedPLoRA-Oneshot v5：从跨域联邦大模型真实问题出发的方案

更新时间：2026/06/12

本文档对应当前代码版本：

- 训练入口：`tasks/fed_train_sft_v4.py`
- v5 脚本：`scripts/RunScripts/run_v5_route_mix_align.sh`
- 通信与方法识别：`utilities/utils.py`
- 参考结果目录：`artifacts_35c/sft_metrics/`、`artifacts_35c/v4_sft_metrics/`

## 1. 顶会顶刊叙事应从实际问题开始

当前故事不能只说“我们只上传 A、B 留本地”。这更像实现细节，而且已经与 FedSA-LoRA 的核心观察接近。更强的故事应从现实部署问题切入：

> 多机构跨域联邦大模型微调的目标不是训练一个所有人都相同的全局 LoRA，而是在一次或极少通信下，让每个机构获得可用的跨域共识能力，同时不牺牲本域私有能力。

典型场景是医院、律所、金融机构、学校、代码平台和通用助手服务方共同微调同一个开源 LLM。它们的数据不能集中，通信窗口有限，且每个机构的目标域不同。对于这种 setting，传统 FL-LLM LoRA 方法有三个实际约束：

1. **隐私约束**：不能上传原始数据；如果上传完整 LoRA 的 A+B，B 中更强的客户端特异信息也会被暴露。
2. **通信约束**：大模型 LoRA 虽然比全参小，但 7 个 target modules、35 clients、8B 模型下，A+B 的通信仍然约为 A-only 的 2 倍。
3. **个性化约束**：跨域数据不是简单 non-IID，而是 domain shift。一个统一全局 adapter 往往提升平均性能，但会牺牲某些高风险域的 worst-case 表现。

因此，本文要解决的问题应定义为：

> 在 one-shot 或极少通信的跨域联邦大模型微调中，如何同时满足低通信、隐私隔离、跨域共享和本地个性化，并避免全局知识对某些域产生负迁移？

## 2. 现有联邦大模型 LoRA 的关键局限

### 2.1 直接聚合 A+B：通信高，且 LoRA 双线性结构导致聚合噪声

LoRA 的有效更新是 `ΔW_i = B_i A_i`，不是独立的 `A_i` 或 `B_i`。如果服务端分别平均 A 和 B，则得到：

```text
ΔW_avg = (Σ p_i B_i)(Σ p_i A_i)
```

其中包含 `B_i A_j (i≠j)` 的交叉项。这类交叉项不是任何客户端真实学习到的更新。FLoRA 明确指出传统 LoRA averaging 会产生数学上的 aggregation noise，并用 stacking 避免这种噪声。

但 FLoRA / normal / flexlora / YOCO 类方法在当前 35c 实验中表现更强，一个重要原因是它们保留或近似保留了更完整的 LoRA 更新容量。代价是通信约为 A-only 方法的 2 倍。

### 2.2 冻结 A 或只训练 B：稳定但表达能力不足

FFA-LoRA / FedSVD 一类思路说明，固定或正交化 A 可以减少 A/B 坐标不稳定，尤其在隐私噪声或 DP 场景下更稳定。但完全固定 A 会限制适应新域的能力，跨域大模型 SFT 中容易出现 underfitting。

这说明：A 不能简单固定，也不能无条件全局替换；A 应当是可学习的共享子空间，但客户端需要保留使用或拒绝该共享子空间的能力。

### 2.3 只聚合 A、B 本地保留：通信和隐私合理，但存在 A/B 坐标错配

FedSA-LoRA 的核心观察是 A 更偏 general knowledge，B 更偏 client-specific knowledge。因此只上传 A、B 留本地是合理起点。但当前 FedPLoRA-Oneshot 的实验说明，A-only 本身还不够：

| 方法 | 35c macro acc | worst acc | hard avg loss | per-client down/up |
|---|---:|---:|---:|---:|
| FedPLoRA-Oneshot | 0.69595 | 0.51982 | 1.7774 | 38MB / 38MB |
| v4_mix_per_domain | 0.69616 | 0.52014 | 1.7717 | 38MB / 38MB |
| normal | 0.69819 | 0.52318 | 1.7481 | 80MB / 80MB |
| YOCO | 0.70051 | 0.52181 | 1.7460 | 80MB / 80MB |
| FlexLoRA | 0.70121 | 0.52961 | 1.6953 | 80MB / 80MB |
| FLoRA | 0.70165 | 0.52718 | 1.6968 | 80MB / 80MB |

诊断结论：

- A-only 的通信优势明确，约为 A+B 的一半。
- 但 macro acc 距离 FLoRA / FlexLoRA 仍有约 0.005。
- v3/v4 主要改 server aggregation，但提升很小，说明瓶颈不只在服务端聚合，而在客户端如何使用下发的 A。
- 当前的失败模式是：`A_down` 携带跨域共识，但本地 `B_i` 是围绕 `A_i^local` 学到的，强行替换为 `A_down` 可能造成坐标错配。

### 2.4 One-shot 方法更容易产生不可修正的负迁移

多轮 FL 中，如果某轮聚合伤害了客户端，后续本地训练还能修正。但 one-shot 只有一次上传和一次下发，客户端没有下一轮通信纠错机会。因此 one-shot 场景下不能只问“服务端如何聚合”，还必须问：

> 客户端收到唯一一次全局 A 后，如何判断它是否真的适合自己？

这是本文与已有 one-shot FedLoRA 工作拉开差异的关键。

## 3. FedPLoRA-Oneshot v5 的核心思想

v5 不再把“只上传 A”当作最终贡献，而把它作为解决真实问题的机制之一。

### 3.1 方法目标

FedPLoRA-Oneshot v5 的目标是：

> 通过一次 A-only 上传获得跨域共享子空间，在客户端保留 B 和本地 A 作为私有个性化知识，并通过本地验证路由与 B-only 对齐，让每个客户端安全地使用全局知识。

### 3.2 三阶段流程

**阶段 1：本地训练与 A-only 上传**

每个客户端训练 LoRA：

```text
ΔW_i = B_i A_i
```

上传：

```text
upload_i = {A_i, task_head, row_importance(B_i, A_i)}
```

保留在本地：

```text
private_i = {B_i, A_i^local}
```

其中 `A_i^local` 不再只是训练中间产物，而是客户端的私有回退路径。

**阶段 2：服务端 conflict-gated A aggregation**

服务端沿用 FedPLoRA-Oneshot 的 A-only 聚合：

```text
A_down = Agg_A({A_i}, A_0, row_importance_i)
```

聚合时参考初始 `A_0`，用 row-level consensus / conflict gate 控制高冲突行，避免在跨域冲突强的维度上强行平均。

**阶段 3：客户端验证路由 + B-only 对齐**

客户端收到 `A_down` 后，不直接替换本地 A，而是在本地验证集上选择：

```text
A_eff(η) = η A_down + (1 - η) A_local
η ∈ {0.0, 0.1, ..., 1.0}
```

路由含义：

- `η=1`：信任全局共享 A，适合低冲突、共识强客户端。
- `η=0`：回退本地 A，适合全局 A 对该客户端负迁移的情况。
- `0<η<1`：混合全局与本地 A，适合部分共享、部分冲突的域。

路由后可选执行少量本地 B-only 对齐：

```text
min_B L_i(B, A_eff) + λ ||B - B_before||^2
```

这一步只在客户端本地执行，不上传 B，不增加通信。它解决的是 `B_i` 原本围绕 `A_i^local` 学到，而现在 `A_eff` 可能含有 `A_down` 成分，因此需要少量坐标对齐。

## 4. 为什么 v5 更能解释并冲击当前性能瓶颈

当前结果中，FLoRA / FlexLoRA / YOCO / normal 表现更好，不应简单解释为“它们方法更强”，而应解释为：

1. 它们上传或聚合了更完整的 A+B 信息，因此 `B` 与 `A` 的配对关系更完整。
2. FedPLoRA-Oneshot 只上传 A，B 留本地，虽然通信和隐私更好，但下发 A 后没有判断该 A 是否适合本地 B。
3. v4 只在服务端聚合端做文章，不能从根本上解决客户端 `A_down` 与 `B_i` 的使用错配。

v5 的补救机制正好针对这个瓶颈：

- **A-only 上传**保留通信与隐私优势。
- **A_local 私有回退**避免 one-shot 全局 A 负迁移。
- **验证路由**把“是否使用全局知识”的权力交给客户端，而不是服务端统一决定。
- **B-only 对齐**用本地计算弥补 A/B 坐标错配，不增加通信。

这使得本文故事从“我们提出一种聚合方式”升级为：

> 我们提出一种 one-shot cross-domain personalized FL-LLM adaptation framework，它把联邦 LoRA 的问题从 server-side aggregation 扩展为 global knowledge acquisition + local safe utilization。

## 5. Motivation 图设计

建议画 5 个连续子图，形成从问题到方法的闭环。

### Figure 1(a)：真实跨域联邦大模型部署场景

画面元素：

- 中央是 frozen LLM。
- 周围 7 类机构/client：general、math、code、medical、legal、finance、education。
- 每个 client 有私有数据锁标识。
- 标注约束：`privacy`、`one-shot communication`、`domain personalization`。

要传达的信息：

> 目标不是得到一个统一 adapter，而是每个域都能在不共享数据的情况下受益于跨域协作。

### Figure 1(b)：现有 A+B 聚合的双线性噪声与高通信

画面元素：

- 左侧多个客户端上传 `A_i+B_i`。
- 中间服务端分别平均 `A` 和 `B`。
- 右侧展开公式：`(ΣB_i)(ΣA_i)` 产生 `B_i A_j` 交叉噪声。
- 通信条形图：A+B 为 `2x`。

要传达的信息：

> 完整 LoRA 聚合有容量优势，但通信高，且直接 averaging 不尊重 LoRA 的乘积结构。

### Figure 1(c)：A/B 角色不对称与 A-only 上传的合理性

画面元素：

- A 矩阵用蓝色表示 general/shared basis。
- B 矩阵用橙色表示 private/domain-specific mapping。
- 服务端只接收 A，B 留在 client。
- 标注：`A: shared transferable subspace`，`B: private response mapping`。

要传达的信息：

> A-only 不是为了省参数而省参数，而是利用 LoRA 内部角色分工，将可共享知识和私有知识解耦。

### Figure 1(d)：FedPLoRA-Oneshot 的剩余 gap：全局 A 可能伤害本地 B

画面元素：

- 客户端本地有 `B_i` 与 `A_i^local` 对齐。
- 服务端下发 `A_down`。
- `B_i A_down` 的箭头有红色 warning：coordinate mismatch / negative transfer。
- 展示当前结果小表：FedPLoRA-Oneshot 低通信但 macro 低于 FLoRA/FlexLoRA。

要传达的信息：

> A-only 解决了通信和隐私，但 one-shot 场景下缺少客户端纠错机制。

### Figure 1(e)：v5 的安全使用机制

画面元素：

- 客户端同时保留 `A_local` 和接收 `A_down`。
- 本地验证 router 输出 `η`。
- 得到 `A_eff = η A_down + (1-η)A_local`。
- 之后有 `B-only alignment`。
- 最终输出 personalized adapter：`B_i A_eff`。

要传达的信息：

> 全局知识不是被强制使用，而是由客户端本地验证决定是否、多少使用；这就是跨域 one-shot 个性化的核心。

## 6. 当前代码实现

### 6.1 新增方法入口

当前已新增：

```bash
--agg_type v5_route_mix_align
--agg_type v5_rpca_route_mix_align
```

`v5_route_mix_align` 复用 FedPLoRA-Oneshot 的服务端 A-only 聚合，但在客户端评估/部署前执行：

1. 保存 `A_i^local`。
2. 下发 `A_down`。
3. 本地验证集搜索 `η`。
4. 安装 `A_eff`。
5. 可选执行 B-only local alignment。

`v5_rpca_route_mix_align` 是更激进的性能增强版本：

1. 服务端先借鉴 FedRPCA，把 `A_i - A_0` 残差拆成 common 与 sparse 成分。
2. common 成分用于提取跨域共享方向，sparse 成分用于保留客户端/簇特异方向。
3. 客户端仍执行同样的本地 route 与 B-only alignment。
4. 如果 RPCA 下发的 `A_down` 伤害本地表现，route 可以退回 `A_local`，因此不会把服务端增强强制施加给所有客户端。

简言之：`v5_route_mix_align` 是稳健主方法，`v5_rpca_route_mix_align` 是“更强 server A + 本地安全回退”的冲性能版本。

通信口径仍按 A-only 计算。`v5_rpca_route_mix_align` 可能为不同 client 生成 personalized A_down，但下发内容仍只包含 A/task head，不下发 B；B 始终保留在客户端本地。

### 6.2 新增参数

```bash
--v5_route_val_scope local|domain|global
--v5_route_search_grid "0.0,0.1,...,1.0"
--v5_route_search_max_batches 4
--v5_route_tie_margin 0.0
--v5_route_tie_breaker best|global|local|mixed
--v5_route_post_align_steps 5
--v5_route_post_align_lr 0.0001
--v5_route_post_align_prox_lambda 0.0
--v3_rpca_rank 1
--v3_sparse_quantile 0.80
```

默认推荐：

- 主实验：`--v5_route_val_scope local`，这是最符合隐私部署的设置。
- 上限实验：`--v5_route_val_scope domain`，代表存在公开 domain anchor 数据。
- 对齐步数：先试 `5`，再消融 `0/2/5/10`。

### 6.3 输出指标

metrics JSON 的 `rounds[-1].v5_route_stats` 会记录：

```json
{
  "scope": "local",
  "num_routes": 245,
  "num_cached_searches": 35,
  "route_counts": {"global": 10, "mixed": 20, "local": 5},
  "mean_eta": 0.57,
  "min_eta": 0.0,
  "max_eta": 1.0,
  "post_align": {
    "enabled": true,
    "num_align_states": 35,
    "num_clients_aligned": 35,
    "mean_final_loss": 1.23
  }
}
```

这些统计非常重要：

- 如果 `η` 大多接近 1，说明全局 A 可靠。
- 如果 `η` 大多接近 0，说明服务端 A 聚合仍不够好，A-only 共享带来负迁移。
- 如果 mixed 较多且性能提升，说明跨域共享与本地个性化确实需要连续融合，而不是二选一。

## 7. 运行命令

### 7.0 先跑 LW7c 轻量筛选

建议先在 SmolLM2-135M + LW7c 上快速筛选 v5 四个配置，确认 route/align/rpca 至少不退化，再上 Llama-3.1-8B 35c。

```bash
cd /Users/hawaiii/codex/FedPLoRa/FedPLoRA-main
bash scripts/RunScripts/LWv4/build_lw7c_benchmark.sh
bash scripts/RunScripts/LWv4/download_lw_model_modelscope.sh
bash scripts/RunScripts/LWv4/run_lwv5_route_mix_align.sh 0
```

LW v5 结果目录：

```bash
artifacts_LW7c/v4_sft_metrics_v5/<tag>/
```

LW route 诊断：

```bash
python scripts/Analysis/summarize_v5_routes.py \
  --inputs "artifacts_LW7c/v4_sft_metrics_v5/*/*.json" \
  --out artifacts_LW7c/v5_route_summary.csv
```

### 7.1 35-client 主实验：v5 四配置

仓库根目录：

```bash
cd /Users/hawaiii/codex/FedPLoRa/FedPLoRA-main
bash scripts/RunScripts/run_v5_route_mix_align.sh 0,1
```

该脚本会连续运行：

| tag | agg_type | scope | B 对齐 | 定位 |
|---|---|---|---:|---|
| local_route | v5_route_mix_align | local | 0 | 只验证本地 route |
| local_route_align | v5_route_mix_align | local | 5 | 主推 v5，可部署隐私设置 |
| domain_anchor_align | v5_route_mix_align | domain | 5 | public anchor 上限，不作为最保守主结论 |
| rpca_local_route_align | v5_rpca_route_mix_align | local | 5 | 冲性能版本：common+sparse server A + 本地安全回退 |

### 7.2 只跑最推荐配置

```bash
cd /Users/hawaiii/codex/FedPLoRa/FedPLoRA-main
set -a
source configs/v4_baseline.env
set +a

CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5_route_mix_align \
  --rounds 1 --local_epochs 1 --lr 0.0002 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --batch_size 2 --max_seq_length 2048 \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
  --gradient_checkpointing \
  --client_state_dir artifacts_35c/v5_client_states_route_align \
  --save_client_state_to_disk \
  --metrics_output_dir artifacts_35c/v4_sft_metrics \
  --eval_max_batches 50 \
  --eval_seeds 42 \
  --v4_mix_save_dir artifacts_35c/v5_route_a_local_main \
  --v5_route_val_scope local \
  --v5_route_search_grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
  --v5_route_search_max_batches 4 \
  --v5_route_post_align_steps 5 \
  --v5_route_post_align_lr 0.0001 \
  --oneshot_anchor_lambda 0.0001
```

### 7.3 更强但要谨慎表述的 domain-anchor 上限

```bash
cd /Users/hawaiii/codex/FedPLoRa/FedPLoRA-main
set -a
source configs/v4_baseline.env
set +a

CUDA_VISIBLE_DEVICES=0,1 python tasks/fed_train_sft_v4.py \
  --model "$MODEL_PATH" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --agg_type v5_route_mix_align \
  --rounds 1 --local_epochs 1 --lr 0.0002 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --batch_size 2 --max_seq_length 2048 \
  --torch_dtype bfloat16 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
  --gradient_checkpointing \
  --client_state_dir artifacts_35c/v5_client_states_domain_anchor \
  --save_client_state_to_disk \
  --metrics_output_dir artifacts_35c/v4_sft_metrics \
  --eval_max_batches 50 \
  --eval_seeds 42 \
  --v4_mix_save_dir artifacts_35c/v5_route_a_local_domain_anchor \
  --v5_route_val_scope domain \
  --v5_route_search_grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
  --v5_route_search_max_batches 4 \
  --v5_route_post_align_steps 5 \
  --v5_route_post_align_lr 0.0001 \
  --oneshot_anchor_lambda 0.0001
```

注意：`domain` scope 需要在论文中表述为 public anchor / validation-assisted setting，不能和纯本地隐私设置混为一谈。

### 7.4 汇总结果

```bash
cd /Users/hawaiii/codex/FedPLoRa/FedPLoRA-main
python scripts/Analysis/summarize_v4.py \
  --inputs "artifacts_35c/sft_metrics/*.json" "artifacts_35c/v4_sft_metrics/*.json" "artifacts_35c/v4_sft_metrics/*/*.json" \
  --out artifacts_35c/v5_summary.csv
```

查看：

```bash
cat artifacts_35c/v5_summary.csv
```

### 7.5 汇总 v5 路由行为

```bash
cd /Users/hawaiii/codex/FedPLoRa/FedPLoRA-main
python scripts/Analysis/summarize_v5_routes.py \
  --inputs "artifacts_35c/v4_sft_metrics/*/*.json" \
  --out artifacts_35c/v5_route_summary.csv
```

查看：

```bash
cat artifacts_35c/v5_route_summary.csv
```

重点看：

- `global_frac`：全局 A 被直接采用的比例。
- `local_frac`：回退本地 A 的比例，过高说明全局 A 仍有负迁移。
- `mixed_frac`：连续融合比例，若该比例高且性能提升，可支撑“跨域共享与个性化需要软融合”的故事。
- `mean_eta`：越高越偏全局共享，越低越偏本地个性化。
- `num_cached_searches`：local scope 应接近 client 数；domain scope 应接近 client × domain 数。

## 8. 推荐实验与消融

### 8.1 主表

主表应按通信预算分组，而不是把所有方法放一起直接比。

| 组别 | 方法 | 通信 | 作用 |
|---|---|---|---|
| A-only / B-local | FedSA-LoRA、FedPLoRA-Oneshot、v4_mix、v5_route_mix_align | 低 | 验证同通信预算下的性能提升 |
| A+B full LoRA | normal、YOCO、FLoRA、FlexLoRA、FedDAT | 高 | 高通信强基线 |
| 个性化 FL | FedALT、FDLoRA-style baseline | 中/高 | 证明个性化必要 |

论文表述应避免直接宣称 v5 必须全面超过 FLoRA/FlexLoRA。更稳的主结论是：

> 在约一半通信预算下，v5 显著缩小甚至超过部分 full-LoRA one-shot baselines，并在 worst-domain 与 hard-domain loss 上更稳定。

如果 v5 跑出来超过 FLoRA/FlexLoRA，再作为 strongest result；如果只接近，也可以通过通信-性能 Pareto frontier 讲清楚贡献。

### 8.2 v5 消融

必须做：

| 消融 | 命令变化 | 验证问题 |
|---|---|---|
| 无路由 | `fedplora_oneshot` | A-only 聚合原始性能 |
| 固定混合 | `v4_mix_fixed05` | 固定 η 是否足够 |
| 只路由 | `v5_route_post_align_steps=0` | 本地验证路由是否有效 |
| 路由 + B 对齐 | `v5_route_post_align_steps=5` | B-only 对齐是否解决 A/B 错配 |
| local scope | `v5_route_val_scope=local` | 可部署隐私设置 |
| domain scope | `v5_route_val_scope=domain` | public anchor 上限 |
| tie breaker | `best/global/local/mixed` | 路由偏好是否影响稳定性 |
| 无 RPCA server | `v5_route_mix_align` | 只看客户端 route + B 对齐 |
| RPCA server | `v5_rpca_route_mix_align` | common+sparse 是否能提高 A_down 质量 |

### 8.3 诊断图

建议补 3 张诊断图：

1. **η 分布图**：横轴 η，纵轴 client/domain 频次。证明不同域确实需要不同 global-local mixing。
2. **domain-wise gain 图**：v5 相对 FedPLoRA-Oneshot 在 7 个域上的 acc/loss 变化。证明不是只提升 easy domain。
3. **通信-性能 Pareto 图**：横轴 GB，纵轴 macro acc 或 hard avg loss。突出 v5 的低通信优势。

## 9. 预期性能优化路径

如果 v5 仍无法超过 FLoRA/FlexLoRA，优先按以下顺序优化：

1. **扩大 `v5_route_search_max_batches` 到 8 或 16**：路由选择更稳定，成本只发生在本地验证。
2. **调 `v5_route_post_align_steps` 为 2/5/10**：过大可能过拟合，优先看 hard avg loss。
3. **把 tie breaker 设为 mixed**：如果 best 路由过于极端，`mixed` 可减少 η=0/1 抖动。
4. **增大 `oneshot_anchor_lambda` 到 3e-4**：增强 A/B 行坐标稳定性，但过大可能限制 A 学习。
5. **加入 row dropout 或 sparse A 正则**：借鉴 FedLoDrop，减少低置信共享方向的过拟合。
6. **加入 common+sparse server aggregation**：已落到 `v5_rpca_route_mix_align`，借鉴 FedRPCA，把 A residual 拆成 common + sparse，再交给客户端 route 安全使用。

## 10. 论文定位与贡献写法

推荐贡献表述：

1. **Problem**：提出并系统研究 cross-domain one-shot personalized federated LLM adaptation，强调这不是传统 IID/non-IID 分类任务，而是跨领域能力共享与本地能力保真的冲突。
2. **Observation**：通过 35-client Llama-3.1-8B 实验发现，A-only FedPLoRA 虽有通信优势，但输给 full-LoRA baselines 的主要原因是 one-shot 下发 A 与本地 B 的坐标错配及负迁移。
3. **Method**：提出 FedPLoRA-Oneshot v5，通过 A-only global basis acquisition、private A/B retention、validation-routed A fusion 和 local B-only alignment，在不增加通信的前提下实现安全个性化。
4. **Evidence**：在 general/math/code/medical/legal/finance/education 七域上评估 macro、worst、hard-domain loss 和 communication Pareto，证明 v5 在低通信预算下更优。

不要这样写：

> 我们是第一个只上传 A 的方法。

这会被 FedSA-LoRA 直接挑战。

更稳的写法：

> Different from prior Share-A LoRA methods that focus on what to aggregate, we study how a client should safely use the one-shot global LoRA basis under cross-domain conflicts. FedPLoRA-Oneshot v5 turns one-shot federated LoRA from a server aggregation problem into a global-local personalized adaptation problem.

## 11. 参考工作与可引用依据

- FedSA-LoRA / Selective Aggregation：A 更偏共享知识，B 更偏客户端特异知识；只上传 A 有理论与实证基础。https://arxiv.org/abs/2410.01463
- FLoRA：直接平均 A/B 会产生 LoRA 双线性聚合噪声；stacking 能保留完整更新但通信更高。https://arxiv.org/abs/2409.05976
- FDLoRA：dual LoRA + adaptive fusion 说明全局与个性化路径融合是合理的 PFL 方向。https://arxiv.org/abs/2406.07925
- FedRPCA：common + sparse 分解说明跨客户端更新中同时存在共识和客户端特异成分。https://arxiv.org/abs/2506.01194
- FedSVD：正交/重参数化 A 可以改善 A/B 坐标稳定性。https://arxiv.org/abs/2505.12805
- FedLoDrop：dropout/sparsity 可缓解过拟合并降低通信，可作为 v5 后续增强。https://arxiv.org/abs/2510.12078

## 12. 当前风险边界

1. 当前 v5 已落代码，但尚未在 35c 重新跑出结果，不能在论文中提前宣称 SOTA。
2. `domain` scope 路由可能使用 domain-level validation，必须标注为 public anchor setting 或 upper bound。
3. `local` scope 是最干净的隐私部署设置，但如果每个 client 的 val 太少，η 搜索会不稳定，需要报告 route statistics。
4. 如果 v5 的性能只接近 full-LoRA baselines，主卖点应转为 communication-performance Pareto，而不是绝对 macro 第一。
5. 所有最终表格需要至少 3 seeds 或 bootstrap CI；当前单 seed 差异在 0.002-0.006 范围内，不能过度解读。
