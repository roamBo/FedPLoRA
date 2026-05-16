# FedPLoRA-Oneshot v3：面向跨域冲突的层次化鲁棒一次通信个性化 LoRA

这个文档记录 `20260515` 提交的日志进行新一轮版本的 **FedPLoRA-Oneshot v3** 方案设计与实现说明。该版本已落到代码，目标是在 v2 的基础上进一步提升 `Macro`、`Worst` 和 `Hard avg`，同时保持 **one-shot 通信**、**A-only 上传**、**B 私有保留** 的核心设定。当前实现位于 `methods/fedplora_oneshot.py` 与 `tasks/fed_train_sft.py`。

已实现的 `agg_type`：

- `fedplora_v3_lite` / `v3_lite`
- `fedplora_v3_cluster` / `v3_cluster`
- `fedplora_v3_rpca` / `v3_rpca`

注意：历史入口中的 `fedplora_oneshot` 在 `fed_train_glue.py` 和 `fed_train_e2e.py` 中仍保留原 YOCO-style 兼容语义；SFT 主实验建议直接使用上述 `fedplora_v3_*` 名称，避免混淆。

---

## 1. 当前 v2 的问题诊断

从微信发我的最新结果来看：

- `domain_macro_loss = 1.2397`
- `worst_domain_loss = 2.3419`
- `hard_domain_avg_loss = 1.7774`
- `mean_conflict = 0.0430`
- `max_conflict = 0.1154`
- `high_conflict_row_frac = 0.0000`
- `mean_init_gate = 0.0000`

可以得到三个结论：

1. v2 相比 YOCO-style baseline 在 `Worst` 和 `Hard avg` 上更好，说明 `A0` 参考聚合对困难域有帮助。
2. v2 的 `Macro` 与 YOCO-style 基本持平，说明全局平均性能还没有明显拉开。
3. 当前 `--oneshot_conflict_threshold=0.35` 下高冲突门控完全没有触发，v2 实际更像 **A0-referenced weighted A aggregation**，不是充分激活的 conflict-gated 方法。

因此，v3 的核心不是简单继续调低阈值，而是将冲突建模从原始 `A_i` 空间迁移到 **残差空间**，并将“高冲突回退到 `A0`”升级为“高冲突进入域簇个性化路径”。

---

## 2. v3 核心定位

FedPLoRA-Oneshot v3 的建议定位：

> FedPLoRA-Oneshot v3 is a hierarchical robust one-shot personalized federated LoRA framework. It uploads only LoRA A and lightweight statistics while keeping B private, decomposes client updates into common and conflict-aware residual components, and returns personalized domain-cluster A adapters in a single downlink.

中文叙事：

> v3 面向跨域强异质场景，只通信一次，只上传 `A_i` 与轻量统计，不上传私有 `B_i`。服务端在 `ΔA = A_i - A0` 残差空间中识别公共知识和跨域冲突，将低冲突残差聚合为全局共享 `A_common`，将高冲突残差按领域簇聚合为个性化 `A_cluster`，最终每个客户端一次性接收与其领域匹配的 `A_down`，并与本地私有 `B_i` 融合。

---

## 3. 与 v2 的关键区别

| 维度 | v2 | v3 |
|---|---|---|
| 冲突空间 | 直接比较 `A_i` 行方向 | 比较 `ΔA_i = A_i - A0` 的残差方向 |
| 阈值 | 固定 `--oneshot_conflict_threshold` | 分层 / 分层内 quantile 自适应阈值 |
| 高冲突处理 | 回退到 `A0` | 进入域簇路径或客户端相似簇路径 |
| 聚合结果 | 单个 `A_global` | `A_common + A_cluster`，一次下发个性化 `A_down` |
| 公共/私有分离 | 依赖 row conflict gate | 显式 common-residual / sparse-residual 分解 |
| 个性化强度 | 主要依赖本地 `B_i` | 本地 `B_i` + 域簇 `A_cluster` 双重个性化 |
| 预期收益 | 提升 hard / worst | 同时提升 macro、hard、worst |

---

## 4. 方法总览

每个客户端本地训练一次：

\[
\Delta W_i = B_i A_i
\]

其中 `B_i` 始终保留本地，服务端只收到：

- `A_i`
- task head
- row-importance
- 客户端样本量
- 可选的 domain id 或匿名聚类统计

服务端保存初始化共享基：

\[
A_0
\]

并计算残差：

\[
R_i = A_i - A_0
\]

v3 服务端不再直接对 `A_i` 做单一路径聚合，而是执行：

1. 残差方向对齐
2. 残差冲突估计
3. common / sparse 分解
4. 域簇或相似簇聚合
5. 一次性个性化下发

最终客户端 `i` 使用：

\[
W_i = W_0 + B_i A_i^{down}
\]

其中：

\[
A_i^{down} = A_0 + R_{common} + \lambda_i R_{cluster(i)}
\]

`R_common` 表示全局公共残差，`R_cluster(i)` 表示客户端所属领域簇的冲突残差，`\lambda_i` 控制个性化强度。

---

## 5. 模块一：残差空间冲突门控

### 5.1 动机

v2 直接比较 `A_i` 与 `A0` 的方向。当前实验中 `A_i` 仍高度接近 `A0`，导致：

\[
mean\_conflict = 0.043
\]

高冲突行比例为 0。也就是说，真正的领域差异被 `A0` 的主方向淹没了。

v3 改为比较：

\[
R_i = A_i - A_0
\]

即比较客户端实际学习到的残差方向。这样更容易暴露 math / code / medical / legal / finance 等领域对共享子空间的不同拉动。

### 5.2 残差行方向

对第 `l` 层 LoRA `A` 的第 `r` 行：

\[
R_{i,l,r} = A_{i,l,r} - A_{0,l,r}
\]

归一化：

\[
\hat{R}_{i,l,r} = \frac{R_{i,l,r}}{\|R_{i,l,r}\|_2 + \epsilon}
\]

如果残差范数过小，则该行视为“未学习有效残差”，直接降低聚合权重。

### 5.3 残差冲突分数

客户端权重：

\[
w_{i,l,r} \propto n_i^\alpha \cdot importance_{i,l,r}^{\beta} \cdot norm(R_{i,l,r})^{\gamma}
\]

残差共识：

\[
c_{l,r} = \left\| \sum_i w_{i,l,r} \hat{R}_{i,l,r} \right\|_2
\]

残差冲突：

\[
conflict_{l,r} = 1 - c_{l,r}
\]

相比 v2，v3 的冲突分数不再由 `A_i` 本身主导，而是由实际更新残差主导。

---

## 6. 模块二：自适应阈值而非固定阈值

v2 使用固定阈值：

\[
\tau = 0.35
\]

在当前实验中没有任何 row 被判定为高冲突。v3 改为分层自适应阈值：

\[
\tau_l = Quantile(\{conflict_{l,r}\}_{r=1}^{rank}, q)
\]

推荐默认：

- `q = 0.80`：每层 top 20% 冲突 row 进入个性化路径
- `q = 0.90`：更保守，只处理 top 10% 冲突 row
- `q = 0.70`：更激进，适合 7c 消融

连续门控：

\[
g_{l,r} = \sigma\left(\frac{conflict_{l,r} - \tau_l}{T}\right)
\]

其中 `T` 是温度系数。这样即使没有超过固定阈值，也可以让相对更冲突的 row 得到更强个性化处理。

---

## 7. 模块三：Common + Sparse 残差分解

该模块借鉴 FedRPCA 的思想：客户端更新中同时存在跨客户端公共知识和客户端 / 领域特异知识。直接平均会压制特异知识，直接放大又会伤害全局泛化。

对每一层或每组 LoRA `A` 残差，将客户端残差展平成矩阵：

\[
D_l =
\begin{bmatrix}
vec(R_{1,l}) \\
vec(R_{2,l}) \\
\cdots \\
vec(R_{N,l})
\end{bmatrix}
\]

做鲁棒分解：

\[
D_l = L_l + S_l
\]

其中：

- `L_l`：跨客户端公共低秩成分
- `S_l`：领域 / 客户端稀疏偏移

聚合：

\[
R_{common,l} = mean_i(L_{i,l})
\]

高冲突残差不再回退到 `A0`，而是进入 sparse / cluster 路径：

\[
R_{cluster,l}^{(k)} = mean_{i \in C_k}(S_{i,l})
\]

最终：

\[
A_{down}^{(k)} = A_0 + R_{common} + \lambda_k \cdot R_{cluster}^{(k)}
\]

这样可以保留 `A0` 作为稳定坐标，同时不把领域特异残差硬抹掉。

---

## 8. 模块四：领域簇层次化聚合

当前任务天然包含 7 个领域：

- `general`
- `math`
- `code`
- `medical`
- `legal`
- `finance`
- `education`

v3 建议支持两种簇定义。

### 8.1 先验领域簇

用于主实验，便于讲故事：

| 簇 | 领域 | 解释 |
|---|---|---|
| `general_education` | general, education | 通用指令与教学反馈 |
| `capability` | math, code | 能力型专业域 |
| `risk` | medical, legal, finance | 高风险垂直域 |

或更细粒度：

| 簇 | 领域 |
|---|---|
| `general` | general |
| `education` | education |
| `math_code` | math, code |
| `risk` | medical, legal, finance |

### 8.2 数据驱动簇

不直接使用领域名，而是根据客户端上传的残差统计构建相似度：

\[
sim(i,j) = cosine(vec(R_i), vec(R_j))
\]

或按 row-importance 加权：

\[
sim(i,j) = cosine(importance_i \odot vec(R_i), importance_j \odot vec(R_j))
\]

再做谱聚类或层次聚类，得到 `C_1, ..., C_K`。

数据驱动簇更客观，但论文叙事上没有先验领域簇直观。建议主结果用领域簇，消融中补数据驱动簇。

---

## 9. 模块五：一次性个性化下发

v2 所有客户端收到同一个 `A_global`。v3 建议服务端一次聚合后给不同簇下发不同 `A_down`：

\[
A_i^{down} = A_{common} + \lambda_i A_{cluster(i)}
\]

这仍然是 one-shot，因为通信流程没有增加：

1. client -> server：一次上传
2. server -> client：一次下发

区别只是 server 下发给不同客户端的 `A` 不完全相同。

### 9.1 个性化强度

默认：

\[
\lambda_i = \lambda_{cluster}
\]

可用簇内验证 loss 或残差冲突强度估计：

\[
\lambda_{cluster} = clip(mean(conflict_{C_k}), \lambda_{min}, \lambda_{max})
\]

建议默认：

- `lambda_min = 0.2`
- `lambda_max = 1.0`
- `risk` 簇可更大
- `general_education` 簇可更小

### 9.2 行级混合

更细粒度的版本：

\[
A_{i,l,r}^{down}
= A_{0,l,r}
+ R_{common,l,r}
+ g_{l,r}^{(cluster)}
\cdot \lambda_i
\cdot R_{cluster(i),l,r}
\]

低冲突 row 主要使用 common，高冲突 row 使用 cluster residual。

---

## 10. 模块六：本地训练正则改进

v2 已有：

- `--yoco_sparse_lambda`
- `--oneshot_anchor_lambda`
- `--oneshot_prox_lambda`

v3 建议增加以下训练侧约束。

### 10.1 残差幅度约束

避免客户端 `A_i` 在 one-shot 中过度漂移：

\[
L_{res} = \sum_l \|A_{i,l} - A_{0,l}\|_F^2
\]

这和 `prox` 类似，但 v3 应配合 residual aggregation 使用，权重不宜太大。

### 10.2 残差方向多样性约束

鼓励不同 rank row 承载不同方向：

\[
L_{div} = \sum_l \| \hat{R}_{i,l}\hat{R}_{i,l}^{T} - I \|_F^2
\]

只对残差范数超过阈值的 row 生效，避免无效 row 参与。

### 10.3 B-private 兼容约束

由于 `B_i` 不上传，服务端不能随意旋转 `A` 坐标。v3 仍不建议默认 QR。训练时增加：

\[
L_{compat} =
1 - cos(B_i A_i, B_i A_0)
\]

或低成本版本：

\[
L_{row\_scale} =
\sum_r
\left|
\|B_{i,:,r}\|_2
\cdot
\|A_{i,r,:}\|_2
-
\|B_{i,:,r}\|_2
\cdot
\|A_{0,r,:}\|_2
\right|
\]

目标是让服务端替换 `A_i` 为 `A_down` 时，本地 `B_i` 不至于失配。

---

## 11. 模块七：可选公共锚点评估校准

如果服务器可使用少量公共或合成 instruction anchor set，可以在不增加客户端通信的情况下做后验校准。

### 11.1 校准内容

候选：

- `A_common`
- `A_common + A_risk`
- `A_common + A_capability`
- `A_common + A_general_education`
- 不同 `lambda_cluster`

服务器在 anchor set 上选择：

\[
\lambda^* = \arg\min_{\lambda} L_{anchor}(W_0 + B_{proxy} A_{down})
\]

由于服务器没有真实 `B_i`，可以使用以下 proxy：

- 使用平均 `B_proxy`，但这会破坏 `B` 私有原则，不建议主线采用。
- 使用 `A` 侧 perplexity proxy 或 small validation prompt 的 loss proxy，工程上需要额外设计。
- 只用公共 anchor 选择 row gate / cluster weight，不接触 `B_i`。

### 11.2 建议定位

该模块收益可能高，但工程成本和审稿风险也更高。建议作为 v3 扩展，不作为主线必需组件。

---

## 12. 推荐 v3 版本拆分

为了降低实现和实验风险，建议将 v3 分成三个子版本。

### 12.1 v3-lite：残差冲突门控

只改服务端聚合：

- 用 `R_i = A_i - A0` 替代 `A_i` 计算 conflict
- 使用 layer-wise quantile threshold
- 高冲突 row 仍可先回退到 `A0`

优点：

- 改动小
- 最容易验证 conflict gate 是否真的触发

风险：

- 仍然只有单个 `A_global`
- 可能提升 worst，但 macro 仍未必明显提升

### 12.2 v3-cluster：层次化个性化 A 下发

在 v3-lite 基础上增加：

- 领域簇
- `A_common + A_cluster`
- 不同客户端一次下发不同 `A_down`

优点：

- 最符合“跨域个性化 one-shot”故事
- 有机会同时提升 macro 和 hard/worst

风险：

- 需要改评估逻辑，使不同客户端使用不同 `A_down`
- 通信统计要区分 server 总下发量与每客户端下发量

### 12.3 v3-rpca：common + sparse 分解

在 v3-cluster 基础上增加：

- Robust PCA / low-rank + sparse decomposition
- common residual 和 sparse residual 分开聚合

优点：

- 故事性最强
- 和 FedRPCA 有明确联系，但我们的 one-shot + B-private + cross-domain 个性化边界清楚

风险：

- 实现复杂度最高
- 大模型 LoRA 层多，需要做 per-layer 或 block-wise 分解，不能直接对全量参数做昂贵 RPCA

---

## 13. 推荐实验路线

### 13.1 第一阶段：7c 快速验证

只跑 7 个客户端，每域 1 个客户端：

| 实验 | 目的 |
|---|---|
| v2 | 现有基线 |
| v3-lite | 看 residual conflict 是否触发 |
| v3-lite + quantile 0.8 | 看自适应阈值是否稳定 |
| v3-cluster | 看个性化 A 是否提升 macro |
| v3-cluster no cluster residual | 消融 cluster 残差 |

重点记录：

- `domain_macro_loss`
- `worst_domain_loss`
- `hard_domain_avg_loss`
- `domain_macro_token_accuracy`
- `conflict_mean`
- `conflict_high_row_frac`
- `cluster_gate_mean`

### 13.2 第二阶段：35c 主实验

只保留最优两个 v3 变体：

| 方法 | 说明 |
|---|---|
| v2 FedPLoRA-Oneshot | 当前版本 |
| v3-lite | 最小新增 |
| v3-cluster | 主推版本 |
| YOCO-style | one-shot baseline |
| FedSA-LoRA | A-only baseline |
| FedALT | personalized baseline |

### 13.3 第三阶段：消融

只在 7c 或 14c 做：

| 消融 | 说明 |
|---|---|
| raw `A` conflict vs residual `ΔA` conflict | 证明残差空间必要性 |
| fixed threshold vs quantile threshold | 证明自适应阈值必要性 |
| A0 fallback vs cluster fallback | 证明高冲突不应简单丢弃 |
| domain cluster vs learned cluster | 证明领域先验是否有效 |
| with / without row-importance | 证明私有 `B` 统计价值 |
| with / without common-sparse decomposition | 证明 RPCA 模块价值 |

---

## 14. 论文主表建议

建议主表不要只放 `Macro`，因为 v3 的核心是跨域冲突和个性化稳健性。

主表列：

| 方法 | Macro↓ | Worst↓ | Hard avg↓ | Token Acc↑ | Worst Acc↑ | Comm GB↓ |
|---|---:|---:|---:|---:|---:|---:|

补充表：

| 方法 | code | math | medical | legal | finance | general | education |
|---|---:|---:|---:|---:|---:|---:|---:|

机制表：

| 方法 | mean conflict | high-row frac | cluster gate | common ratio | sparse ratio |
|---|---:|---:|---:|---:|---:|

这样可以避免审稿人只盯着 `Macro` 持平，而忽略 hardest domain 和高风险域鲁棒性。

---

## 15. 与相关工作的边界

### 15.1 相对 FedSA-LoRA

FedSA-LoRA 证明 `A` 更偏 general knowledge，`B` 更偏 client-specific knowledge，因此只共享 `A`。v3 不 claim 第一个 A-only FedLoRA，而是强调：

- one-shot
- cross-domain personalized LLM SFT
- residual-space conflict modeling
- hierarchical personalized A downlink

### 15.2 相对 YOCO

YOCO 强调 true one-shot，并通过初始预训练模型提供隐式全局监督，使用方向约束和 PCWA。v3 的区别：

- 不上传 / 聚合 `B`
- 不做全局 `A+B` 组合
- 在 `ΔA` 残差空间做冲突分解
- 下发 cluster-personalized `A_down`

### 15.3 相对 FedRPCA

FedRPCA 使用 Robust PCA 将 LoRA 更新拆成 common low-rank 与 client-specific sparse。v3 借鉴 common/sparse 分解，但差异是：

- 我们只上传 `A`
- `B_i` 完全私有
- 只通信一次
- 面向跨域 LLM SFT
- sparse residual 不直接全局平均，而是进入 domain-cluster 个性化路径

### 15.4 相对 FedALT / FDLoRA

FedALT 和 FDLoRA 都强调个性化组件与共享组件的组合。v3 的区别：

- 不需要多轮训练
- 不引入额外客户端上传的完整个性化 LoRA
- 个性化主要来自本地 `B_i` 与服务端一次性生成的 `A_cluster`

### 15.5 相对 one-shot FL 蒸馏类方法

DENSE、FedHydra、FedSD2C、OSCAR 等 one-shot FL 方法通常依赖数据生成、蒸馏或结构化合成信息。v3 的主线不依赖公共数据或生成器，只用 LoRA `A` 与轻量统计。公共锚点校准可以作为可选扩展，不作为核心假设。

---

## 16. 预期收益与风险

### 16.1 预期收益

- `Worst`：应继续优于 YOCO-style 和 FedSA-LoRA。
- `Hard avg`：风险域 cluster residual 应进一步提升 medical / legal / finance。
- `Macro`：通过 cluster-specific A 恢复 general / math 等被单一全局 A 拉低的领域，有机会超过 v2。
- 通信：上行仍保持 A-only；下行每客户端仍只收一个 A_down，单客户端通信基本不变。

### 16.2 主要风险

- cluster residual 太强会损害跨域泛化。
- 如果数据集本身领域差异不够强，v3 的优势可能主要体现在 hardest domain。
- RPCA 对大模型 LoRA 层逐层处理可能耗时，需要 block-wise 简化。
- 下发不同 `A_down` 后，评估逻辑要严格保证每个客户端使用对应 A，否则结果会混乱。

---

## 17. 推荐默认配置

v3-lite：

```text
residual_conflict = true
threshold_mode = quantile
conflict_quantile = 0.80
residual_norm_power = 1.0
importance_power = 1.0
cluster_mode = none
```

v3-cluster：

```text
residual_conflict = true
threshold_mode = quantile
conflict_quantile = 0.80
cluster_mode = domain_prior
cluster_groups = general_education,capability,risk
cluster_lambda_min = 0.2
cluster_lambda_max = 1.0
cluster_gate_temperature = 0.05
```

v3-rpca：

```text
residual_conflict = true
common_sparse_decompose = rpca
rpca_scope = per_layer
rpca_max_iter = 50
rpca_sparse_scale = 0.5
cluster_mode = domain_prior
```

---

## 18. 最推荐实现顺序

1. 先实现 v3-lite：验证 `ΔA` 残差冲突能否让 `high_conflict_row_frac` 从 0 变成合理值。
2. 再实现 v3-cluster：每个客户端下发对应领域簇的 `A_down`。
3. 最后实现 v3-rpca：如果 v3-cluster 已明显提升，再加入 common/sparse 分解增强故事性。

如果时间有限，只做前两步。v3-cluster 的论文故事已经足够清楚：

> One-shot personalized FL should not force all domains to share the same post-aggregation LoRA A. Low-conflict residuals should be shared globally, while high-conflict residuals should be routed to domain-level personalized A adapters, with B kept private locally.

---

## 19. 代码接口与运行命令

### 19.1 数据集接口

若已有域划分后的联邦数据，直接传入：

```bash
--benchmark_dir <DOMAIN_BENCHMARK_SPLIT_DIR>
```

若需要从原始 JSONL 构建联邦域划分：

```bash
--build_benchmark --benchmark_jsonl <RAW_DOMAIN_JSONL> --benchmark_output_dir data/domain_benchmark
```

原始 JSONL 需要至少包含 `domain`、`prompt`、`response` 字段。

### 19.2 模型接口

模型路径支持本地路径或 Hugging Face ID：

```bash
--model <MODEL_PATH_OR_HF_ID>
```

### 19.3 v3-lite

```bash
python tasks/fed_train_sft.py \
  --model <MODEL_PATH_OR_HF_ID> \
  --benchmark_dir <DOMAIN_BENCHMARK_SPLIT_DIR> \
  --agg_type fedplora_v3_lite \
  --rounds 1 \
  --local_epochs 1 \
  --batch_size 2 \
  --lora_r 8 \
  --lora_alpha 16 \
  --v3_conflict_quantile 0.8 \
  --v3_gate_temperature 0.05 \
  --v3_conflict_blend 1.0 \
  --oneshot_anchor_lambda 1e-4
```

### 19.4 v3-cluster

```bash
python tasks/fed_train_sft.py \
  --model <MODEL_PATH_OR_HF_ID> \
  --benchmark_dir <DOMAIN_BENCHMARK_SPLIT_DIR> \
  --agg_type fedplora_v3_cluster \
  --rounds 1 \
  --local_epochs 1 \
  --batch_size 2 \
  --lora_r 8 \
  --lora_alpha 16 \
  --v3_conflict_quantile 0.8 \
  --v3_gate_temperature 0.05 \
  --v3_cluster_mode domain_prior \
  --v3_cluster_lambda_min 0.2 \
  --v3_cluster_lambda_max 1.0 \
  --oneshot_anchor_lambda 1e-4
```

### 19.5 v3-rpca

```bash
python tasks/fed_train_sft.py \
  --model <MODEL_PATH_OR_HF_ID> \
  --benchmark_dir <DOMAIN_BENCHMARK_SPLIT_DIR> \
  --agg_type fedplora_v3_rpca \
  --rounds 1 \
  --local_epochs 1 \
  --batch_size 2 \
  --lora_r 8 \
  --lora_alpha 16 \
  --v3_conflict_quantile 0.8 \
  --v3_gate_temperature 0.05 \
  --v3_cluster_mode domain_prior \
  --v3_rpca_rank 1 \
  --v3_sparse_quantile 0.8 \
  --oneshot_anchor_lambda 1e-4
```

### 19.6 运行后输出

训练完成后会在 `artifacts/sft_metrics/` 下生成 round-wise 指标 JSON，包含：

- `domain_macro_loss`
- `worst_domain_loss`
- `hard_domain_avg_loss`
- `fedplora_v3_stats`
- `fedplora_v3_client_clusters`

## 20. 参考文献

- Selective Aggregation for Low-Rank Adaptation in Federated Learning: https://arxiv.org/abs/2410.01463
- You Only Communicate Once: One-shot Federated Low-Rank Adaptation of MLLM: https://proceedings.neurips.cc/paper_files/paper/2025/hash/58e6c003c9fb3992265005ff6aef1913-Abstract-Conference.html
- FedRPCA: Enhancing Federated LoRA Aggregation Using Robust PCA: https://arxiv.org/abs/2506.01194
- FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA: https://arxiv.org/abs/2505.12805
- DENSE: Data-Free One-Shot Federated Learning: https://arxiv.org/abs/2112.12371
- FedHydra: A Unified Solution to Diverse Heterogeneities in One-shot Federated Learning: https://arxiv.org/abs/2410.21119
- FedSD2C: One-shot Federated Learning via Synthetic Distiller-Distillate Communication: https://arxiv.org/abs/2412.05186
- OSCAR: One-Shot Federated Learning with Classifier-Free Diffusion Models: https://arxiv.org/abs/2502.08488
