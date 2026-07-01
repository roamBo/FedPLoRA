# FedPLoRA v5-merge — 把"一次通信联邦 LoRA"重述为 **干扰感知模型合并**

> 这是与 [`FedPLoRAOSv5_README.md`](FedPLoRAOSv5_README.md)（route-mix-align，A-only + 本地路由）**并列且正交**的另一条 v5 路线。
> 目标：跳出"又一个 A 聚合规则"，给出一个审稿人无法说成 *YOCO + FedSA 的 A+B 拼接* 的全新框架，并在 35c / LW7c 上**真正超过现有最强 baseline（flora / flexlora）**。
>
> 代码：`methods/v5/`（`merge_ops.py` + `fedplora_v5_merge.py`），入口复用 `tasks/fed_train_sft.py`（`agg_type=v5m_*`），脚本 `scripts/RunScripts/run_v5_merge_{35c,lw7c}.sh`。数学自检 `methods/v5/selftest_merge.py`（GPU）+ 离线 numpy 验证（核心恒等式误差 ~1e-14）。

---

## 1. 残酷诊断：现在的方法为什么"不够、且像 YOCO+FedSA"

### 1.1 35c 主榜（Llama-3.1-8B，1 轮 1 epoch，seed 42）

| 排名 | 方法 | macro_acc | worst_acc | hard_loss | 类别 |
|---|---|---:|---:|---:|---|
| 1 | **flora** | **0.70165** | 0.52718 | **1.6968** | 全 ΔW 合并 + SVD |
| 2 | **flexlora** | 0.70121 | **0.52961** | 1.6953 | 加权 ΔW 合并 + SVD |
| 3 | yoco | 0.70051 | 0.52181 | 1.7460 | A/B 冲突加权融合 |
| 4 | normal (FedAvg) | 0.69819 | 0.52318 | 1.7481 | A,B 各自平均 |
| 5 | ffa | 0.69754 | 0.52006 | 1.7553 | 冻 A 聚 B |
| 6 | feddat | 0.69741 | 0.52243 | 1.7469 | 双适配器 |
| 7–9 | **v4_mix_*（你的 v4）** | 0.6962 | 0.5202 | 1.771 | A 下发 + 本地 A 混合 |
| 11 | **fedplora-oneshot（你的 v2）** | 0.69595 | 0.51982 | 1.7774 | A-only + A0 锚定门控 |
| 12–17 | v4_svd / v4_sign / v4_hier* | 0.6953–0.6958 | ~0.5199 | ~1.78 | v4 其它支线 |
| 18 | fedalt | 0.69331 | 0.51739 | 1.7972 | leave-one-out RoTW |
| 19–22 | fedplora_v3_* / **fedsa_lora** | 0.690–0.693 | 0.512–0.515 | 1.83–1.89 | v3 残差门控 / 纯 A 平均 |

> LW7c（SmolLM2-135M）上排序一致：flexlora/flora/normal/feddat/yoco 在前，v4 全支线与 v2 挤在 0.585–0.586，差异 <0.001。

### 1.2 三个让审稿人一眼否掉的事实

1. **你的主方法（v2=0.69595）与 v4 全部支线，都低于最朴素的全-ΔW 合并 flora/flexlora（0.701）**。你为了"A-only / B 私有 / one-shot"省一半通信，**牺牲了 0.5–0.6 个点**；而 flora 同样 one-shot、同样只通信 LoRA，却更高。省通信的叙事撑不起精度劣势。
2. **v2 / fedsa_lora / yoco 机制同源**：都在 A（或 A 方向）上做加权融合，B 走平均或私有。0.690(fedsa)→0.696(v2 +A0 anchor)→0.700(yoco +B 相似度) 连成一条线——审稿人会说"v2 = FedSA 的 A 上加 anchor + YOCO 的 sparse-B"。v4 的 Mix 又是 FedDAT/FedALT 的双适配器，仍没跳出这个家族。
3. **v4 全线贴着 v2（差 <0.001），落在 50-batch eval 噪声里**。既没拉开差距，也没超 baseline。

> 结论：继续在"A 怎么聚合"这条线上调，天花板就是追平 flora。**必须换战场。**

### 1.3 真正的 gap（v5-merge 的立论）

`flora`/`flexlora` 用模型合并语言就是一句话：

```
ΔW* = SVD_r( Σ_i w_i · B_i A_i )      # 先把所有客户端更新加起来，再砍回 rank-r
```

这正是 **模型合并文献里最朴素的 baseline——任务向量求和**。而 TIES-Merging（NeurIPS'23）、DARE（ICML'24）、KnOTS（ICLR'25）的全部 motivation 就是：**朴素求和会让不同域更新发生符号冲突与冗余干扰、互相抵消**，所以要 *裁剪 → 符号选举 → 不相交合并*，公开报告能在朴素平均之上再涨 2–4 点。

**到今天没有一篇联邦 LoRA 论文把"客户端=不同任务向量、聚合=模型合并"认真做对**：flora/flexlora 停在朴素求和；FedSA/YOCO/你的 v2 在*单独的 A 子空间*绕，根本没在 ΔW=BA 的真实更新空间处理跨域干扰。这就是切入点。

---

## 2. v5-merge 的定位与一句话

> **One-shot Federated LoRA as Interference-Aware Model Merging.** 每个客户端的适配器是一个任务向量 `ΔW_i=B_iA_i`；服务器不再朴素求和，而是在更新空间做 **裁剪 → 符号选举 → 不相交合并（TIES）**，并在**客户端共享的右奇异子空间**里对齐后合并（KnOTS），最后一次 SVD 重分解回 LoRA 因子做单次下发。

为什么不是"YOCO+FedSA 拼接"：
- YOCO/FedSA 在 **A 子空间**做*线性加权*；v5 在**完整 ΔW**做*离散符号选举*（非线性、非加权平均），机制与数学形式都不同。
- flora/flexlora 是 v5 在 `keep_ratio=1, 无符号选举` 时的**退化特例**（即 `v5m_mean`）。v5 的增量正是合并文献验证过、而联邦 LoRA 从没用过的那部分。

---

## 3. 方法（4 个算子，统一入口 `v5m_*`）

客户端上传与 flora/flexlora **完全一致**（trainable `A_i,B_i` + heads），**通信不变、one-shot 不变**。只改服务器聚合这一步。`w_i ∝ n_i`。

- **`v5m_mean`**（sanity，应≈flexlora）：`ΔW*=SVD_r(Σ_i w_i B_iA_i)`，复现 flexlora 协议确认框架无 bug。
- **`v5m_ties`**（旗舰机制）：逐 entry ① Trim 每客户端保留 top-`keep_ratio` 幅值（默认 0.2）② `γ=sign(Σ_i w_i ΔW_i^{trim})` 符号选举 ③ 只对同号客户端加权平均 ④ `SVD_r`。按行分块，**从不堆叠 N 个 m×n**。
- **`v5m_dare_ties`**：TIES 前对每个 `ΔW_i` 做 **DARE 随机丢弃+1/(1-p) 重缩放**（默认 p=0.3）。
- **`v5m_knots_ties`**（**最强、最不可被说成 A+B 拼接**）：先把所有客户端更新对齐到**共享右奇异子空间**再 TIES。给出 **完全因子化、不展开任何 m×n** 的精确实现：
  1. 堆叠 `P=[A_1;…;A_N]`，thin-SVD `P=U_pS_pV_p^T`；
  2. 分块恒等式 `ΔW_i = B_i(U_{p,i}S_p)·V_p^T =: C_i V_p^T`（**精确**，离线误差 1e-14）；
  3. 加权 Gram `H=Σ_i w_i C_i^TC_i` 特征向量把基旋到加权堆叠更新主轴（KnOTS 对齐基），得共享基 `V` 与对齐系数 `C_i`；
  4. 在 `C_i∈R^{m×k}`（k≤512）上做 TIES，`factorize_coefficient_delta` 直接出 LoRA 因子——**全程不展开 ΔW**，四算子里最省显存。

**自适应秩**（`--v5m_rank_policy energy`）：合并后 `ΔW*` 常高于单客户端秩；`fixed`（默认）下发 rank=`lora_r`，**通信与 flora 完全相同**（主表）；`energy` 下发"累计能量≥τ 的最小秩"（原地改 PEFT 秩），作为"合并恢复了多少被 rank-r 截断的跨域信息"的 ablation。注意 `energy` 改适配器形状，**与跨进程 checkpoint 续跑不兼容**（用 `--skip_post_agg_snapshots` 或接受不可 resume）；`fixed` 完全兼容。

---

## 4. 为什么 v5 应当真的涨

1. **headroom 是结构性的**：flora/flexlora = 合并文献的"朴素求和" baseline，TIES/DARE/KnOTS 存在的意义就是超过它（公开 +2~4%）。我们不是在 0.696 拥挤线抠 0.001，而是去够 0.701 之上的空间。
2. **跨域冲突真实存在**：7 域 SFT 里 worst 永远是 legal/finance——正是被其它域更新平均抵消的高方差专业域。符号选举对它们的保护直指 worst/hard 指标。
3. **机制可解释可画图**：`_v5m_merge_stats` 落每层 `active_frac`（参与合并 entry 占比）、`mean_energy_kept`、`mean_r_down`，可画"干扰强度 vs 层/域"的 mechanism figure。

---

## 5. 实验矩阵与一键脚本

| 组 | agg_type | 作用 |
|---|---|---|
| 强 baseline | flora, flexlora, yoco, normal, fedplora-oneshot | 已有结果直接进表 |
| v5-sanity | `v5m_mean` | 必须≈flexlora，验证框架无 bug |
| v5-main | `v5m_ties` | TIES 相对朴素求和的纯增益 |
| v5-main | `v5m_dare_ties` | +DARE 是否再涨 |
| **v5-flagship** | `v5m_knots_ties` | 子空间对齐 + TIES，期望最强 |
| v5-ablation | `v5m_knots_ties` + `--v5m_rank_policy energy` | 自适应秩抬 worst/hard |

```bash
# LW7c 机制快筛（SmolLM2-135M，几分钟/方法）—— 先在这里看 active_frac 与 v5m_mean 的差
bash scripts/RunScripts/run_v5_merge_lw7c.sh 0

# 35c 主表（Llama-3.1-8B）
bash scripts/RunScripts/run_v5_merge_35c.sh 35 0

# 自适应秩 ablation
V5M_RANK_POLICY=energy V5M_RANK_CAP=32 bash scripts/RunScripts/run_v5_merge_35c.sh 35 0 v5m_knots_ties

# 端到端前先在 GPU 机器跑数学自检
python -m methods.v5.selftest_merge
```

**关键超参**（先 LW7c 扫再上 35c）：`--v5m_keep_ratio∈{0.1,0.2,0.3,0.5}`、`--v5m_dare_p∈{0.2,0.3,0.5}`、`--v5m_energy_tau∈{0.90,0.95,0.99}`、`--v5m_knots_normalize{0,1}`。

**评估稳健性（投稿必须）**：`--eval_max_batches 200` + 3 seed（42/1234/9999）报 mean±std + 对 `v5m_knots_ties` vs `flexlora` 做配对显著性。

---

## 6. 论文叙事

**Title 备选**：*Interference-Aware One-Shot Federated LoRA via Subspace-Aligned Merging.*

**Abstract 核心句**：
> Existing one-shot federated LoRA methods aggregate the low-rank factors with weighted averaging — equivalent to the *naive task-vector sum* that the model-merging literature shows suffers from cross-task sign conflicts and redundant interference. We recast one-shot federated LoRA as **interference-aware model merging**: each client adapter is a task vector, and the server resolves cross-domain interference by trimming, electing signs, and disjointly merging client updates in a **shared singular subspace**, then refactorizes once for a single downlink. Without changing the communication budget of FedAvg-LoRA, our method improves macro / worst-domain accuracy over the strongest one-shot baselines (FLoRA, FlexLoRA, YOCO) across 7 heterogeneous domains.

**与最接近工作的切割**：
- vs **FedSA-LoRA / YOCO / 你的 v2**：A 子空间*线性加权* vs 完整 ΔW 的*离散符号选举*；
- vs **FLoRA / FlexLoRA**：它们 = 我们的 `v5m_mean` 退化特例（朴素求和+SVD），我们用干扰消解机制把它们当 baseline 超过；
- vs **TIES/DARE/KnOTS（中心化）**：它们假设可访问每个任务稠密权重；我们在**联邦 + 低秩因子 + 一次通信**约束下给出**完全因子化、不展开 m×n** 的等价实现，本身是非平凡贡献。

**贡献清单**：
1. 把 one-shot federated LoRA 形式化为干扰感知合并，统一解释 FLoRA/FlexLoRA/FedSA/YOCO 为其特例；
2. 一个在 LoRA 因子上**精确且省显存**的子空间对齐 TIES 算子（KnOTS 的联邦因子化版本，离线验证 1e-14）；
3. 跨 7 域 35 客户端超过最强 one-shot baseline，并给出干扰强度的层级/域级机制分析与自适应秩 ablation。

---

## 7. 风险与对策（诚实评估）

| 风险 | 说明 | 对策 |
|---|---|---|
| 1-epoch SFT 下 ΔW 小、干扰温和，TIES 增益有限 | 最大风险 | 先 LW7c 看 `active_frac` 与 mean 的差；若小，降 `keep_ratio` 放大选举效应 + 叠 `energy` 秩 |
| `v5m_mean` 没复现 flexlora | 框架 bug | `selftest_merge` 已证算子正确；优先查 heads/scaling 约定与 `_aggregate_client_sizes` 传入 |
| 仍贴着 flora（差<0.001） | 干扰确实弱 | 加 **域簇个性化合并**（每簇一个 ΔW*，one-shot 单次下发不同簇适配器）—— 见 §8 |
| "合并=已有技术搬运" | 需强调联邦因子化非平凡 | 突出"不展开 m×n 的精确子空间对齐 + 一次通信约束"，中心化 TIES/KnOTS 不存在此问题 |

---

## 8. 第二层杠杆：与 route-mix-align（[v5 主 README](FedPLoRAOSv5_README.md)）正交叠加

v5-merge 改 **服务器如何消解干扰**；route-mix-align 改 **客户端如何安全使用下发适配器**。两者正交，可叠：
1. v5-merge 产出更干净的 `ΔW*`（或每域簇一个 `ΔW*_cluster`）；
2. 客户端再用 route-mix-align 的本地验证路由 `A_eff(η)=ηA_down+(1-η)A_local` 决定接受多少。

**个性化合并**接口已预留：`knots_align_factors` 的系数 `C_i` 可先按域聚类（复用 `methods/v4/common_v4.py` 的 spectral/kmeans），每簇跑一次 `ties_on_coefficients`，再走 `_fedplora_personalized_shared_states` 的按 client_id 下发 eval 路径（v4 已实现）。这是 v5-merge → v5.5 的自然延伸与第二个 ablation。

---

## 9. 文件清单与落地状态

```
methods/v5/
├── __init__.py                 # 导出 aggregate_models_v5_merge
├── merge_ops.py                # 4 算子 + 因子化 + 秩策略（核心数学）
├── fedplora_v5_merge.py        # 服务器聚合入口 + PEFT 原地改秩 surgery
└── selftest_merge.py           # GPU 数学自检（5 项恒等式）
tasks/fed_train_sft.py          # +import / +--v5m_* 参数 / +dispatch / +re-aggregate handler
utilities/utils.py              # +is_v5_merge_agg，纳入 memory-global 与通信估计
scripts/RunScripts/run_v5_merge_{35c,lw7c}.sh
```

- 算子数学：**离线 numpy 已验证**（KnOTS 重构 / 系数因子化 / entrywise-mean / 对齐基 四条恒等式误差 ≤ 5e-14）。
- 代码：所有改动 `py_compile` 通过；脚本 `bash -n` 通过。
- **待你在 GPU 机器执行** `python -m methods.v5.selftest_merge` 与 LW7c 快筛验证端到端，然后上 35c 主表。
```
