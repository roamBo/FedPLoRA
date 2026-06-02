# FedPLoRA v4 实验说明

> 代码布局（与 v2/v3 并列，均在仓库根）：`methods/v4/`、`tasks/fed_train_sft_v4.py`、`configs/v4_baseline.env`、`scripts/RunScripts/run_v4_*.sh`。支线设计见 `docs/v4/branches/`。

### Branch A — **FedPLoRA-Hier++**

**一句话**：把 v3 的硬冲突门控改成**软残差混合 + 数据驱动簇 + 个性化 A 路径强制使用**，目标是把 v3-cluster 从 `Macro -0.003` 翻回 `Macro +0.005`。

- 残差空间不变：`R_i = A_i − A_0`。
- 门控：用 *soft sigmoid blending*，`A_down = A_0 + (1−g)·R_common + g·R_cluster`；当 `g=0` 时退化为 v2，当 `g=1` 时全簇路径。
- 簇定义：可选 `domain_prior`（与 v3 同），`spectral`（对 `[vec(R_i)]` 做 K-Means / Spectral 聚类得 K 个簇）。
- **关键修复**：评估时严格按客户端 ID 加载 `personalized_shared_states[client_id]`，并确认主路径 `global_dict[key]` 不要被 default A 覆盖（修 v3 bug）。
- 详见 [docs/v4/branches/BRANCH_A_HierPlus.md](docs/v4/branches/BRANCH_A_HierPlus.md)，实现见 [methods/v4/fedplora_v4_hier.py](methods/v4/fedplora_v4_hier.py)。

### Branch B — **FedPLoRA-SVD**

**一句话**：初始化阶段对 `A_0` 做 QR 正交，one-shot 后做一次"客户端联合更新"的 SVD 重分解，每个客户端拿到的 A 都对齐到同一组主成分坐标系，从而避免 v2 中 A_i 与本地 B_i 坐标失配的问题。

- 初始化：`A_0` 用 Kaiming 后立即 `A_0 = Q^T` 其中 `Q,_ = qr(A_0^T)`。
- 客户端本地训练正常进行（与 v2 一致）。
- 聚合：把 `[A_1; A_2; …; A_N]` 拼成 `(N·r, d)` 矩阵，做 SVD 得 `U S V^T`，取 top-r 行作为新的 `A_new = V[:r]^T`（带 sign-disambiguation）。
- 下行：每个客户端收到统一的 `A_new`；本地 `B_i` 在第一次前向时先做一次小步 LR 的"快速对齐"再正常使用（可选）。
- 详见 [docs/v4/branches/BRANCH_B_SVD.md](docs/v4/branches/BRANCH_B_SVD.md)。

### Branch C — **FedPLoRA-Sign**

**一句话**：在本地训练目标里加 `B 方向正则`，使所有客户端的 `sign(B_i)` 与初始 `sign(B_0)` 保持一致，让 A 的聚合空间天然兼容多客户端的 B；这是 *最不改通信* 的支线。

- 本地训练新增项：`λ_Bsign · ||tanh(γ B_i) − sign(B_0)||_1`（YOCO 思路）。
- 本地训练新增项：`λ_Asparse · ||A_i||_1`（与 YOCO sparse-A 一致，原 `--yoco_sparse_lambda` 已存在，复用）。
- 服务端聚合：与 v2 fedplora-oneshot 完全一致，**不修改通信、不修改聚合**。
- 详见 [docs/v4/branches/BRANCH_C_Sign.md](docs/v4/branches/BRANCH_C_Sign.md)，实现见 [methods/v4/fedplora_v4_sign.py](methods/v4/fedplora_v4_sign.py)（实际只是本地正则项，聚合复用 v2）。

### Branch D — **FedPLoRA-Mix**

**一句话**：每个客户端保留两份 A —— 本地训练得到的 `A_i^{local}`、服务端下发的 `A_down`。推理时通过 *学习得到* 的标量或 per-instance 路由 η_i 做 MoE-style mixing：`ΔW = B_i (η · A_down + (1−η) · A_local)`。

- **不增加通信**：A_local 已经在客户端，A_down 由服务端 one-shot 返回。
- 三种 mixer：
  1. `fixed`：η=0.5（最简，0 训练成本）。
  2. `per_domain`：每个域用 50 条 val 数据搜出最优 η。
  3. `per_input`（MoE）：一个轻量门控网络 `g(x) ∈ [0,1]` 在客户端本地训练。
- 详见 [docs/v4/branches/BRANCH_D_Mix.md](docs/v4/branches/BRANCH_D_Mix.md)，实现见 [methods/v4/fedplora_v4_mix.py](methods/v4/fedplora_v4_mix.py)。
- **预测**：Macro 提升最有希望（A_local 保留个性化）；Worst 收益取决于 legal/finance 是否真的偏好本地 A。

### Branch E — **FedPLoRA-Anchor**

**一句话**：服务端持有 7 × 50 条公开 instruction（HF datasets `flan_v2`、`alpaca_eval` 等），聚合后用这些 anchor 做 *后验超参选择*：从一组候选 `{λ_cluster, gate_threshold, R_common rank}` 中选 anchor 上 loss 最低的组合。

- 是否引入 anchor 是 paper-defendable trade-off：v4 给出 `--use_anchor 0/1` 开关，0 时退化为 Branch A。
- anchor 数据放 `data/anchor_seed/seed_42/`，固定不随客户端动。
- 详见 [docs/v4/branches/BRANCH_E_Anchor.md](docs/v4/branches/BRANCH_E_Anchor.md)。

### Branch F — **FedPLoRA-AdaRank**

**一句话**：让 medical/legal/finance 用 r=16，math/code 用 r=8，general/education 用 r=4，total params 不变；聚合时按 layer × row 对齐到 r_max=16，未使用 row 被忽略。

- 客户端按域读取 rank from manifest。
- 上行 `A_i` 是 (r_i, d)，服务端 pad 到 (16, d) 后聚合；下行 truncate 回客户端的 r_i。
- 通信：变化范围 ±25%，不破坏 one-shot 叙事。
- 详见 [docs/v4/branches/BRANCH_F_AdaRank.md](docs/v4/branches/BRANCH_F_AdaRank.md)。

---

## 实验矩阵（35c 主实验）

为每条支线给出 ≤3 个对比设置，全部走 1 轮 1 epoch，固定 `seed=42`：


| Branch          | 配置                                     | `agg_type`              | 假设                 |
| --------------- | -------------------------------------- | ----------------------- | ------------------ |
| baseline-v2     | 复跑 v2                                  | `fedplora_oneshot`      | 锚点对照               |
| baseline-fedalt | 复跑 FedSA-LoRA                          | `fedalt`                | A-only mean 对照     |
| A1              | Hier++ soft gate, prior cluster        | `v4_hier_soft_prior`    | 修 v3 退化，Macro ≥ v2 |
| A2              | Hier++ soft gate, spectral cluster K=3 | `v4_hier_soft_spectral` | 数据驱动簇 ≥ 先验         |
| A3              | Hier++ soft + 强制 personalized eval     | `v4_hier_soft_pfl_eval` | 验证 v3 bug 修复       |
| B1              | A_0 正交，无 SVD 重分解                       | `v4_svd_orth_only`      | 只看正交化收益            |
| B2              | A_0 正交 + 聚合 SVD 重分解                    | `v4_svd_full`           | 完整 FedSVD-style    |
| C1              | YOCO Bsign 正则，A 聚合用 v2                 | `v4_sign_v2agg`         | Bsign 是否独立有用       |
| C2              | YOCO Bsign + sparse-A，A 聚合用 v2         | `v4_sign_full`          | 完整 YOCO local 增强   |
| D1              | Mix fixed η=0.5                        | `v4_mix_fixed05`        | 双路径基线              |
| D2              | Mix per_domain η（grid on val）          | `v4_mix_per_domain`     | 域级 mixer           |
| D3              | Mix per_input MoE                      | `v4_mix_moe`            | 学习 mixer           |
| E1              | Anchor 50/域，调 `gate_threshold`         | `v4_anchor_gate`        | anchor 调阈值         |
| E2              | Anchor 50/域，调 `cluster_lambda`         | `v4_anchor_lambda`      | anchor 调融合         |
| F1              | risk r=16, others r=8                  | `v4_adarank_risk16`     | 难域加容量              |
| F2              | risk r=16, gen/edu r=4, math/code r=8  | `v4_adarank_full`       | 全异构                |


总 16 个 run，加 baseline 共 18 个；按 35c 单 run 约 1.5–2h 估算 35–60 GPU·h，单卡 A100/H100 1–2 天可跑完。

### 一键脚本（35c）

| 脚本 | Branch | 包含配置 |
|------|--------|----------|
| `run_v4_baseline.sh` | baseline | v2 oneshot + fedalt |
| `run_v4_branch_a.sh` | A | A1/A2/A3 |
| `run_v4_branch_b.sh` | B | B1/B2 |
| `run_v4_branch_c.sh` | C | C1/C2 |
| `run_v4_branch_d.sh` | D | D1/D2/D3 |
| `run_v4_branch_e.sh` | E | E1/E2（Stage 5 stub） |
| `run_v4_branch_f.sh` | F | F1/F2（Stage 4 stub） |
| `run_v4_pilot_7c.sh` | 快验 | 7 客户端 smoke |

**推进顺序**：主线先 A → C → D；B/F 为 Stage 4，E 为 Stage 5。E/F 当前聚合器为 **stub**（分别回退 Hier++ / v2 oneshot），脚本可跑通 pipeline，但机制未完整实现。

### 防白训（与 v2 一致）

`tasks/fed_train_sft_v4.py` 默认启用 run checkpoint（`../trained_models/<agg>_<model>_<benchmark>_r1_e1_seed42/`）：

1. **聚合后、eval 前**：`snapshots/round_001_post_agg/` — eval 崩溃后重跑 **跳过 35 客户端训练**，只做 eval。
2. **eval 完成后**：根目录 `checkpoint_ok.json`（`phase=final`）— 同配置 **训练 + eval 均跳过**。
3. 必须加 **`--save_client_state_to_disk`**（各 `run_v4_*.sh` 已包含），否则 bundle 缺 `clients/*.pt`。
4. 强制重训：`--force_retrain`；仅补 eval：`--eval_only_from_checkpoint <bundle 或 snapshot 路径>`。

---

## 评估改造（先做这一步）

当前结果方法间差异 ≤0.005 macro_acc，**这低于 50-batch eval 的统计噪声**。v4 必须先把评估稳住，否则后面所有结论都不可信：

1. `**eval_max_batches` 从 50 提到 200**（每域每客户端），开销约 4×。或者只跑 1 个 anchor client 评估全域（标准 fedsa 风格，省 35×）。
2. **加 3 种 seed 重复**（`seed ∈ {42, 1234, 9999}`），均值 ± std 报告。（这个最后再做）
3. 加 **bootstrap 置信区间**：在 50/200 batch 上做 1000× resample，给 95% CI。
4. 加 **per-domain per-client matrix** 输出（35×7 的 acc / ppl 矩阵），方便事后做诊断。

新增脚本：[utilities/v4/eval_robust.py](utilities/v4/eval_robust.py)（待实现），[scripts/Analysis/summarize_v4.py](scripts/Analysis/summarize_v4.py)（已实现）。

---

## 推进顺序


| 阶段      | 周期    | 内容                                                           |
| ------- | ----- | ------------------------------------------------------------ |
| Stage 0 | 1–2 天 | 评估稳定化（§5）；复跑 baseline-v2 + baseline-fedalt 3 seeds 拿基线均值±std |
| Stage 1 | 3–4 天 | Branch A 三组 (A1/A2/A3) —— 直接修 v3 退化                          |
| Stage 2 | 3–5 天 | Branch C 两组 (C1/C2) —— 仅本地改动，验证 YOCO 思路                      |
| Stage 3 | 4–7 天 | Branch D 三组 (D1/D2/D3) —— 个性化潜力最大                            |
| Stage 4 | 5–7 天 | Branch B / F —— 重参数化与异构 rank                                 |
| Stage 5 | 视效果   | Branch E —— 引入 anchor 假设是论文 trade-off                        |


主线推荐：先 A → C → D，三周内可以判定 v4 的故事是 Hier++、Sign+ 还是 Mix。

---

## 备注

- **不要** 一上来就把所有支线打开。每条支线先在 7c 快速 pilot（每 run < 30 min）验证机制再上 35c 主实验。
- **不要** 删除 v2 代码 —— v4 全部通过新文件 + 新入口接入。
- **不要** 在 v4 阶段就追求新 baseline；先把 v2 严格超过再拓展。

