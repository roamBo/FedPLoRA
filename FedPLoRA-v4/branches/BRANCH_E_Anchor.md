# Branch E — FedPLoRA-Anchor（FedSD2C / FOL 借鉴：服务端公共校准集）

## E.0 一句话定位

> **服务端持有一个小型公开 anchor 数据集（每域 50 条公开 instruction），聚合后用它在 *candidate λ / threshold / rank* 网格上选最优配置**。引入一个轻量"假设"（服务端有少量公共数据）换 +0.005 ~ +0.01 macro_acc 的潜在上限。是否纳入主论文取决于审稿人对该假设的接受度。

## E.1 假设与预期

| 假设 | 检验方式 |
|---|---|
| H-E.1 50 条/域 的 anchor 足以稳定选出最优 λ | E1 vs grid search 实验 |
| H-E.2 anchor-driven gate threshold > 固定 threshold | E1 vs A1 |
| H-E.3 anchor-driven cluster lambda > 固定 lambda | E2 vs A1 |
| H-E.4 加 anchor 不破坏 worst-domain 表现 | 全 branch 对照 |

预期收益（基于 Branch A best）：

- E1：**+0.002 ~ +0.005**
- E2：**+0.003 ~ +0.007**

## E.2 anchor 数据集设计

每域 50 条公开 instruction：

| 域 | 来源 |
|---|---|
| general | flan_v2 / dolly_v2_subset |
| math | gsm8k validation 50 |
| code | humaneval validation 50 |
| medical | medqa_en validation 50 |
| legal | lex_glue / ledgar small subset |
| finance | financial_phrasebank validation 50 |
| education | openschoolqa / arc_easy validation 50 |

放在 `data/anchor_seed/seed_42/anchor.jsonl`，350 条总样本。

## E.3 候选配置网格

```python
candidates = []
for lam_min in [0.1, 0.2, 0.3]:
    for lam_max in [0.7, 0.9, 1.0]:
        for kappa in [0.5, 1.0, 2.0]:
            for cluster_k in [3, 5, 7]:
                candidates.append((lam_min, lam_max, kappa, cluster_k))
```

= 81 个候选。每个候选评估 50 batches × 7 域 ≈ 350 forwards。在 Llama-3.1-8B 上单 forward ~ 80 ms，共 350 × 81 × 0.08 ≈ 38 min。可接受。

## E.4 算法

```python
def aggregate_v4_anchor(global_model, client_uploads, args, anchor_loaders):
    # 1. Compute base residuals
    R = compute_residuals(client_uploads)

    # 2. Sweep candidates
    best_config = None
    best_loss = float("inf")
    for config in candidates:
        A_down = build_A_down(R, config)
        loss = eval_on_anchor(global_model, A_down, anchor_loaders)
        if loss < best_loss:
            best_loss = loss
            best_config = config

    # 3. Apply best
    A_down = build_A_down(R, best_config)
    return A_down
```

## E.5 与 LoraHub 的差别

LoraHub 用 black-box optimization 在 *任务模块* 之间搜组合权重；E 是在 *超参* 之间搜组合，且只用 350 个公共样本。

## E.6 风险

1. anchor 数据质量：若 anchor 不代表客户端真实分布，选出的 config 是 anchor-optimal 但不是 client-optimal。需要做 sanity check：anchor loss 与 client val loss 的相关性 ≥ 0.5。
2. anchor data 引入审稿争议。建议在论文里给出两套结果：with-anchor 和 without-anchor，让审稿人选。
3. 如果 anchor 集 size > 50/域，会触发 "anchor 等于 mini-dataset" 的非 one-shot 担忧。**严格 cap 在 50/域**。

## E.7 实现入口

- Anchor data loader：[utilities/anchor_data.py](../utilities/anchor_data.py)
- 服务端候选搜索：[methods/fedplora_v4_anchor.py](../methods/fedplora_v4_anchor.py)
- 训练入口：`agg_type ∈ {v4_anchor_gate, v4_anchor_lambda, v4_anchor_full}`
- 运行脚本：[scripts/RunScripts/run_v4_branch_e.sh](../scripts/RunScripts/run_v4_branch_e.sh)
