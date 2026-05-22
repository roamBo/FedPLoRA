# Branch A — FedPLoRA-Hier++（修复 v3 退化的硬修复版）

## A.0 一句话定位

> 在 v3 的残差 + 簇路径基础上，**去掉 quantile 硬阈、改 soft sigmoid blending、加 spectral 数据驱动簇，并修复评估时 cluster A_down 没正确加载的实现缺陷**，目标是让残差分支真正服务于跨域个性化而不是"丢弃 20% row"。

## A.1 假设与预期

| 假设 | 检验方式 |
|---|---|
| H-A.1 *soft gate* 比 *hard quantile gate* 表现更好（不丢信息） | A1 vs v3-cluster |
| H-A.2 *spectral cluster* 比 *先验 3 簇* 表现更好（数据驱动） | A2 vs A1 |
| H-A.3 评估时严格按 client_id 加载 personalized A 能恢复 v3 损失的个性化粒度 | A3 vs v3-cluster |
| H-A.4 Hier++ 在 Macro 上 ≥ v2，在 Worst 上 ≥ v2 | A1/A2/A3 vs v2 |

预期收益（基于 v2 base 0.69595 macro_acc）：

- A1 soft prior cluster：**+0.000 ~ +0.003**（修 v3，但因 cluster 仍为 3 簇，提升有限）
- A2 spectral K=5：**+0.002 ~ +0.005**（数据驱动簇适配 7 域）
- A3 修 eval：**与 A1 持平或略好**（验证机制本身）

## A.2 与 v3 的差异

| 维度 | v3 (`fedplora_oneshot.py:_aggregate_v3`) | v4 Branch A |
|---|---|---|
| gate 形式 | `gate = where(conflict > threshold, sigmoid((c−τ)/T), 0)` | `gate = sigmoid((c − μ_c) / (σ_c · κ))`，无硬阈 |
| 阈值来源 | layer-wise quantile q=0.80 | 标准化后的连续 sigmoid，温度 κ ∈ [0.5, 2] 可调 |
| 簇划分 | 固定 prior 3 簇 | `prior` / `spectral_k` / `kmeans_k` / `none` 多选 |
| 簇个性化范围 | residual 簇均 | residual 簇均 + 行内 row-importance 加权 |
| 评估时 personalized A | 加载 ✓，但 default 路径用簇均 | 加载 ✓，且 default 路径用 *unweighted-cluster* 全平均，避免和簇路径同质 |
| `lambda` 范围 | `[0.2, 1.0]` linear in g | `lambda(g) = lambda_min + (lambda_max − lambda_min) · g^p`, p ∈ {0.5, 1, 2} |

## A.3 算法细节（伪代码）

```python
def aggregate_models_v4_hier(global_model, client_uploads, args):
    A0 = args._fedplora_initial_A
    R = {key: [A_i[key] - A0[key] for i in clients]
         for key in lora_a_keys}

    # 1. row-level conflict on residual space, no hard threshold
    for key in lora_a_keys:
        R_norm = [||R_i||_row for i]
        weights = sample_size * importance * R_norm   # per-row
        R_dir = R_i / ||R_i||_row
        mean_dir = sum(weights * R_dir) / sum(weights)
        consensus = ||mean_dir||
        conflict = 1 - consensus                     # in [0, 1]

        # SOFT sigmoid gate, no quantile cutoff
        c_mean, c_std = mean(conflict), std(conflict)
        z = (conflict - c_mean) / (c_std + eps)      # standardize
        gate = sigmoid(z / kappa)                     # kappa = temperature

        # 2. cluster discovery
        if args.v4_cluster_mode == "spectral":
            clusters = spectral_cluster(R, K=args.v4_cluster_k)
        elif args.v4_cluster_mode == "kmeans":
            clusters = kmeans(R, K=args.v4_cluster_k)
        elif args.v4_cluster_mode == "prior":
            clusters = _DOMAIN_PRIOR_CLUSTERS
        else:
            clusters = {c: [i] for c in clients}      # no cluster

        # 3. blend
        R_common = weighted_mean(R, weights)          # all clients
        R_cluster[k] = weighted_mean(R[in_cluster_k])
        lambda_g = lambda_min + (lambda_max - lambda_min) * gate ** p

        # Default path: ALL clients use unweighted full mean
        default_A = A0 + (1 - gate) * R_common + lambda_g * gate * weighted_mean(R, uniform_weights)

        # Per-client personalized path
        for i in clients:
            personalized_A[i][key] = A0 + (1 - gate) * R_common \
                                         + lambda_g * gate * R_cluster[cluster(i)]

    return personalized_A, default_A
```

关键点：

- **`gate` 始终连续**，conflict 越高 gate 越大，但不强制 0 / 1。
- **`spectral_cluster`** 直接对 `[vec(R_i)]` 矩阵做 (N×D) → SVD top-K → K-Means，K∈{3,5,7}。
- **`default_A`** 不再使用簇均，避免与个性化路径同质化（这是 v3 的 bug）。
- **per-client A_down** 严格按 `client_id` 写入 `personalized_states`，evaluation 时正确加载。

## A.4 超参建议

| 参数 | A1 (prior) | A2 (spectral K=5) | A3 (prior + eval fix) |
|---|---|---|---|
| `--v4_gate_kappa` | 1.0 | 1.0 | 1.0 |
| `--v4_gate_power` | 1.0 | 1.0 | 1.0 |
| `--v4_cluster_mode` | `prior` | `spectral` | `prior` |
| `--v4_cluster_k` | 3 | 5 | 3 |
| `--v4_lambda_min` | 0.3 | 0.3 | 0.3 |
| `--v4_lambda_max` | 0.9 | 0.9 | 0.9 |
| `--v4_personalized_eval` | 1 | 1 | 1 |
| `--v4_default_uniform` | 1 | 1 | 1 |

## A.5 实现入口

- 服务端聚合：[methods/fedplora_v4_hier.py](../methods/fedplora_v4_hier.py)
- 客户端 local 训练：与 v2 fedplora-oneshot 完全一致（复用 `_add_fedplora_oneshot_anchor`）
- 训练入口：[tasks/fed_train_sft_v4.py](../tasks/fed_train_sft_v4.py) 中 `agg_type ∈ {v4_hier_soft_prior, v4_hier_soft_spectral, v4_hier_soft_pfl_eval}`
- 运行脚本：[scripts/RunScripts/run_v4_branch_a.sh](../scripts/RunScripts/run_v4_branch_a.sh)

## A.6 风险

1. spectral cluster 在 N=35 客户端、D=O(10M) 的客户端残差上要内存约 35·40MB ≈ 1.4 GB，**先在一层做 SVD 降维**到 100 维再聚类。
2. soft gate 可能让 gate 始终 ≈0.5（中间值），如果是这种情况下，加 `--v4_gate_power=2` 让 gate 更"两极化"。
3. spectral cluster K 选错会损害效果，A2 的 K∈{3,5,7} 三档先做 pilot 选最优。
