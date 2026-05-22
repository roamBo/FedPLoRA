# Branch B — FedPLoRA-SVD（FedSVD 思路：服务端正交 A_0 + 聚合后 SVD 重分解）

## B.0 一句话定位

> **把初始 `A_0` 用 QR 正交化为 row-orthonormal 矩阵 (rows 是 d 维单位向量、互相正交)，聚合后对 stacked-A 做 SVD 重分解，让所有客户端的 A_down 都对齐到同一组主成分坐标系**。预期：B 的更新方向不被 A 的非正交性"扭曲"，跨客户端的 A 聚合更稳定。

## B.1 假设与预期

| 假设 | 检验方式 |
|---|---|
| H-B.1 仅做 A_0 初始正交化就能让 A 聚合更稳定 | B1 vs v2 |
| H-B.2 聚合后 SVD 重分解能进一步提升 | B2 vs B1 |
| H-B.3 SVD 后的 A 与本地训得 B 仍兼容（核心风险） | B2 看是否退化 |

预期收益：

- B1 A_0 正交：**+0.000 ~ +0.003**（小提升）
- B2 SVD 重分解：**+0.002 ~ +0.006**（中等提升，但有崩溃风险）

## B.2 设计要点

### B.2.1 A_0 初始正交化

PEFT 默认 LoRA A 用 Kaiming uniform 初始化，**不是正交**。改动：

```python
# In utilities/models.py or new utilities/v4_orth_init.py
def orthogonalize_lora_A(model, eps=1e-6):
    for name, p in model.named_parameters():
        if "lora_A" in name and name.endswith("default.weight"):
            # p shape: (r, d), r << d
            with torch.no_grad():
                # QR on transpose: (d, r), pad if necessary
                pt = p.transpose(0, 1).float()
                q, _ = torch.linalg.qr(pt, mode="reduced")
                # q shape: (d, r)
                p.copy_(q.transpose(0, 1).to(dtype=p.dtype) * (p.norm() / (q.norm() + eps)))
```

正交化后 row-norm 保持原值（避免改变 LoRA scaling）。

### B.2.2 聚合后 SVD 重分解（FedSVD-style）

```python
def aggregate_models_v4_svd(global_model, client_uploads, args, *, do_refactor=True):
    global_dict = global_model.state_dict()
    weights = _client_weights(client_uploads, args)

    for key in lora_a_keys:
        # stack: (N, r, d) → (N*r, d)
        mats = [client_states[i][key].float() for i in range(N)]
        stacked = torch.cat(mats, dim=0)        # (N*r, d)

        if do_refactor:
            # SVD on stacked → top-r right singular vectors
            U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
            # Take top-r rows of Vh as new A (row-orthonormal)
            A_new = Vh[:r]                       # (r, d)
            # Sign disambiguation: align with weighted mean of client A
            mean_A = sum(weights[i] * mats[i] for i in range(N))
            signs = torch.sign((A_new * mean_A).sum(dim=1, keepdim=True))
            A_new = A_new * signs
            # Scale: preserve weighted-mean Frobenius norm
            target_norm = torch.linalg.matrix_norm(mean_A)
            A_new = A_new * (target_norm / torch.linalg.matrix_norm(A_new) + eps)
        else:
            # B1: only orth init, then standard FedAvg
            A_new = sum(weights[i] * mats[i] for i in range(N))

        global_dict[key] = A_new
    ...
```

### B.2.3 B 兼容性问题

SVD 后的新 `A_new` 行方向可能与客户端的 `B_i` 训练时假设的 A 行方向不一致 —— `B_i` 在原 A 列上学到了某些 row 系数，SVD 后行被重组，`B_i` 的列对应关系被打乱。

**两种缓解**：

1. **Procrustes 对齐**（FedRot-LoRA 思路）：找 `R = argmin ||R · A_new − A_mean||_F` s.t. `R^T R = I`，然后下发 `R · A_new`。这样新 A 既保留 SVD 的正交性又最大程度对齐客户端历史方向。

2. **客户端 first-batch 快速校准**：下发后客户端在 forward 时先固定 A_new 做 1 个 batch 的 B-only gradient step，再开始评估。但这破坏了 "no extra round" 叙事，仅作 fallback。

## B.3 超参建议

| 参数 | B1 | B2 |
|---|---|---|
| `--v4_svd_orth_init` | 1 | 1 |
| `--v4_svd_refactor` | 0 | 1 |
| `--v4_svd_procrustes` | 0 | 1 |
| `--v4_svd_rank` | (= lora_r) | (= lora_r) |

## B.4 实现入口

- 初始化改造：[utilities/v4_orth_init.py](../utilities/v4_orth_init.py)
- 服务端聚合：[methods/fedplora_v4_svd.py](../methods/fedplora_v4_svd.py)
- 训练入口：`agg_type ∈ {v4_svd_orth_only, v4_svd_full}`
- 运行脚本：[scripts/RunScripts/run_v4_branch_b.sh](../scripts/RunScripts/run_v4_branch_b.sh)

## B.5 风险

1. **正交 A 影响 LoRA scaling**：PEFT 内部 `lora_alpha / lora_r` 假设 A 行 norm 是 *某个* 经验值；强制正交化后行 norm 可能偏离。需要 scale 后再写回。
2. **SVD 重分解破坏 B-A 兼容**：B 是 v2 的关键私有量，A 全部换主成分坐标后 B 可能"失效"。Procrustes 是缓解手段，但有上限。
3. **数值稳定**：N·r ≥ d 时 SVD 退化；Llama-3.1-8B q_proj 的 d=4096，N·r = 35×8 = 280 << 4096，安全。但若 r=16 或客户端数增加，需要换 randomized SVD。
