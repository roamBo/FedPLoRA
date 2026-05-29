# Branch C �?FedPLoRA-Sign（YOCO Bsign + sparse-A 本地正则�?
## C.0 一句话定位

> **完全不改服务端聚�?*，只在客户端本地训练阶段加两个正则项�?i) `||tanh(γ B_i) �?sign(B_0)||_1` �?B 的符号与初始保持一致；(ii) `||A_i||_1` �?A 稀疏。预期通过 *initial 全局监督* 替代 *aggregated 全局监督*，让 A 聚合空间天然兼容多客户端 B —�?这是 YOCO (NeurIPS 2025) 的核心思想�?
## C.1 假设与预�?
| 假设 | 检验方�?|
|---|---|
| H-C.1 仅加 Bsign 正则就能�?A_i 在聚合后更稳定（无需改聚合规则） | C1 vs v2 |
| H-C.2 Bsign + sparseA 比单�?Bsign 更优（Y0CO 完整版） | C2 vs C1 |
| H-C.3 C 系列�?Branch A �?*正交* 的（可叠加） | C2 + A2 联合试验 |

预期收益（基�?v2 base 0.69595 macro_acc）：

- C1 Bsign only�?*+0.001 ~ +0.003**
- C2 Bsign + sparseA�?*+0.002 ~ +0.006**
- C2 + A2 联合�?*+0.003 ~ +0.010**（最有希望，但风险也高）

## C.2 �?v2 的差�?
| 维度 | v2 | v4 Branch C |
|---|---|---|
| 服务端聚�?| `aggregate_models_fedplora_oneshot` | **完全不变** |
| 客户�?local 训练 | base + `_add_fedplora_oneshot_anchor` (A �? | base + Bsign + sparseA + (可�? A �?|
| 通信 | 38 / 38 MB | 38 / 38 MB（完全相同） |

## C.3 正则�?
### C.3.1 B 方向正则（核心）

YOCO 论文里的形式�?
```python
def _add_bsign_regularizer(loss, model, args):
    initial_B = getattr(args, "_v4_initial_B", None)
    if not initial_B:
        return loss
    lam = float(getattr(args, "v4_bsign_lambda", 1e-3))
    gamma = float(getattr(args, "v4_bsign_gamma", 5.0))
    if lam <= 0:
        return loss
    terms = []
    for key, B_local in model.named_parameters():
        if "lora_B" not in key or not key.endswith("default.weight"):
            continue
        if not B_local.requires_grad:
            continue
        B0 = initial_B.get(key, None)
        if B0 is None:
            continue
        B0 = B0.to(device=B_local.device, dtype=B_local.dtype)
        target = torch.sign(B0)                                  # ±1 mask
        soft = torch.tanh(gamma * B_local.float())               # �?[-1, 1]
        terms.append((soft - target).abs().mean())
    if terms:
        loss = loss + lam * torch.stack(terms).mean()
    return loss
```

直觉：B 的元�?*符号* �?B-A 乘法相对方向�?骨架"，符号变化会�?BA 完全反向，从而破坏与服务�?A 的相容性。强�?B 符号稳定比强�?B 数值稳定更"温和"�?
### C.3.2 A 稀疏正�?
```python
def _add_asparse_regularizer(loss, model, args):
    lam = float(getattr(args, "v4_asparse_lambda", 1e-4))
    if lam <= 0:
        return loss
    terms = []
    for key, A_local in model.named_parameters():
        if "lora_A" not in key or not key.endswith("default.weight"):
            continue
        if not A_local.requires_grad:
            continue
        terms.append(A_local.abs().mean())
    if terms:
        loss = loss + lam * torch.stack(terms).mean()
    return loss
```

注意 v2 已有 `--yoco_sparse_lambda` 参数（[fed_train_sft.py:139-143](../../../../tasks/fed_train_sft.py:139)），实现位置 [train_eval.py:_add_yoco_sparse](../../../../utilities/train_eval.py:144) —�?C2 直接复用，不重复实现�?
### C.3.3 �?v2 `_add_fedplora_oneshot_anchor` 的关�?
v2 已经�?A 方向锚定：`1 �?|cos(A_i_row, A_0_row)|`。Branch C �?Bsign �?*互补* 的（A 锚约�?A 的方向，Bsign 约束 B 的符号），可以同时开�?
## C.4 超参建议

| 参数 | C1 | C2 |
|---|---|---|
| `--v4_bsign_lambda` | 1e-3 | 1e-3 |
| `--v4_bsign_gamma` | 5.0 | 5.0 |
| `--v4_asparse_lambda` (= `--yoco_sparse_lambda`) | 0 | 1e-4 |
| `--oneshot_anchor_lambda` (复用 v2) | 1e-4 | 1e-4 |
| `--oneshot_prox_lambda` | 0 | 0 |

C1 关掉 sparseA，单独测 Bsign；C2 开 sparseA，对齐完�?YOCO 配方�?
## C.5 实现入口

- 服务端聚合：复用 v2 `aggregate_models_fedplora_oneshot`�?*完全不改**�?- 客户�?local hook：[methods/fedplora_v4_sign.py](../../../methods/v4/fedplora_v4_sign.py) 里实�?`_add_bsign_regularizer`�?- 训练入口：`agg_type �?{v4_sign_v2agg, v4_sign_full}`，本质都�?fedplora-oneshot + 额外正则项�?- 运行脚本：[scripts/RunScripts/run_v4_branch_c.sh](../../../scripts/RunScripts/run_v4_branch_c.sh)�?
## C.6 风险

1. `tanh(γ B)` �?γ=5 时已近饱和，γ 太大会让 gradient vanish；�?太小�?Bsign 约束太弱。先固定 γ=5，C1 跑完�?sweep�?2. Bsign 正则�?B 初始化方式紧耦合 —�?PEFT 默认 B 初始化为零矩阵，`sign(B_0) = 0`�?*这是关键问题**：sign(0) �?torch 中定义为 0，正则项退化为 `|tanh(γ B)|`，�?B �?0。这�?v4 Branch C 的实现红�?—�?*必须等本地训第一步后再快�?B_0**，或�?*very small random init* 替代零初始化。详见实现注释�?3. �?PEFT 不允许改 B 初始化，回退方案：在�?1 �?batch 后立刻快�?`B^(1)` 作为 anchor target，从�?2 �?batch 开始施加正则�?