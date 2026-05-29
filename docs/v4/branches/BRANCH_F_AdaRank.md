# Branch F �?FedPLoRA-AdaRank（HetLoRA / AdaLoRA 借鉴：按域异�?rank�?
## F.0 一句话定位

> **按域分配 LoRA rank：高风险专业域（medical/legal/finance）r=16，能力域（math/code）r=8，通用域（general/education）r=4**。聚合时 pad �?r_max=16，下发时按域 truncate，整体通信变化 ±25%，让高风险域获得更多容量、低难度域省通信�?
## F.1 假设与预�?
| 假设 | 检验方�?|
|---|---|
| H-F.1 高风险域分配更高 rank 能改�?Worst | F1 vs v2 |
| H-F.2 通用域降�?rank 不损�?Macro（通用知识可在�?rank 表达�?| F2 vs F1 |
| H-F.3 异构 rank 不破�?FedSA-LoRA �?A 共享假设 | F1/F2 看是否退�?|

预期收益（基�?v2 base 0.69595 macro_acc）：

- F1 risk-r16-only�?*Worst +0.003 ~ +0.008**，Macro 几乎不变
- F2 全异构：**Macro +0.001 ~ +0.003**，通信 -10%

## F.2 设计要点

### F.2.1 异构 rank 配置

```python
DOMAIN_RANK = {
    "general":   4,
    "education": 4,
    "math":      8,
    "code":      8,
    "medical":  16,
    "legal":    16,
    "finance":  16,
}
```

客户端初始化 LoRA 时：

```python
def create_peft_causal_lm_model_v4(args, client_domain):
    rank = DOMAIN_RANK.get(client_domain, args.lora_r)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,            # 保持 alpha/r 比例
        ...
    )
    ...
```

### F.2.2 聚合�?padding

```python
def aggregate_models_v4_adarank(global_model, client_uploads, args):
    R_MAX = 16
    for key in lora_a_keys:
        # Each client has A_i shape (r_i, d)
        padded = []
        masks = []
        for upload in client_uploads:
            A_i = upload["state_dict"][key]
            r_i = A_i.shape[0]
            pad = torch.zeros((R_MAX - r_i, A_i.shape[1]))
            padded.append(torch.cat([A_i, pad], dim=0))         # (R_MAX, d)
            mask = torch.cat([torch.ones(r_i), torch.zeros(R_MAX - r_i)])
            masks.append(mask)
        # Weighted average, normalized by per-row valid client count
        stacked = torch.stack(padded)                           # (N, R_MAX, d)
        mask_stack = torch.stack(masks).unsqueeze(-1)           # (N, R_MAX, 1)
        valid = mask_stack.sum(dim=0).clamp_min(1)              # (R_MAX, 1)
        A_global = (stacked * mask_stack).sum(dim=0) / valid    # (R_MAX, d)

        # Store padded A_global
        global_dict[key] = A_global

    # Downlink: truncate per client based on their domain
    for client_id, domain in args._fedplora_client_domains.items():
        r = DOMAIN_RANK[domain]
        A_down_client[client_id][key] = A_global[:r]            # (r, d)
```

### F.2.3 通信统计

- 客户�?c 的域 r 异构 �?A 上行字节 = `r · d · 4` (bf16: 2 bytes)
- 服务端聚合后 r_max=16，但下发时按客户�?r 截取
- 总下�?= `sum_c (r_c · d · 2)`，总上�?= `sum_c (r_c · d · 2)`

| 配置 | 总上�?| 总下�?|
|---|---:|---:|
| v2 (r=8) | 38.0 MB × 35 | 38.0 MB × 35 |
| F1 (risk r=16, other r=8) | 38.0 × 20 + 76.0 × 15 = 1900 MB | 同上 |
| F2 (gen/edu r=4, math/code r=8, risk r=16) | 19×10 + 38×10 + 76×15 = 1710 MB | 同上 |

平均每客户端 F1 �?54.3 MB，F2 �?48.9 MB —�?实际介于 v2 (38) �?yoco-style 之间�?
## F.3 �?HetLoRA 的差�?
| 维度 | HetLoRA | v4-AdaRank |
|---|---|---|
| Rank 分配粒度 | per-client (可任�? | per-domain (7 �? |
| 聚合方式 | Frobenius-weighted FedAvg | 加权平均 + zero padding mask |
| 通信 | 客户�?r_i | per-client r_domain(i) |

按域而非按客户端的好处：

- 调参组合数从 35^k 降到 7^k�?- �?v2/v3 �?cluster 概念天然对齐�?- 便于�?难域专属容量"的故事�?
## F.4 超参建议

| 参数 | F1 | F2 |
|---|---|---|
| `--v4_adarank_mode` | `risk_only` | `full` |
| `--v4_adarank_rank_map` | "medical:16,legal:16,finance:16" | "medical:16,legal:16,finance:16,math:8,code:8,general:4,education:4" |
| `--v4_adarank_pad_mode` | `zero` | `zero` |

## F.5 实现入口

- 模型构造：扩展 [utilities/models.py](../../../utilities/v4/models.py) �?`create_peft_causal_lm_model_v4_adarank`
- 服务端聚合：[methods/fedplora_v4_adarank.py](../../../methods/v4/fedplora_v4_adarank.py)
- 客户端本地训练：�?v2 一致（�?LoRA �?r 改变�?- 评估：每客户端按其域 r 加载对应 truncated A_down
- 运行脚本：[scripts/RunScripts/run_v4_branch_f.sh](../../../scripts/RunScripts/run_v4_branch_f.sh)

## F.6 风险

1. **alpha/r 比例对学习率敏感**：r 改变，alpha 应同比例改变，否�?LoRA scaling 偏移�?2. **PEFT API 不直接支持异�?rank 共聚�?*：需要写 padding 包装；本支线工作量最大�?3. **难域容量增加是否真有�?* 取决�?1 epoch 训练能不能填满更�?r —�?35c × 1 epoch �?finance 客户�?720 样本 / batch=2 / r=16 �?360 steps，每 step 更新 16×4096+4096×16 �?130K 参数，可能不够充分。建议先 e=2 �?pilot�?