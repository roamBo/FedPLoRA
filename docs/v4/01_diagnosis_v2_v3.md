# v2 / v3 现状深度诊断

本文�?`Result/` 中的 6 �?JSON + 6 �?log 出发，回答三个问题：

1. v2 为什么比 v3 好？
2. v3 �?高冲突门�?为什么反而拉胯了�?3. 哪些指标的提升空间最大、最值得 v4 重点解决�?
---

## 1. 结果汇总（35c, r=1, e=1, seed=42, Meta-Llama-3.1-8B�?
| 方法 | macro_acc | worst_acc | hard_avg_loss | macro_ppl | comm_down/up (MB) |
|---|---:|---:|---:|---:|---:|
| `fedalt`          | 0.69331 | 0.51739 | 1.831 | 4.278 | 80 / 80 |
| `fedplora-oneshot` (v2) | **0.69595** | **0.51982** | **1.774** | **4.163** | 38 / 38 |
| `fedplora_v3_lite` | 0.69278 | 0.51460 | 1.838 | 4.352 | 38 / 38 |
| `fedplora_v3_cluster` | 0.69272 | 0.51462 | 1.834 | 4.355 | 38 / 38 |
| `fedplora_v3_rpca` | 0.69068 | 0.51240 | 1.882 | 4.512 | 38 / 38 |
| `yoco` (PCWA)     | 0.69490 | 0.51946 | 1.787 | 4.206 | 16144 / 80 |

> `hard_avg_loss = mean({legal, finance, medical} loss)`，由 [fed_train_sft.py:_domain_loss_diagnostics](../../tasks/fed_train_sft.py) 计算�?
## 2. 每域细分（v2 vs 最�?v3�?
| Domain | n_train/client | v2 acc | v2 ppl | v3-lite acc | v3-lite ppl | Δacc (v3 vs v2) |
|---|---:|---:|---:|---:|---:|---:|
| code      | 720 | 0.8153 | 2.03 | 0.8113 | 2.10 | **-0.0040** |
| education |  70 | 0.7803 | 2.30 | 0.7810 | 2.36 | +0.0007 |
| finance   | 720 | 0.5198 | 6.01 | 0.5146 | 6.40 | -0.0052 |
| general   | 720 | 0.7528 | 2.93 | 0.7503 | 3.03 | -0.0025 |
| legal     | 720 | 0.5240 | 10.45 | 0.5210 | 10.86 | -0.0030 |
| math      | 720 | 0.7993 | 2.08 | 0.7967 | 2.16 | -0.0026 |
| medical   | 288 | 0.6802 | 3.34 | 0.6746 | 3.55 | -0.0056 |

**观察**：v3 �?6/7 个域都下降，唯一上升�?education 只多 70 训练样本/客户�?—�?说明 v3 的损害是系统性的，不是某一类域的退化�?
## 3. 冲突统计对比

来自 log �?`_summary` 字段�?
| 方法 | mean_conflict | max_conflict | high_row_frac | mean_gate / init_gate |
|---|---:|---:|---:|---:|
| `fedplora-oneshot` (v2) | 0.0430 | 0.1154 | 0.000 | 0.000 (init_gate) |
| `fedplora_v3_lite` | 0.6974 | ~0.95+ | 0.2500 | 0.2500 (mean_gate) |
| `fedplora_v3_cluster` | 0.6974 | ~0.95+ | 0.2500 | 0.2500 |
| `fedplora_v3_rpca` | 0.6974 | ~0.95+ | 0.2500 | 0.2500 |

**关键解读**�?
- v2 �?*原始 A 空间* 测冲�?�?几乎�?0（A_i 几乎都和 A_0 同方向）�?gate 不触�?�?实际�?*锚定 A_0 的加�?A 平均*。这�?v2 优于 FedSA-LoRA 的原因（FedSA-LoRA 是无锚的纯平均）�?- v3 �?*残差 ΔA 空间* 测冲�?�?mean=0.70，high_row_frac=0.25�?*但这并非"v3 真的发现�?25% 的冲�?row"**，而是 quantile-based 阈�?`q=0.80` 在分位数定义下保证了 ~20% row 被门控触�?—�?这是定义意义上的循环，与是否真的冲突无关�?- 0.25 �?row 被乘�?`(1−g)` 后压�?0（保�?`A_0`）�?*实质等价于把 25% 的客户端更新方向直接丢弃**。在 7 �?35 客户端�? epoch 的小更新下，能学到的领域信号本就不多，丢一半就够受了�?
## 4. v3 �?4 个具体设计缺�?
�?[fedplora_oneshot.py:_aggregate_v3](../../methods/fedplora_oneshot.py) 后定位的代码层面问题�?
1. **quantile 阈值的循环依赖**（[fedplora_oneshot.py:239](../../methods/fedplora_oneshot.py:239)）：
   ```python
   threshold = _safe_quantile(conflict[active], q, default=1.0)
   ```
   不管真实冲突分布如何，q=0.80 都保证有 20% row 被标记高冲突。这不是"自适应"，是"按比例丢�?�?
2. **gate 硬阈过滤**（[fedplora_oneshot.py:241](../../methods/fedplora_oneshot.py:241)）：
   ```python
   gate = torch.where(conflict > threshold, raw_gate, torch.zeros_like(raw_gate))
   ```
   conflict �?threshold �?gate = 0，conflict > threshold �?gate �?(0, 1]。这意味着�?   - 低冲�?row 完全�?`R_common`（聚合所有客户端）—�?**OK**�?   - 高冲�?row �?`(1 �?g) · R_common + λ · g · R_cluster`，其�?`λ �?[0.2, 1.0]`。当 g=1, λ=0.2 时，高冲�?row 的更新强度只有低冲突�?20%�?*这是过度抑制**�?
3. **cluster 划分太粗**（[fedplora_oneshot.py:21-29](../../methods/fedplora_oneshot.py:21)）：
   ```python
   _DOMAIN_PRIOR_CLUSTERS = {
       "general": "general_education", "education": "general_education",
       "math": "capability", "code": "capability",
       "medical": "risk", "legal": "risk", "finance": "risk",
   }
   ```
   3 个簇覆盖 7 个域，risk 簇里 medical/legal/finance 文体差距巨大（医学英�?vs 法律拉丁词根 vs 金融数字密集）—�?强制共享�?residual 等于另一种平均，损害了原�?v2 的逐客户端加权细粒度�?
4. **personalized �?default 路径覆盖�?global_dict**（[fedplora_oneshot.py:405-417](../../methods/fedplora_oneshot.py:405)）：
   ```python
   global_dict[key] = default_candidate.to(...)        # 405
   ...
   for idx, client_id in enumerate(client_ids):
       personalized_A = ref + common_path + lambda_gate * cluster_residual
       personalized_states[int(client_id)][key] = personalized_A.detach().cpu().clone()
   ```
   `default_candidate` �?*所有客户端* 的平�?cluster residual 而非"无簇"的整体平均，这其实是个第 4 个簇 —�?但在评估代码里，`personalized_shared_states[client_id]` 会覆�?`global_dict`（见 [fed_train_sft.py:520-524](../../tasks/fed_train_sft.py:520)）。所�?default path **没问�?*�?   
   真正的问题在于：**evaluation 时同一域内�?5 个客户端 *�? 用同一�?cluster_residual**，因�?cluster 内不同客户端�?A_down 实际相同。这意味着 v3-cluster 在评估上的有效个性化粒度还是 **3 个簇**，�?v2 �?�?row 加权"反而保留了 35 客户端的微分信息�?*这是 v3 退化的最隐蔽原因**�?
## 5. 评估稳定性问�?
| 项目 | 现�?| 问题 |
|---|---|---|
| `eval_max_batches` | 50 | batch_size=2 �?100 样本，方差大。差�?0.003 是不是真的？ |
| `seed` �?| 1 (=42) | 不能给出 ± std |
| domain 评测的客户端聚合 | 35 客户端均�?| 同域 5 客户端，*实际加载相同 A 和不�?B*，差异主要来�?B；这不等价于"5 个独�?model" |

**v4 必须解决�?* 评估稳定�?�?.001 �?std，差�?�?.005 才可信�?
## 6. v4 必须解决的优先级清单

| 优先�?| 问题 | v4 的回�?|
|---|---|---|
| **P0** | eval 不稳定，方法间差距落入噪�?| §5 评估改造（eval_max_batches=200, 3 seeds, bootstrap CI�?|
| **P0** | v3 残差 quantile 阈值的循环依赖 �?强行 20% 丢弃 | Branch A �?soft sigmoid，不依赖 quantile 阈�?|
| **P1** | v3 cluster 粒度 3，小�?v2 row-level 加权 | Branch A �?spectral clustering K∈{5, 7} 自动�?|
| **P1** | 35 客户�?1 epoch �?A 漂移幅度小（mean_conflict=0.04），单纯 A-only 难拉开差距 | Branch C/D �?B 侧用 sign 正则或保�?A_local 增加信号 |
| **P2** | 难域 legal/finance ppl=10/6 表明 A_0 表征对它们不友好 | Branch F 给难域分�?r=16；Branch B 用正�?A_0 提供更稳定基 |
| **P2** | personalized A �?evaluation 路径正确，但 cluster 内同质化 | Branch D 在客户端侧加 A_local 个性化，绕�?cluster |
| **P3** | 通信效率：v2/v3 已经 38MB，进一步压缩空间不�?| F 异构 rank 可探 Pareto，但不是 v4 主线 |

## 7. 评估改造的最低修�?PR

为后续支线对照稳定，先做这两件事（不属于支线，是基础设施）：

1. **新增 `--eval_seeds "42,1234,9999"` 参数**：在 `federated_sft` 入口处用 set_seed 切换 3 次后�?3 次完整训�?�?输出 mean±std�?2. **新增 `utilities/eval_robust.py`**：从 JSON 里提�?metrics，bootstrap 1000 次给�?95% CI�?3. **新增 per-domain × per-client 矩阵 dump**：JSON 多写一�?`per_client_per_domain` 字段�?5 × 7 表格，便于后续做相关性分析�?
注：这三件事可以作为 v4 �?P0 ground-work 一起放�?[tasks/fed_train_sft_v4.py](../tasks/fed_train_sft_v4.py)�?