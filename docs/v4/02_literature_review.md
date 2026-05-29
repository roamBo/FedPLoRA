# 文献综述：跨�?One-shot Personalized FedLoRA

> 走查 PaperLibarary 24 �?+ 网搜 2025�?026 �?12 篇，按照"�?v4 设计的可借鉴�?评分排序�?
---

## A. 一类：A-only / 选择性聚合（v4 直接对话�?
### FedSA-LoRA (Guo et al., ICLR 2025)
- **核心**：A 学通用、B 学客户端特异，因此只聚合 A、B 不上传�?- **v4 借鉴**：保�?v2/v3 �?A-only 范式，所�?6 条支线都遵守�?- **链接**：https://openreview.net/forum?id=iX3uESGdsO , GitHub: https://github.com/Pengxin-Guo/FedSA-LoRA

### FedRPCA (Jhunjhunwala et al., 2025)
- **核心**：把 LoRA 客户端更新分解成低秩公共 + 稀疏私有，分别用不�?scaling 平均。报�?+15% 优于 FedAvg-LoRA�?- **v4 借鉴**：Branch A �?`R_i = A_i �?A_0` �?SVD �?top-1 当公共，残余作为簇内信号 —�?�?v3-rpca 思路一致但 *不再压缩 80% 的稀�?row*�?- **链接**：https://arxiv.org/abs/2506.01194

### FFA-LoRA (Sun et al., ICLR 2024)
- **核心**：冻 A、只�?B，DP-SGD 下噪声不�?BA 的乘法放大�?- **v4 借鉴**：对照实验，�?Branch B (SVD) 形成一个矩阵：A �?vs A 正交聚合�?- **链接**：https://arxiv.org/abs/2403.12313

### FedSVD (Lee et al., NeurIPS 2025)
- **核心**：服务端 SVD 重分�?BA 乘积，重新给出正�?A 和精�?B；每轮通信只发 B�?- **v4 借鉴**：Branch B 的核心。把 `A_0` 重参数化为正交基，让 B 更新方向不会�?A 的非正交性扭曲�?- **链接**：https://arxiv.org/abs/2505.12805

---

## B. 一类：One-shot / 单轮通信

### YOCO (Xu et al., NeurIPS 2025)
- **核心**：one-shot 关键不是"减少通信"，而是 *initial* 全局监督就够了。本地训练加 sign-regularized B（强�?B �?B_0 同号�? sparse-regularized A，无需 aggregated 监督�?- **v4 借鉴**：Branch C 直接落地 Bsign + sparseA local regularizer�?*完全复用 v2 服务�?*�?- **链接**：https://openreview.net/forum?id=FoVF3iL6o3

### FedSD2C (Wu et al., NeurIPS 2024)
- **核心**：服务端�?Autoencoder 合成 distillates 做后训练�?.6× 优于其他 one-shot baseline�?- **v4 借鉴**：Branch E 用更轻的 anchor set（无合成）做超参选择。完整合成蒸馏暂留作论文 V5 扩展�?- **链接**：https://arxiv.org/abs/2412.05186

### FOL (Federated Oriented Learning, ICML 2025)
- **核心**：one-shot PFL 四阶段：pretrain �?collect �?align (fine-tune, prune, ensemble) �?distill�?- **v4 借鉴**：alignment 阶段�?ensemble refinement �?Branch D 提供启发（多�?A 做集成再蒸馏到客户端）�?- **链接**：https://icml.cc/virtual/2025/poster/44279

### Towards One-shot FL Survey (Wang et al., 2025)
- **核心**：综述，分类 OSFL �?ensemble-based / synthetic-data / direct-aggregation 三大类�?- **v4 借鉴**：定�?—�?v4 主线属于 direct-aggregation + dual-adapter（Branch D），辅以 anchor calibration（E）�?- **链接**：https://arxiv.org/abs/2505.02426

---

## C. 一类：个性化 / 双适配�?
### FedDPA (Yang et al., NeurIPS 2024)
- **核心**：每客户端两�?adapter（global + local），�?*instance-wise dynamic weighting* 在推理时融合，handle test-time distribution shift�?- **v4 借鉴**：Branch D �?per_input MoE mixer。但 v4 不引�?*额外* 一�?LoRA 参数 —�?A_local 是本地训练的产物，存储成本但通信成本不变�?- **链接**：https://arxiv.org/abs/2403.19211

### FedALT (Bian et al., 2025)
- **核心**：抛�?FedAvg �?先平均再 init"，每客户端继续训自己 LoRA + 一�?RoTW (Rest-of-the-World) LoRA 表示外部，MoE adaptive mixer�?- **v4 借鉴**：Branch D �?*rest-of-world* 解释 —�?`A_down` 就是 RoTW，`A_local` �?self�?- **链接**：https://arxiv.org/abs/2503.11880

### FDLoRA (2024)
- **核心**：双 LoRA（global + local�? adaptive fusion�?- **v4 借鉴**：Branch D 的另一�?mixer 实现（learned scalar per layer）�?- **链接**：https://arxiv.org/abs/2406.07925

### pFedLoRA (2023)
- **核心**：异�?rank + 本地 LoRA + 全局 LoRA 蒸馏�?- **v4 借鉴**：Branch F 异构 rank 的工程参考；不是直接对话方法�?- **链接**：https://arxiv.org/abs/2310.13283

---

## D. 一类：异构 / 容量分配

### HetLoRA (Cho et al., EMNLP 2024)
- **核心**：客户端可用不同 rank，用 Frobenius-norm 加权 FedAvg�?- **v4 借鉴**：Branch F �?base 算法；按域而非按客户端分配 rank�?- **链接**：https://arxiv.org/abs/2401.06432

### AdaLoRA (Zhang et al., ICLR 2023)
- **核心**：SVD 参数�?+ 重要性打分动态裁�?rank�?- **v4 借鉴**：Branch F �?server-side rank 决策可参考它�?importance 准则�?- **链接**：https://arxiv.org/abs/2303.10512

### FedHL (2025)
- **核心**：full-rank model �?unbiased aggregation，避�?truncation bias�?- **v4 借鉴**：Branch F 的对照，验证 pad-to-max 是否引入 bias�?- **链接**：https://arxiv.org/abs/2505.18494

### FedLoDrop (2024)
- **核心**：LoRA 上做 dropout 提升泛化�?- **v4 借鉴**：可�?row-dropout �?Branch F 的副实验�?- **链接**：见 PaperLibarary

---

## E. 一类：冲突 / 对齐

### HFLoRA (2025)
- **核心**：fine-grained joint conflict regulation + global LoRA consistent re-decomposition�?- **v4 借鉴**：Branch A �?row-level "异常抑制" 因子�?- **链接**：参�?https://openreview.net/pdf/6c0549998b3c3adebe138efc18212696b8c2c78c.pdf

### FedRot-LoRA (2025)
- **核心**：聚合前�?Procrustes / orthogonal transformation 对齐客户�?A�?- **v4 借鉴**：Branch B 的初始正交化 + 聚合前对齐�?- **链接**：https://arxiv.org/pdf/2602.23638

### FedGaLore (2025)
- **核心**：用 gradient-subspace 替代固定 LoRA 子空间，避免 update-space 失配�?- **v4 借鉴**：Branch F 的进�?—�?�?rank 异构，则梯度投影�?client-specific 子空间�?- **链接**：见 arxiv 2602.01746

### LoRA-A2 (ACL 2025)
- **核心**：偶数轮聚合 B、奇数轮聚合 A �?alternating�?- **v4 借鉴**：one-shot 设定下不适用，但启发 Branch C —�?�?B 学得更稳后再聚合 A�?- **链接**：见 PaperLibarary

---

## F. 一类：合成数据 / 蒸馏 / Anchor

### DENSE (2021)
- **核心**：data-free OSFL via synthetic generation�?- **v4 借鉴**：Branch E 的合成版本（V5 候选）�?
### OSCAR (2025)
- **核心**：one-shot via classifier-free diffusion�?- **v4 借鉴**：可能太重，不放主线�?- **链接**：https://arxiv.org/abs/2502.08488

### FedHydra (2024)
- **核心**：解�?one-shot FL 的多种异构性�?- **链接**：https://arxiv.org/abs/2410.21119

---

## G. 一类：MoE / LoRA composition

### LoraHub (Huang et al., COLM 2024)
- **核心**：多�?LoRA 通过 black-box optimization 组合权重做零样本任务适配�?- **v4 借鉴**：Branch E �?anchor-driven coefficient search�?- **链接**：https://arxiv.org/abs/2307.13269

### FedAMoLE (2024)
- **核心**：MoE LoRA + shape-invariant router for heterogeneous experts�?- **v4 借鉴**：Branch D �?per-input mixer 设计参考�?
### Mixture of LoRA Experts (各种)
- **借鉴**：Branch D mixer 设计；但 v4 不要引入额外 LoRA 专家，只融合 A_local �?A_down 两份�?
---

## H. v4 �?为什么不"

- **不做** 完整 federated MoE：通信和工程成本太高，不符�?one-shot 叙事�?- **不做** 合成数据蒸馏（FedSD2C 全套）：增加额外假设、上游模型依赖，留给 V5�?- **不做** 客户�?rank 全部异构（HetLoRA full version）：�?*�? 而非�?*客户�? 异构，可减少配置组合爆炸（Branch F）�?- **不做** DP-SGD / privacy 强保证：v4 论文走效用方向；privacy 留给独立工作�?