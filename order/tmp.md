<!--
Source PDF: /Users/hawaiii/codex/FedPLoRA/paper/AAAI-2027/FedPLoRA_AAAI2027_20260726.pdf
Created: 2026-07-26
Purpose: PDF-to-Markdown transcription with globally consistent numeric citations and a verified reference list.
Citation policy: references are numbered by first appearance; repeated works retain the same number.
Verification policy: publication metadata was checked against primary publisher/proceedings pages or the authors' arXiv record on 2026-07-26.
Transcription note: the source PDF contains a mismatched Figure 2 caption about a backdoor-defense pipeline and an empty Conclusion section. Both are retained and explicitly marked below so that the Markdown remains faithful to the PDF rather than silently repairing it.
-->

# Domain Skills Live in LoRA-\(B\): One-Shot Federated Personalization by Factor-Geometric Routing

**Anonymous submission**

## Abstract

Federated fine-tuning of language models inherently involves trade-offs among three core objectives: adapting to heterogeneous domains, preserving local personalization, and maintaining strict communication efficiency. Existing federated LoRA methods often force a rigid choice between globally averaging a single adapter, which washes out domain-specific skills, and maintaining complex personalized or expert structures, which compromises the communication advantage of low-rank adaptation. We uncover a factor-level asymmetry: under shared initialization, LoRA-\(A\) subspaces remain broadly aligned across domains, whereas LoRA-\(B\) forms domain-organized blocks. Swapping \(B\) between compatible clients causes little degradation, while cross-domain replacement substantially harms accuracy, showing that \(B\) is too heterogeneous to average globally yet too structured to keep isolated. Motivated by this asymmetry, we propose FedPLoRA to *share what transfers and route what differs*. FedPLoRA is a one-shot personalized federated LoRA framework that aggregates a low-rank sketch of the \(A\) update and routes pooled \(B\) experts by principal-angle subspace affinity. The method requires neither domain labels, learned routers, nor additional communication rounds, and the routed expert bank naturally supports few-shot domain-label-free onboarding of unseen clients. Evaluated on benchmark cross-domain datasets, our framework achieves state-of-the-art in-domain average performance while consuming only 63% of the communication cost of full factor uploads. These results identify LoRA factor geometry as a practical interface for efficient communication and effective personalized federated adaptation.

![Motivation figure: LoRA-A and LoRA-B geometry](/Users/hawaiii/codex/FedPLoRA/paper/AAAI-2027/Introduction/fig1_insight_data_20260724.png)

**Figure 1: Client-specific does not imply client-private.** LoRA-\(A\) remains broadly aligned across clients, whereas LoRA-\(B\) exhibits domain-organized and behaviorally consequential geometry, motivating shared-\(A\) transfer and compatibility-aware \(B\) routing.

> This is an anonymized submission for review purposes only. Distribution, citation, or public sharing of this manuscript is strictly prohibited. Copyright and publication details will appear in the final version if accepted.

## Introduction

Collaborative adaptation of foundation models is increasingly required in settings where useful data are private, distributed, and domain-specific. Federated learning (FL) enables organizations to learn without centralizing raw records, and its practical motivation and unresolved systems, statistical, and privacy challenges are well established in the federated-learning literature [1,2 all]. Multi-institutional healthcare studies further illustrate why useful data often cannot be pooled directly [3,4 all]. Low-Rank Adaptation (LoRA) reduces the trainable and communicated state of a foundation model to a small pair of factors per adapted layer [5 all]. More broadly, parameter-efficient fine-tuning spans compact adapters, continuous prompts, activation scaling, adaptive rank allocation, quantization-aware adaptation, and weight-decomposed low-rank updates [6-12]. While highly attractive for privacy-sensitive domains such as healthcare and finance, practical deployments are strictly constrained by governance, availability, and bandwidth, permitting only limited interaction among highly heterogeneous clients. Statistical heterogeneity can induce unstable local drift and degrade naive aggregation, especially when clients perform multiple local updates [13,14]. Therefore, an effective and efficient framework must disentangle globally transferable knowledge from specialized domain expertise, achieving effective refinement under strictly constrained interactions.

Existing federated LoRA methods process the factor pair differently. Some prior works aggregate factors into a single shared adapter, which inherently struggles under severe domain shifts [15-17]. Others retain certain factors locally, sharing \(A\) while isolating \(B\), but completely block knowledge reuse across compatible clients [18]. Meanwhile, personalized and expert-based strategies maintain client-specific adapters, routed experts, or hierarchical structures, yet may rely on metadata, learned routers, heavier state, or multi-round refinement [19-22]. However, realizing this regime under one-shot communication presents two fundamental obstacles. First, the factors cannot be directly compared: LoRA parameterizations are gauge-dependent, meaning functionally identical updates can manifest as unrelated raw coordinates [23]. Second, under a single round of communication, a mistaken grouping cannot be repaired in subsequent rounds.

As illustrated in Fig. 1, this missing middle ground lies in the asymmetric geometry of \(B\). Given a shared initialization, the row spaces of factor \(A\) remain broadly aligned across diverse domains, whereas the column spaces of \(B\) partition into distinct domain-organized subspaces (Fig. 1(a,b)). In fact, domain-specific information is overwhelmingly concentrated in \(B\) rather than \(A\) (Fig. 1(c)). This geometric structure carries direct functional implications because holding \(A\) fixed and swapping \(B\) within the same domain induces negligible performance drop, while cross-domain swapping causes severe degradation (Fig. 1(d)). Hence, the factor traditionally treated as client-private is actually shareable among compatible groups. More broadly, clustered and representation-sharing personalized FL demonstrates that compatible clients can benefit from structured knowledge reuse rather than either one global model or fully isolated local models [24-26]. In essence, \(B\) is too heterogeneous to average globally, yet too structured to remain isolated. Capturing this geometric compatibility allows compatible clients to reuse expert knowledge, leading directly to our principle: *sharing what transfers and routing what differs*.

Motivated by this factor-information asymmetry, we introduce FedPLoRA, a one-shot geometry-aware framework for personalized federated LoRA. All clients receive the same frozen backbone and layer-wise LoRA initialization, train both factors once on private data, and encode a single upload. For the broadly transferable path, each client sends a rank-\(k\) truncated-SVD sketch of its \(A\)-correction; the server reconstructs and sample-weightedly aggregates these sketches into one shared factor \(A^\star\). For the heterogeneous path, each client retains the complete \(B\) in its payload. The server extracts orthonormal bases of the \(B\)-column spaces, measures client compatibility using layer-averaged principal-angle affinity, discovers groups by average-linkage clustering with silhouette-based model-order selection, and sample-weightedly pools the complete \(B\) factors only within each group. Because the affinity depends on represented subspaces, it is insensitive to invertible changes of basis inside the LoRA latent coordinates. The server finally hard-routes one pooled expert \(\bar B_{c(i)}\) to client \(i\), so every personalized adapter combines the common \(A^\star\) with a compatible \(B\) expert. FedPLoRA therefore completes personalization with one upload and one personalized download, without domain labels, a trainable router, global-\(B\) interpolation, post-aggregation fine-tuning, or an additional communication round. The resulting expert bank can also be reused by a client excluded from federation through a small, domain-label-free \(B\)-geometry probe, without restarting collaborative training.

Our contributions are as follows:

1. We quantify that domain separation is concentrated in the \(B\)-column spaces while \(A\) remains broadly aligned. Controlled \(B\)-swap interventions show that this geometry is behaviorally consequential and, crucially, that client-specific \(B\) expertise is reusable within compatible groups.

2. Our method turns one ordinary local LoRA update into a compressed shared-\(A\) channel and a geometry-routed \(B\)-expert channel. It constructs and returns personalized adapters in one upload and one download, without domain labels, a learned router, or another optimization round.

3. Results on D1-9k and FlowerTune-Mixed, factor and route controls, stronger non-IID settings, backbones up to 3B parameters, and strict held-out-client evaluation verify the communication-personalization trade-off and the reuse of the discovered expert bank beyond participating clients.

## Related Work

The broader parameter-efficient fine-tuning literature establishes several ways to adapt frozen foundation models with compact trainable state, including bottleneck adapters, prefix and prompt tuning, activation scaling, adaptive-rank LoRA, quantization-aware LoRA, and weight-decomposed LoRA [6-12]. FedPETuning provided an early systematic study of bringing representative parameter-efficient tuning methods into federated language-model adaptation [27].

Federated Low-Rank Adaptation initially focused on making a global low-rank update cheaper and mathematically better defined. Direct FedAvg-LoRA applies federated averaging to the two factors of a standard LoRA adapter, despite the fact that separately averaging a bilinear factorization need not reproduce the average composed update [1,5 done:1]. FFA-LoRA addresses both instability and communication by freezing the randomly initialized \(A\) factor and training and transmitting only the zero-initialized \(B\) factor [15]. FLoRA instead stacks client factors so that the server represents the weighted sum of heterogeneous-rank updates without factor-averaging cross terms [16]. FlexLoRA synthesizes an aggregated update and uses singular value decomposition to redistribute rank-specific adapters to resource-heterogeneous clients, while HetLoRA adapts local ranks to heterogeneous on-device resources [28,29]. Communication-oriented work reduces repeated traffic through round-robin LoRA-segment sharing, adaptive sparsification, and lossless encoding in EcoLoRA, while YOCO targets true one-shot multimodal federated learning through directional supervision, sign-regularized \(B\), and sparsely regularized \(A\) [17,30]. FedEx-LoRA and LoRA-FAIR further correct aggregation error induced by separately averaged low-rank factors, whereas GLoRA replaces raw-factor averaging with a consensus update subspace and shared reference coordinates that are invariant to gauge-equivalent factorizations [23,31,32]. Collectively, this line improves aggregation correctness, privacy, rank compatibility, or communication, but primarily returns one global server representation or enforces global consistency rather than constructing domain-matched personalized adapters.

A complementary line exploits factor asymmetry or introduces explicit personalized structure. FedSA-LoRA shows that \(A\) tends to capture general knowledge whereas \(B\) is more client-specific, and consequently aggregates only \(A\) while leaving each \(B\) at its source [18]. Beyond factor selection, FedDAT uses a dual-adapter teacher and mutual knowledge distillation to regularize heterogeneous multimodal clients, and the centralized HydraLoRA reference shares one \(A\) across multiple routed \(B\) experts [22,33]. Personalized federated methods further maintain separate knowledge paths: FedALT pairs an individual LoRA with a frozen Rest-of-World LoRA and learns an input-dependent mixer [21]. pFedLoRA shares a homogeneous low-rank adapter across model-heterogeneous clients, FedTT communicates tensorized adapters, and FedSelect learns which parameter subsets should remain personalized [34-36]. FedLEASE is particularly close to our setting: it uses cosine similarity between briefly trained \(B\) matrices and silhouette-based clustering to allocate domain experts, followed by iterative expert training and an adaptive top-\(M\) Mixture-of-Experts router [19]. HiLoRA likewise infers latent groups from LoRA-subspace similarity, but organizes adaptation into root, cluster, and leaf tiers trained through cross-tier orthogonality and cascaded optimization [20]. Thus, prior methods either collapse heterogeneous knowledge into a global state, isolate reusable \(B\) factors, or require paired or hierarchical adapters and iterative router optimization; FedPLoRA fills the unresolved one-shot gap by compressing and sharing the transferable \(A\) correction while using basis-insensitive \(B\)-subspace geometry to discover, pool, and hard-route one compatible expert from the first and only upload.

> **Source-PDF transcription note.** The PDF places the FedPLoRA framework here but gives it the unrelated caption below. The caption is retained verbatim for audit rather than silently corrected.

**Figure 2: Comparison between the backdoor attack workflow and the MDBD two-phase defense pipeline.**

## Method

### Overview

This section develops FedPLoRA, a one-shot federated adaptation method that preserves cross-domain structure without repeatedly exchanging LoRA parameters. All clients start from the same frozen backbone and LoRA initialization, train once on private instruction data, and upload the dense LoRA-\(B\) factors together with low-rank sketches of their LoRA-\(A\) updates. The server then completes three coupled steps: it discovers compatible clients from the subspace geometry of \(B\), pools one routed \(B\) expert per group, and reconstructs a shared \(A\) correction from the uploaded sketches. Each client finally receives the shared \(A\) and its routed \(B\), yielding a personalized adapter after a single upload and a single download. Thus, FedPLoRA separates cross-client knowledge sharing from domain-sensitive routing while retaining a compact one-shot protocol.

### Problem Formulation

We first formalize the shared and personalized quantities that FedPLoRA must estimate. Consider \(N\) clients with private datasets \(\{\mathcal D_i\}_{i=1}^{N}\), where \(n_i=|\mathcal D_i|\), \(n=\sum_i n_i\), and \(\omega_i=n_i/n\). The frozen backbone contains linear layers \(W_{0,\ell}\in\mathbb R^{q_\ell\times p_\ell}\). We use the standard LoRA parameterization at every adapted layer [5].

\[
W_{i,\ell}
=W_{0,\ell}+s_\ell B_{i,\ell}A_{i,\ell},
\qquad
A_{i,\ell}\in\mathbb R^{r_\ell\times p_\ell},
\quad
B_{i,\ell}\in\mathbb R^{q_\ell\times r_\ell},
\tag{1}
\]

where \(r_\ell\ll\min(p_\ell,q_\ell)\) and \(s_\ell\) is the LoRA scaling factor. A single global average of both factors is restrictive under domain heterogeneity: \(A\) should preserve transferable input-side coordinates, whereas \(B\) must retain domain-sensitive output directions. We therefore seek one shared factor \(A_\ell^\star\), a set of routed factors \(\{\bar B_{c,\ell}\}_{c=1}^{C}\), and a client-to-pool assignment \(c(i)\). The resulting personalized objective is

\[
\min_{A^\star,\{\bar B_c\},c}
\sum_{i=1}^{N}\omega_i
\mathbb E_{(x,y)\sim\mathcal D_i}
\left[
\ell\!\left(f_{\Theta_0,A^\star,\bar B_{c(i)}}(x),y\right)
\right],
\tag{2}
\]

where \(\Theta_0\) denotes the frozen backbone and \(\ell\) is the token-level language-modeling loss. This formulation makes the method's central goal explicit: learn a common coordinate factor while routing only the factor that expresses heterogeneous output geometry.

### One-Shot Local Adaptation and Sketched Upload

The first step extracts both transferable and domain-sensitive adaptation signals without adding local regularizers to the finalized training rule. Client \(i\) initializes \(A_{i,\ell}=A_{0,\ell}\) and \(B_{i,\ell}=0\), freezes \(\Theta_0\), and optimizes the standard causal language-modeling loss:

\[
(A_i,B_i)=\arg\min_{A,B}\mathcal L_i(A,B),
\qquad
\mathcal L_i
=-\frac{1}{|\mathcal T_i|}
\sum_{t\in\mathcal T_i}
\log p_{\Theta_0,A,B}(y_t\mid y_{<t},x),
\tag{3}
\]

where \(\mathcal T_i\) is the set of supervised target tokens at client \(i\). After local training, the client keeps the full \(B_{i,\ell}\) payload because it is needed for geometric routing and pool construction. The client forms \(\Delta A_{i,\ell}=A_{i,\ell}-A_{0,\ell}\) and uploads its rank-\(k_\ell\) truncated SVD:

\[
\Pi_{k_\ell}(\Delta A_{i,\ell})
=U_{i,\ell}\Sigma_{i,\ell}V_{i,\ell}^{\top},
\tag{4}
\]

where \(U_{i,\ell}\in\mathbb R^{r_\ell\times k_\ell}\), \(\Sigma_{i,\ell}\in\mathbb R^{k_\ell\times k_\ell}\) is diagonal, and \(V_{i,\ell}\in\mathbb R^{p_\ell\times k_\ell}\). If \(\{\sigma_{i,\ell,h}\}\) are the singular values of \(\Delta A_{i,\ell}\) in descending order, the discarded energy is

\[
e_{i,\ell}^{(k_\ell)}
=\|\Delta A_{i,\ell}-\Pi_{k_\ell}(\Delta A_{i,\ell})\|_F^2
=\sum_{h>k_\ell}\sigma_{i,\ell,h}^{2}.
\tag{5}
\]

The truncated SVD is the minimum-error rank-\(k_\ell\) approximation under the Frobenius norm by the Eckart-Young theorem [37]. The upload is therefore

\[
\mathcal U_i
=\{B_{i,\ell},U_{i,\ell},\Sigma_{i,\ell},V_{i,\ell}^{\top}\}_{\ell},
\quad\text{together with }n_i,
\]

plus any trainable task head. This step preserves the complete routing signal in \(B_i\) while compressing the shared correction carried by \(A_i\).

### Geometry-Aware Routed Aggregation

The next step prevents incompatible \(B\) factors from being averaged into a single global adapter. For each layer, the server computes a reduced QR factorization \(B_{i,\ell}=Q_{i,\ell}R_{i,\ell}\). Let \(\sigma_h(Q_{i,\ell}^{\top}Q_{j,\ell})\) be the cosine of the \(h\)-th principal angle, let \(m_\ell\) be the number of such angles, and let \(\mathcal L_{ij}\) contain the layers available for both clients. The layer similarity, client similarity, and distance are

\[
s_{ij}^{(\ell)}
=\frac{1}{m_\ell}\sum_{h=1}^{m_\ell}
\sigma_h(Q_{i,\ell}^{\top}Q_{j,\ell})
=\frac{\|Q_{i,\ell}^{\top}Q_{j,\ell}\|_*}{m_\ell},
\]

\[
s_{ij}
=\frac{1}{|\mathcal L_{ij}|}
\sum_{\ell\in\mathcal L_{ij}}s_{ij}^{(\ell)},
\qquad
d_{ij}=1-s_{ij}.
\tag{6}
\]

The computation of principal angles from orthonormal bases and the singular values of \(Q_i^\top Q_j\) follows the classical Björck-Golub construction [38]. The metric is insensitive to LoRA right-coordinate changes. For any invertible \(G_{i,\ell}\),

\[
(B_{i,\ell}G_{i,\ell})(G_{i,\ell}^{-1}A_{i,\ell})
=B_{i,\ell}A_{i,\ell},
\qquad
\operatorname{col}(B_{i,\ell}G_{i,\ell})
=\operatorname{col}(B_{i,\ell}).
\tag{7}
\]

The singular values in Eq. (6) remain unchanged under Eq. (7). The server next applies average-linkage agglomerative clustering, a standard hierarchical clustering construction [39]. For two current clusters \(\mathcal C_a\) and \(\mathcal C_b\), the merge distance is

\[
D(\mathcal C_a,\mathcal C_b)
=\frac{1}{|\mathcal C_a||\mathcal C_b|}
\sum_{i\in\mathcal C_a}
\sum_{j\in\mathcal C_b}d_{ij}.
\tag{8}
\]

To determine the number of pools, define the mean distance from client \(i\) to a set \(\mathcal C\) as

\[
\bar d(i,\mathcal C)
=|\mathcal C|^{-1}\sum_{j\in\mathcal C}d_{ij}.
\]

For a candidate partition with assignment \(c_K(i)\),

\[
a_i^{(K)}
=\bar d\!\left(i,\mathcal C_{c_K(i)}\setminus\{i\}\right),
\qquad
b_i^{(K)}
=\min_{c\ne c_K(i)}\bar d(i,\mathcal C_c),
\]

\[
h_i^{(K)}
=\frac{b_i^{(K)}-a_i^{(K)}}
{\max(a_i^{(K)},b_i^{(K)})}.
\tag{9}
\]

The implementation sets \(a_i^{(K)}=0\) for a singleton and selects

\[
K^\star
=\arg\max_{2\le K\le K_{\max}}
\frac{1}{N}\sum_{i=1}^{N}h_i^{(K)}.
\tag{10}
\]

The silhouette coefficient supplies the cluster-validity criterion used in Eqs. (9)-(10) [40]. If every candidate score is negative, the router falls back to one pool. No domain label enters either Eq. (8) or Eq. (10). For each discovered cluster \(\mathcal C_c\), the server forms a sample-size-weighted \(B\) expert:

\[
\bar B_{c,\ell}
=\sum_{i\in\mathcal C_c}\rho_{i\mid c}B_{i,\ell},
\qquad
\rho_{i\mid c}
=\frac{n_i}{\sum_{j\in\mathcal C_c}n_j}.
\tag{11}
\]

Each participating client is hard-routed to its own discovered cluster \(c(i)\), rather than blending several experts or mixing in a global \(B\). This step turns unlabeled \(B\)-subspace geometry into one stable domain-compatible factor per client.

### Shared-Coordinate Correction and Personalized Composition

The final step injects transferable information into a common \(A\) without discarding the routed \(B\) structure. The server reconstructs each sketched local factor and computes

\[
\widehat A_{i,\ell}
=A_{0,\ell}
+U_{i,\ell}\Sigma_{i,\ell}V_{i,\ell}^{\top},
\]

\[
A_\ell^\star
=\sum_{i=1}^{N}\omega_i\widehat A_{i,\ell}
=A_{0,\ell}
+\sum_{i=1}^{N}
\omega_iU_{i,\ell}\Sigma_{i,\ell}V_{i,\ell}^{\top}.
\tag{12}
\]

Let

\[
A_\ell^{\mathrm{dense}}
=A_{0,\ell}+\sum_i\omega_i\Delta A_{i,\ell}
\]

denote the aggregate without sketching. Equations (5) and (12) yield

\[
\|A_\ell^{\mathrm{dense}}-A_\ell^\star\|_F
\le
\sum_{i=1}^{N}\omega_i
\sqrt{e_{i,\ell}^{(k_\ell)}}.
\tag{13}
\]

The finalized method applies the complete reconstructed correction, with neither row-norm clipping nor an initialization-anchor penalty. Client \(i\) receives \(A^\star\) and \(\bar B_{c(i)}\), producing

\[
W_{i,\ell}^{\star}
=W_{0,\ell}
+s_\ell\bar B_{c(i),\ell}A_\ell^\star.
\tag{14}
\]

Thus, all clients share the same corrected input-side coordinates, while their output-side adaptation remains personalized through a routed pool.

### Algorithm 1: FedPLoRA

**Input:** frozen backbone \(\Theta_0\), initial \(A_0\), client datasets \(\{\mathcal D_i\}_{i=1}^{N}\), and sketch ranks \(\{k_\ell\}\).

**Output:** personalized adapters \(\{(A^\star,\bar B_{c(i)})\}_{i=1}^{N}\).

1. For each client \(i=1,\ldots,N\) in parallel:
   1. Initialize \(A_i\leftarrow A_0\) and \(B_i\leftarrow0\).
   2. Optimize \((A_i,B_i)\) using Eq. (3).
   3. Sketch each \(\Delta A_{i,\ell}\) using Eq. (4).
   4. Upload \(\mathcal U_i\) and \(n_i\).
2. Compute \(D=[d_{ij}]\) using Eq. (6).
3. Infer \(\{\mathcal C_c\}_{c=1}^{C}\) using Eqs. (8)-(10).
4. Construct \(\{\bar B_c\}_{c=1}^{C}\) using Eq. (11).
5. Reconstruct and aggregate \(A^\star\) using Eq. (12).
6. For each client \(i=1,\ldots,N\), return \((A^\star,\bar B_{c(i)})\).

## Experiment

### Experimental Setup

#### Benchmarks

We evaluate multi-domain federated instruction tuning on two complementary suites. D1-9k contains seven public-data domains - Code, Education, Finance, General, Legal, Math, and Medical - with five clients per domain and \(35\) clients in total. Within each domain, examples are partitioned by a Dirichlet distribution with concentration \(0.5\). FlowerTune-Mixed combines four public FlowerTune sources - Code, Finance, General, and Medical - into a custom mixed-domain federation with five clients per domain and \(20\) clients in total. FlowerTune provides the underlying cross-domain federated LLM benchmark and source taxonomy [41].

SmolLM2-135M is the primary backbone, and its model family is documented in the SmolLM2 technical report [42]. We additionally evaluate SmolLM2-1.7B and Qwen2.5-3B on D1-9k to test scale; Qwen2.5 is described in its official technical report [43]. Every main comparison averages three independently constructed splits or training seeds \(42,43,44\).

#### Baselines and Comparison Protocol

We compare against twelve baselines spanning four design families: (i) global-update and rank-heterogeneous aggregation - FedAvg-LoRA, FLoRA, and FlexLoRA; (ii) factor-selective and communication-efficient aggregation - FFA-LoRA, EcoLoRA, and FedSA-LoRA; (iii) heterogeneity-aware or native one-shot adaptation - FedDAT and YOCO; and (iv) personalized and expert-based adaptation - FedALT, HydraLoRA, HiLoRA, and FedLEASE [1,15-22,28,30,33 done:1]. All methods receive the same backbone, tokenizer, LoRA capacity, data split, optimizer, local epoch, and evaluation routine.

#### Implementation and Metrics

We attach LoRA to the query, key, value, output, gate, up, and down projections. Unless stated otherwise, the LoRA rank is \(8\), the scaling parameter is \(16\), and dropout is \(0.05\). Clients run one local epoch with AdamW, learning rate \(2\times10^{-4}\), warmup ratio \(0.03\), batch size \(2\), maximum sequence length \(256\), bfloat16 precision, and gradient checkpointing; AdamW follows decoupled weight-decay optimization [44]. FedPLoRA uses rank-\(2\) sketches of \(\Delta A\), \(K_{\max}=8\), full \(A\)-correction strength, and no anchor, proximal penalty, norm clipping, or global-\(B\) interpolation. Full evaluation is performed after aggregation without batch truncation. Experiments run on NVIDIA RTX 6000 Ada Generation 48GB GPUs.

The primary personalization metric is **Local** accuracy, the unweighted macro-average of client-local test token accuracy. **Macro** averages token accuracy over domain test sets, and **Worst** is the minimum domain accuracy. These two metrics expose broad-coverage and tail behavior that Local alone can hide. **Comm** is the payload-accounted bidirectional adapter traffic. Cluster NMI and domain labels are used only for post-hoc diagnosis, never for automatic routing or model selection.



## Conclusion

> **Source-PDF transcription note.** The source PDF contains the “Conclusion” heading but no conclusion text.

## References

[1] McMahan et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017.

[2] Kairouz et al. Advances and Open Problems in Federated Learning. Foundations and Trends in Machine Learning, 2021.

[3] Sheller et al. Federated Learning in Medicine: Facilitating Multi-Institutional Collaborations without Sharing Patient Data. Scientific Reports, 2020.

[4] Rieke et al. The Future of Digital Health with Federated Learning. npj Digital Medicine, 2020.

[5] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

[6] Houlsby et al. Parameter-Efficient Transfer Learning for NLP. ICML 2019.

[7] Li and Liang. Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL-IJCNLP 2021.

[8] Lester et al. The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP 2021.

[9] Liu et al. Few-Shot Parameter-Efficient Fine-Tuning Is Better and Cheaper than In-Context Learning. NeurIPS 2022.

[10] Zhang et al. Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. ICLR 2023.

[11] Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.

[12] Liu et al. DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024.

[13] Li et al. Federated Optimization in Heterogeneous Networks. MLSys 2020.

[14] Karimireddy et al. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. ICML 2020.

[15] Sun et al. Improving LoRA in Privacy-Preserving Federated Learning. ICLR 2024.

[16] Wang et al. FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations. NeurIPS 2024.

[17] Xu et al. You Only Communicate Once: One-Shot Federated Low-Rank Adaptation of MLLM. NeurIPS 2025.

[18] Guo et al. Selective Aggregation for Low-Rank Adaptation in Federated Learning. ICLR 2025.

[19] Wang et al. Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning. NeurIPS 2025.

[20] Peng et al. HiLoRA: Hierarchical Low-Rank Adaptation for Personalized Federated Learning. CVPR 2026.

[21] Bian et al. FedALT: Federated Fine-Tuning Through Adaptive Local Training with Rest-of-World LoRA. AAAI 2026.

[22] Tian et al. HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning. NeurIPS 2024.

[23] Chen, Liu, and Zhu. Beyond Factor Aggregation: Gauge-Aware Low-Rank Server Representations for Federated LoRA. arXiv preprint arXiv:2605.06733, 2026.

[24] Ghosh et al. An Efficient Framework for Clustered Federated Learning. NeurIPS 2020.

[25] Collins et al. Exploiting Shared Representations for Personalized Federated Learning. ICML 2021.

[26] Li et al. Ditto: Fair and Robust Federated Learning Through Personalization. ICML 2021.

[27] Zhang et al. FedPETuning: When Federated Learning Meets the Parameter-Efficient Tuning Methods of Pre-Trained Language Models. Findings of ACL 2023.

[28] Bai et al. Federated Fine-Tuning of Large Language Models under Heterogeneous Tasks and Client Resources. NeurIPS 2024.

[29] Cho et al. Heterogeneous LoRA for Federated Fine-Tuning of On-Device Foundation Models. EMNLP 2024.

[30] Liu et al. EcoLoRA: Communication-Efficient Federated Fine-Tuning of Large Language Models. EMNLP 2025.

[31] Singhal, Ponkshe, and Vepakomma. FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models. ACL 2025.

[32] Bian et al. LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement. ICCV 2025.

[33] Chen et al. FedDAT: An Approach for Foundation Model Finetuning in Multi-Modal Heterogeneous Federated Learning. AAAI 2024.

[34] Yi et al. pFedLoRA: Model-Heterogeneous Personalized Federated Learning with Homogeneous Low-Rank Adapter Sharing on Mobile Edge Devices. IEEE Transactions on Mobile Computing, 2026.

[35] Ghiasvand et al. Communication-Efficient and Tensorized Federated Fine-Tuning of Large Language Models. Findings of ACL 2025.

[36] Tamirisa et al. FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning. CVPR 2024.

[37] Eckart and Young. The Approximation of One Matrix by Another of Lower Rank. Psychometrika, 1936.

[38] Björck and Golub. Numerical Methods for Computing Angles Between Linear Subspaces. Mathematics of Computation, 1973.

[39] Murtagh and Contreras. Algorithms for Hierarchical Clustering: An Overview. WIREs Data Mining and Knowledge Discovery, 2012.

[40] Rousseeuw. Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. Journal of Computational and Applied Mathematics, 1987.

[41] Gao et al. FlowerTune: A Cross-Domain Benchmark for Federated Fine-Tuning of Large Language Models. arXiv preprint arXiv:2506.02961, 2025.

[42] Allal et al. SmolLM2: When Smol Goes Big - Data-Centric Training of a Small Language Model. arXiv preprint arXiv:2502.02737, 2025.

[43] Qwen Team. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115, 2024.

[44] Loshchilov and Hutter. Decoupled Weight Decay Regularization. ICLR 2019.

<!--
Primary-source verification index (internal authoring aid; remove before camera-ready if desired):
[1] https://proceedings.mlr.press/v54/mcmahan17a.html
[2] https://www.nowpublishers.com/article/DownloadSummary/MAL-083
[3] https://www.nature.com/articles/s41598-020-69250-1
[4] https://www.nature.com/articles/s41746-020-00323-1
[5] https://openreview.net/forum?id=nZeVKeeFYf9
[6] https://proceedings.mlr.press/v97/houlsby19a.html
[7] https://aclanthology.org/2021.acl-long.353/
[8] https://aclanthology.org/2021.emnlp-main.243/
[9] https://papers.nips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html
[10] https://openreview.net/forum?id=lq62uWRJjiY
[11] https://papers.nips.cc/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html
[12] https://proceedings.mlr.press/v235/liu24bn.html
[13] https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html
[14] https://proceedings.mlr.press/v119/karimireddy20a.html
[15] https://proceedings.iclr.cc/paper_files/paper/2024/hash/4e243e95c913b367775d71d7182b99d9-Abstract-Conference.html
[16] https://papers.nips.cc/paper_files/paper/2024/hash/28312c9491d60ed0c77f7fff4ad86dd1-Abstract-Conference.html
[17] https://papers.nips.cc/paper_files/paper/2025/hash/58e6c003c9fb3992265005ff6aef1913-Abstract-Conference.html
[18] https://proceedings.iclr.cc/paper_files/paper/2025/hash/f53a37f820d5be5930415d964f4a0187-Abstract-Conference.html
[19] https://papers.neurips.cc/paper_files/paper/2025/hash/6df1b2b45e64d402588746f79b68b82c-Abstract-Conference.html
[20] https://openaccess.thecvf.com/content/CVPR2026/html/Peng_HiLoRA_Hierarchical_Low-Rank_Adaptation_for_Personalized_Federated_Learning_CVPR_2026_paper.html
[21] https://ojs.aaai.org/index.php/AAAI/article/view/39054
[22] https://papers.nips.cc/paper_files/paper/2024/hash/123fd8a56501194823c8e0dca00733df-Abstract-Conference.html
[23] https://arxiv.org/abs/2605.06733
[24] https://papers.nips.cc/paper/2020/hash/e32cc80bf07915058ce90722ee17bb71-Abstract.html
[25] https://proceedings.mlr.press/v139/collins21a.html
[26] https://proceedings.mlr.press/v139/li21h.html
[27] https://aclanthology.org/2023.findings-acl.632/
[28] https://papers.nips.cc/paper_files/paper/2024/hash/1a134b50202088aa8c595cc99b310e5a-Abstract-Conference.html
[29] https://aclanthology.org/2024.emnlp-main.717/
[30] https://aclanthology.org/2025.emnlp-main.1046/
[31] https://aclanthology.org/2025.acl-long.67/
[32] https://openaccess.thecvf.com/content/ICCV2025/html/Bian_LoRA-FAIR_Federated_LoRA_Fine-Tuning_with_Aggregation_and_Initialization_Refinement_ICCV_2025_paper.html
[33] https://ojs.aaai.org/index.php/AAAI/article/view/29007
[34] https://doi.org/10.1109/TMC.2026.3674996
[35] https://aclanthology.org/2025.findings-acl.1241/
[36] https://openaccess.thecvf.com/content/CVPR2024/html/Tamirisa_FedSelect_Personalized_Federated_Learning_with_Customized_Selection_of_Parameters_for_CVPR_2024_paper.html
[37] https://doi.org/10.1007/BF02288367
[38] https://doi.org/10.2307/2005662
[39] https://doi.org/10.1002/widm.53
[40] https://doi.org/10.1016/0377-0427(87)90125-7
[41] https://arxiv.org/abs/2506.02961
[42] https://arxiv.org/abs/2502.02737
[43] https://arxiv.org/abs/2412.15115
[44] https://openreview.net/forum?id=Bkg6RiCqY7
-->
