"""
FedPLoRA-v7: 非对称分层联邦 LoRA 聚合（Hierarchical Asymmetric FedLoRA）。

依据三项诊断（claude/dcr_feasibility_and_reviewer_risks.md §5.9/5.95/5.97）：
  - A 域通用（跨域子空间几乎共线，~5% 域信号）-> 全局共享 A。
  - B 域特异（跨域换 B 重损 +12.5%）-> 跨域绝不聚合 B。
  - B 域内功能冗余（同域换 B +0.1%）-> 同域 B 可池化/去噪。

v7 聚合规则（one-shot）：
  A_global = mean_i A_i                         # 全局，域通用
  B_d      = consolidate_{i in domain d} B_i    # 按域池化，跨域隔离
  下发：客户端 i(域 d) 得 (A_global, B_d)        # 每域一套，per-domain 个性化

B 池化模式（--v7_b_mode）：
  mean : 同域 B 直接平均（最简；若 gauge 不齐可能部分抵消）
  rep  : 取同域一个代表客户端的 B（gauge 自洽，不去噪；swap 实验证明单个 B 即可用）
  svd  : 同域 B 列空间 SVD 共识（gauge 鲁棒去噪，保留共同列方向能量）

本模块是纯函数（输入/输出都是 {key: tensor} 字典），不依赖训练框架，
既可被独立评测脚本调用，也可后续接入 tasks/fed_train_sft.py。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch


def aggregate_global_A(A_by_client: Dict[int, Dict[str, torch.Tensor]],
                       a_keys: List[str]) -> Dict[str, torch.Tensor]:
    """A_global = 跨所有客户端按层平均（A 域通用）。"""
    out = {}
    cids = list(A_by_client.keys())
    for k in a_keys:
        out[k] = torch.stack([A_by_client[c][k].float() for c in cids], 0).mean(0)
    return out


def _consolidate_B(B_list: List[torch.Tensor], mode: str) -> torch.Tensor:
    """把同域若干 B_i (d_out×r) 合成一个域级 B_d (d_out×r)。"""
    r = B_list[0].shape[1]
    if mode == "rep":
        return B_list[0].float().clone()
    if mode == "mean":
        return torch.stack([B.float() for B in B_list], 0).mean(0)
    if mode == "svd":
        # 同域 B 列方向共识：横拼 (d_out, n*r) -> SVD -> 取 top-r 左奇异方向(带能量)
        stacked = torch.cat([B.float() for B in B_list], dim=1)   # (d_out, n*r)
        U, S, _ = torch.linalg.svd(stacked, full_matrices=False)
        r_eff = min(r, S.numel())
        Bd = U[:, :r_eff] * S[:r_eff].unsqueeze(0)
        # 缩放到同域平均 Frobenius 范数，保持与 A_global 的量级匹配
        target = torch.stack([B.float() for B in B_list], 0).norm(dim=(1, 2)).mean()
        Bd = Bd * (target / Bd.norm().clamp_min(1e-12))
        if r_eff < r:  # 极少见：补零到 r
            Bd = torch.cat([Bd, torch.zeros(Bd.shape[0], r - r_eff)], dim=1)
        return Bd
    raise ValueError(f"unknown v7_b_mode: {mode}")


def aggregate_per_domain_B(B_by_client: Dict[int, Dict[str, torch.Tensor]],
                           domain_of: Dict[int, str],
                           b_keys: List[str],
                           mode: str = "mean") -> Dict[str, Dict[str, torch.Tensor]]:
    """返回 {domain: {b_key: B_d}}，每域一套（跨域隔离）。"""
    by_dom = defaultdict(list)
    for cid in B_by_client:
        by_dom[domain_of.get(cid, "?")].append(cid)
    out = {}
    for dom, cids in by_dom.items():
        out[dom] = {}
        for bk in b_keys:
            B_list = [B_by_client[c][bk] for c in cids if bk in B_by_client[c]]
            if B_list:
                out[dom][bk] = _consolidate_B(B_list, mode)
    return out


def build_v7_client_state(A_global: Dict[str, torch.Tensor],
                          B_per_domain: Dict[str, Dict[str, torch.Tensor]],
                          domain: str) -> Dict[str, torch.Tensor]:
    """组装某域客户端的下发权重：全局 A + 本域 B。返回 {key: tensor} 可直接 load。"""
    state = {}
    state.update({k: v.clone() for k, v in A_global.items()})
    state.update({k: v.clone() for k, v in B_per_domain.get(domain, {}).items()})
    return state
