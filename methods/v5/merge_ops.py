"""Core merge operators for FedPLoRA v5-merge.

All operators consume per-client LoRA factors (B_i ∈ R^{m×r_i}, A_i ∈ R^{r_i×n})
and normalized client weights w_i, and produce LoRA factors (B*, A*) of rank
``r_down`` whose product approximates the merged update ΔW*.

Memory discipline:
- ``ties``/``mean`` materialize ΔW* one layer at a time (m×n fp32, ≤ ~235 MB for
  Llama-8B MLP) but never stack all clients: per-client products are processed in
  row chunks so peak extra memory is O(chunk · n · N).
- ``knots`` never materializes any m×n matrix: it works entirely in the factored
  coefficient space C_i = B_i (U_i S) ∈ R^{m×k}, k ≤ Σ r_i.

References: TIES-Merging (Yadav et al., NeurIPS 2023), DARE (Yu et al., ICML 2024),
KnOTS (Stoica et al., ICLR 2025), FlexLoRA / FLoRA (NeurIPS 2024) for the
ΔW-SVD baseline protocol.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def normalize_weights(weights: Sequence[float], n: int) -> torch.Tensor:
    if weights is None or len(weights) < n:
        return torch.full((n,), 1.0 / max(n, 1), dtype=torch.float32)
    w = torch.tensor([float(weights[i]) for i in range(n)], dtype=torch.float32)
    s = float(w.sum())
    if s <= 0:
        return torch.full((n,), 1.0 / max(n, 1), dtype=torch.float32)
    return w / s


def delta_frobenius_norm(B: torch.Tensor, A: torch.Tensor) -> float:
    """||B A||_F without forming the product: trace((BᵀB)(AAᵀ))."""
    BtB = B.transpose(0, 1) @ B          # (r, r)
    AAt = A @ A.transpose(0, 1)          # (r, r)
    val = torch.sum(BtB * AAt.transpose(0, 1))
    return float(val.clamp_min(0.0).sqrt().item())


def estimate_abs_quantile(
    B: torch.Tensor,
    A: torch.Tensor,
    keep_ratio: float,
    sample_rows: int = 256,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Estimate the (1-keep_ratio) |ΔW| quantile from a row subsample of B·A."""
    keep_ratio = min(max(float(keep_ratio), 1e-4), 1.0)
    if keep_ratio >= 1.0:
        return 0.0
    m = B.shape[0]
    take = min(int(sample_rows), m)
    if take <= 0:
        return 0.0
    if take < m:
        idx = torch.randperm(m, generator=generator)[:take]
        rows = B[idx] @ A
    else:
        rows = B @ A
    vals = rows.abs().reshape(-1)
    # torch.quantile caps input size; subsample flat values if needed.
    if vals.numel() > 8_000_000:
        stride = vals.numel() // 8_000_000 + 1
        vals = vals[::stride]
    q = 1.0 - keep_ratio
    return float(torch.quantile(vals.float(), q).item())


def _apply_dare(t: torch.Tensor, dare_p: float, generator: Optional[torch.Generator]) -> torch.Tensor:
    """DARE: randomly drop entries with prob p, rescale survivors by 1/(1-p)."""
    p = float(dare_p)
    if p <= 0.0:
        return t
    p = min(p, 0.99)
    mask = torch.rand(t.shape, generator=generator, device=t.device) >= p
    return t * mask.to(t.dtype) / (1.0 - p)


# ---------------------------------------------------------------------------
# Entry-wise merges on ΔW (mean / ties), row-chunked
# ---------------------------------------------------------------------------


def merge_delta_entrywise(
    B_list: List[torch.Tensor],
    A_list: List[torch.Tensor],
    weights: torch.Tensor,
    *,
    mode: str = "ties",            # "mean" | "ties"
    keep_ratio: float = 0.2,
    dare_p: float = 0.0,
    chunk_rows: int = 2048,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Merge ΔW_i = B_i A_i entry-wise. Returns (ΔW* on CPU fp32, stats)."""
    n_clients = len(B_list)
    assert n_clients == len(A_list) and n_clients == weights.numel()
    device = device or torch.device("cpu")
    m, n = B_list[0].shape[0], A_list[0].shape[1]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    dev_gen = None
    if device.type != "cpu":
        dev_gen = torch.Generator(device=device)
        dev_gen.manual_seed(int(seed) + 1)

    Bs = [B_list[i].float().to(device) for i in range(n_clients)]
    As = [A_list[i].float().to(device) for i in range(n_clients)]
    w = weights.to(device)

    thresholds = None
    if mode == "ties" and keep_ratio < 1.0:
        thresholds = [
            estimate_abs_quantile(Bs[i], As[i], keep_ratio, generator=dev_gen if device.type != "cpu" else gen)
            for i in range(n_clients)
        ]

    merged = torch.zeros((m, n), dtype=torch.float32, device="cpu")
    total_entries = 0
    matched_entries = 0
    active_entries = 0

    for start in range(0, m, chunk_rows):
        stop = min(start + chunk_rows, m)
        if mode == "mean":
            acc = None
            for i in range(n_clients):
                d = (Bs[i][start:stop] @ As[i]) * float(w[i])
                acc = d if acc is None else acc + d
            merged[start:stop] = acc.cpu()
            continue

        # TIES path: trim -> (DARE) -> elect sign -> disjoint weighted mean
        trimmed = []
        for i in range(n_clients):
            d = Bs[i][start:stop] @ As[i]
            if thresholds is not None and thresholds[i] > 0:
                d = torch.where(d.abs() >= thresholds[i], d, torch.zeros_like(d))
            if dare_p > 0.0:
                d = _apply_dare(d, dare_p, dev_gen if device.type != "cpu" else gen)
            trimmed.append(d)

        elected = torch.zeros_like(trimmed[0])
        for i in range(n_clients):
            elected = elected + float(w[i]) * trimmed[i]
        gamma = torch.sign(elected)

        num = torch.zeros_like(trimmed[0])
        den = torch.zeros_like(trimmed[0])
        for i in range(n_clients):
            t = trimmed[i]
            match = (torch.sign(t) == gamma) & (t != 0)
            mf = match.to(t.dtype) * float(w[i])
            num = num + mf * t
            den = den + mf

        chunk_merged = torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))
        merged[start:stop] = chunk_merged.cpu()

        total_entries += int(gamma.numel())
        active_entries += int((den > 0).sum().item())
        matched_entries += int(sum(((torch.sign(t) == gamma) & (t != 0)).sum().item() for t in trimmed))

    stats = {
        "mode": mode,
        "active_frac": (active_entries / total_entries) if total_entries else 0.0,
        "mean_votes_per_active": (matched_entries / max(active_entries, 1)),
    }
    return merged, stats


# ---------------------------------------------------------------------------
# KnOTS-style subspace-aligned TIES (fully factored; never materializes m×n)
# ---------------------------------------------------------------------------


def knots_align_factors(
    B_list: List[torch.Tensor],
    A_list: List[torch.Tensor],
    weights: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    basis_energy_tau: float = 0.9999,
    k_cap: int = 512,
) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, float]]:
    """
    Build shared right basis V ∈ R^{n×k} of the weighted stacked updates
    concat_i(√w_i · B_i A_i) and per-client aligned coefficients C_i = ΔW_i · V.

    Steps (all thin / factored):
      1. P = stack_i(A_i) ∈ R^{(Σr_i)×n}; thin SVD P = U_p S_p V_pᵀ → provisional
         basis V_p spanning all client row spaces.
      2. Provisional coefficients C_i' = B_i (U_{p,i} S_p) where U_{p,i} is client
         i's row block of U_p.
      3. Exact alignment: H = Σ_i w_i C_i'ᵀ C_i' (k0×k0), eigh(H) → rotation W_h
         so V = V_p W_h is the exact right singular basis of the weighted stack
         restricted to span(P); C_i = C_i' W_h.

    Returns (C_list, V, stats).
    """
    n_clients = len(B_list)
    device = device or torch.device("cpu")
    As = [A_list[i].float().to(device) for i in range(n_clients)]
    Bs = [B_list[i].float().to(device) for i in range(n_clients)]
    w = weights.to(device)

    ranks = [a.shape[0] for a in As]
    P = torch.cat(As, dim=0)                                  # (Σr, n)
    U_p, S_p, Vh_p = torch.linalg.svd(P, full_matrices=False)  # thin: Σr ≤ ~10³
    # Drop numerically-dead directions of the stacked A.
    tol = float(S_p.max().item()) * 1e-7 if S_p.numel() else 0.0
    k0 = int((S_p > tol).sum().item())
    k0 = max(min(k0, int(k_cap)), 1)
    U_p = U_p[:, :k0]
    S_p = S_p[:k0]
    V_p = Vh_p[:k0, :].transpose(0, 1)                        # (n, k0)

    C_prov = []
    offset = 0
    for i in range(n_clients):
        Ui = U_p[offset: offset + ranks[i], :]                # (r_i, k0)
        offset += ranks[i]
        C_prov.append(Bs[i] @ (Ui * S_p.unsqueeze(0)))        # (m, k0)

    H = torch.zeros((k0, k0), dtype=torch.float32, device=device)
    for i in range(n_clients):
        H = H + float(w[i]) * (C_prov[i].transpose(0, 1) @ C_prov[i])
    evals, evecs = torch.linalg.eigh(H)                       # ascending
    order = torch.argsort(evals, descending=True)
    evals = evals.clamp_min(0.0)[order]
    W_h = evecs[:, order]                                     # (k0, k0)

    total = float(evals.sum().item())
    if total > 0 and basis_energy_tau < 1.0:
        cum = torch.cumsum(evals, dim=0) / total
        k = int((cum < basis_energy_tau).sum().item()) + 1
    else:
        k = k0
    k = max(min(k, k0), 1)
    W_h = W_h[:, :k]

    V = V_p @ W_h                                             # (n, k)
    C_list = [c @ W_h for c in C_prov]                        # (m, k)
    stats = {"k_basis": float(k), "k0": float(k0)}
    return C_list, V, stats


def ties_on_coefficients(
    C_list: List[torch.Tensor],
    weights: torch.Tensor,
    *,
    keep_ratio: float = 0.2,
    dare_p: float = 0.0,
    normalize: bool = True,
    seed: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """TIES on aligned coefficient matrices C_i ∈ R^{m×k}; returns (C*, stats)."""
    n_clients = len(C_list)
    device = C_list[0].device
    w = weights.to(device)
    gen = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    gen.manual_seed(int(seed) + 7)

    norms = torch.stack([c.norm().clamp_min(1e-12) for c in C_list])
    target_norm = float((w * norms).sum().item())

    trimmed = []
    for i in range(n_clients):
        c = C_list[i]
        if normalize:
            c = c / norms[i]
        if keep_ratio < 1.0:
            flat = c.abs().reshape(-1)
            if flat.numel() > 8_000_000:
                stride = flat.numel() // 8_000_000 + 1
                flat = flat[::stride]
            thr = torch.quantile(flat.float(), 1.0 - keep_ratio)
            c = torch.where(c.abs() >= thr, c, torch.zeros_like(c))
        if dare_p > 0.0:
            c = _apply_dare(c, dare_p, gen)
        trimmed.append(c)

    elected = torch.zeros_like(trimmed[0])
    for i in range(n_clients):
        elected = elected + float(w[i]) * trimmed[i]
    gamma = torch.sign(elected)

    num = torch.zeros_like(trimmed[0])
    den = torch.zeros_like(trimmed[0])
    for i in range(n_clients):
        t = trimmed[i]
        match = (torch.sign(t) == gamma) & (t != 0)
        mf = match.to(t.dtype) * float(w[i])
        num = num + mf * t
        den = den + mf
    merged = torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))

    if normalize:
        mn = merged.norm().clamp_min(1e-12)
        merged = merged * (target_norm / float(mn.item()))

    stats = {
        "active_frac": float((den > 0).float().mean().item()),
        "merged_norm": float(merged.norm().item()),
        "target_norm": target_norm,
    }
    return merged, stats


# ---------------------------------------------------------------------------
# Factorization back to LoRA factors
# ---------------------------------------------------------------------------


def factorize_dense_delta(
    delta: torch.Tensor,
    r_down: int,
    *,
    device: Optional[torch.device] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank-r SVD of dense ΔW* → (A_new (r,n), B_new (m,r), singular values)."""
    device = device or torch.device("cpu")
    d = delta.float().to(device)
    m, n = d.shape
    r = int(min(r_down, m, n))
    if min(m, n) > max(4 * r, 64):
        U, S, V = torch.svd_lowrank(d, q=min(max(2 * r, r + 8), min(m, n)), niter=4)
        U, S, V = U[:, :r], S[:r], V[:, :r]
        Vh = V.transpose(0, 1)
    else:
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
    sqrt_e = torch.sqrt(S.clamp_min(eps))
    B_new = U * sqrt_e.unsqueeze(0)
    A_new = sqrt_e.unsqueeze(1) * Vh
    return A_new, B_new, S


def factorize_coefficient_delta(
    C_star: torch.Tensor,
    V: torch.Tensor,
    r_down: int,
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ΔW* = C* Vᵀ with C* ∈ R^{m×k}, V ∈ R^{n×k}: thin SVD of C* gives exact
    factors without materializing m×n.
    """
    U_c, S_c, Wh_c = torch.linalg.svd(C_star, full_matrices=False)   # k ≤ ~512
    r = int(min(r_down, S_c.numel()))
    U_c, S_c, Wh_c = U_c[:, :r], S_c[:r], Wh_c[:r, :]
    sqrt_e = torch.sqrt(S_c.clamp_min(eps))
    B_new = U_c * sqrt_e.unsqueeze(0)                                 # (m, r)
    A_new = sqrt_e.unsqueeze(1) * (V @ Wh_c.transpose(0, 1)).transpose(0, 1)  # (r, n)
    return A_new, B_new, S_c


def pick_rank_by_energy(
    singular_values: torch.Tensor,
    *,
    tau: float = 0.95,
    r_min: int = 1,
    r_cap: int = 64,
) -> int:
    """Smallest r with cumulative squared-singular-value energy ≥ tau (capped)."""
    s2 = singular_values.float().pow(2)
    total = float(s2.sum().item())
    if total <= 0:
        return int(r_min)
    cum = torch.cumsum(s2, dim=0) / total
    r = int((cum < float(tau)).sum().item()) + 1
    return int(min(max(r, r_min), r_cap, singular_values.numel()))
