"""Shared helpers for FedPLoRA-v4 server aggregators.

We re-export the v2 helpers and add v4-specific utilities (residual extraction,
soft sigmoid gate, spectral clustering on stacked residuals).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from methods import common as M  # noqa: E402  (v2 helpers)
from utilities.utils import (  # noqa: E402
    is_lora_a_param_name,
    is_task_head_param_name,
)


__all__ = [
    "M",
    "is_lora_a_param_name",
    "is_task_head_param_name",
    "client_weights",
    "metadata_from_uploads",
    "row_importance_tensor",
    "residual_state",
    "soft_sigmoid_gate",
    "spectral_clusters_from_residuals",
    "kmeans_clusters_from_residuals",
    "weighted_row_mean",
    "task_head_average",
]


def client_weights(client_uploads, args):
    sizes = [M.upload_package_client_size(u) for u in client_uploads]
    if all(s is None for s in sizes):
        sizes = getattr(args, "_fedplora_client_sizes", None)
    if sizes is None:
        n = max(len(client_uploads), 1)
        return np.ones(n, dtype=np.float64) / n
    arr = np.asarray(sizes, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        n = max(len(client_uploads), 1)
        return np.ones(n, dtype=np.float64) / n
    return arr / total


def metadata_from_uploads(client_uploads, args):
    runtime_ids = list(getattr(args, "_fedplora_round_client_ids", []) or [])
    domain_map = getattr(args, "_fedplora_client_domains", {}) or {}
    client_ids, domains = [], []
    for idx, upload in enumerate(client_uploads):
        cid = upload.get("client_id") if isinstance(upload, dict) else None
        domain = upload.get("domain") if isinstance(upload, dict) else None
        if cid is None and idx < len(runtime_ids):
            cid = runtime_ids[idx]
        if cid is None:
            cid = idx
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            cid = idx
        if domain is None:
            domain = domain_map.get(cid, domain_map.get(str(cid), "unknown"))
        domain = str(domain or "unknown").strip().lower()
        client_ids.append(cid)
        domains.append(domain)
    return client_ids, domains


def row_importance_tensor(row_importance_dict, key, num_rows):
    imp = row_importance_dict.get(key, None)
    if imp is None:
        return torch.ones(num_rows, dtype=torch.float32).view(-1, 1)
    imp = imp.detach().cpu().float()
    if imp.numel() != num_rows:
        return torch.ones(num_rows, dtype=torch.float32).view(-1, 1)
    return imp.view(-1, 1).clamp_min(1e-8)


def residual_state(client_states, client_row_importance, key, global_dict, args):
    """
    Compute per-layer residual stats used by Branch A.

    Returns:
        ref:           A_0 reference (CPU float32), shape (r, d)
        residuals:     list of (A_i - ref), CPU float32, shape (r, d)
        row_norms:     list of (r, 1)
        weights:       np array (N,) per-client sample weight
        importances:   list of (r, 1) per-client per-row importance
        active:        (r,) bool, rows with any meaningful update
        conflict_raw:  (r, 1) in [0, 1], not yet gated
    """
    initial_A = getattr(args, "_fedplora_initial_A", None) or {}
    mats = [state[key].detach().cpu().float() for state in client_states]
    ref = initial_A.get(key, None)
    if ref is None or tuple(ref.shape) != tuple(mats[0].shape):
        ref = global_dict[key].detach().cpu().float()
    else:
        ref = ref.detach().cpu().float()

    eps = float(getattr(args, "v4_residual_eps", 1e-7))
    residuals = [Ai - ref for Ai in mats]
    num_rows = ref.shape[0]

    row_norms, importances = [], []
    dir_acc = torch.zeros_like(ref)
    weight_sum = torch.zeros((num_rows, 1), dtype=torch.float32)
    norm_acc = torch.zeros((num_rows, 1), dtype=torch.float32)
    weights = client_weights([{"client_size": None}] * len(client_states), args)  # uniform if unknown

    # Override if uploads carry size info; caller should pre-set _fedplora_client_sizes.
    sizes_known = getattr(args, "_fedplora_client_sizes", None)
    if sizes_known is not None and len(sizes_known) == len(client_states):
        arr = np.asarray(sizes_known, dtype=np.float64)
        total = float(arr.sum())
        if total > 0:
            weights = arr / total

    for idx, residual in enumerate(residuals):
        norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
        row_norms.append(norm)
        direction = residual / norm.clamp_min(eps)
        imp = row_importance_tensor(client_row_importance[idx], key, num_rows)
        importances.append(imp)
        row_weight = float(weights[idx]) * imp * norm.clamp_min(eps)
        dir_acc = dir_acc + row_weight * direction
        weight_sum = weight_sum + row_weight
        norm_acc = norm_acc + float(weights[idx]) * norm

    safe = weight_sum.clamp_min(eps)
    mean_dir = dir_acc / safe
    consensus = torch.linalg.vector_norm(mean_dir, dim=1, keepdim=True).clamp(0.0, 1.0)
    conflict_raw = (1.0 - consensus).clamp(0.0, 1.0)
    active = (norm_acc.squeeze(1) > eps) & (weight_sum.squeeze(1) > eps)

    return {
        "ref": ref,
        "residuals": residuals,
        "row_norms": row_norms,
        "weights": weights,
        "importances": importances,
        "weight_sum": weight_sum,
        "active": active,
        "conflict_raw": conflict_raw,
        "norm_acc": norm_acc,
    }


def soft_sigmoid_gate(conflict, kappa=1.0, eps=1e-6):
    """Standardize conflict per layer, apply sigmoid — no hard quantile threshold.

    Argument:
        conflict: (r, 1) tensor in [0, 1]
        kappa:    temperature, smaller = sharper

    Returns:
        gate: (r, 1) in (0, 1)
    """
    c = conflict.detach().float().reshape(-1)
    if c.numel() <= 1:
        return torch.full_like(conflict, 0.5)
    mean = c.mean()
    std = c.std().clamp_min(eps)
    z = (conflict - mean) / std
    return torch.sigmoid(z / max(kappa, eps))


def _stack_flat_residuals(residuals):
    return torch.stack([r.detach().cpu().float().reshape(-1) for r in residuals], dim=0)


def spectral_clusters_from_residuals(residuals, k):
    """Spectral clustering on stacked client residual vectors (N, D).

    Returns dict cluster_id -> [client_idx, ...].
    """
    if k <= 0 or len(residuals) <= 1:
        return {0: list(range(len(residuals)))}
    flat = _stack_flat_residuals(residuals)                       # (N, D)
    flat = flat - flat.mean(dim=0, keepdim=True)
    U, S, _ = torch.linalg.svd(flat, full_matrices=False)
    proj = U[:, :min(k, U.shape[1])] * S[:min(k, S.shape[0])]      # (N, k)
    proj_np = proj.detach().cpu().numpy()
    return _kmeans_lloyd(proj_np, k)


def kmeans_clusters_from_residuals(residuals, k, max_iter=20, seed=0):
    """Vanilla k-means on flattened residuals."""
    if k <= 0 or len(residuals) <= 1:
        return {0: list(range(len(residuals)))}
    flat = _stack_flat_residuals(residuals).detach().cpu().numpy()
    return _kmeans_lloyd(flat, k, max_iter=max_iter, seed=seed)


def _kmeans_lloyd(data, k, max_iter=20, seed=0):
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    if n <= k:
        return {i: [i] for i in range(n)}
    idx = rng.choice(n, size=k, replace=False)
    centers = data[idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # Use squared L2 distance, broadcasting friendly
        d2 = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = data[mask].mean(axis=0)
    clusters = {}
    for i, c in enumerate(labels):
        clusters.setdefault(int(c), []).append(int(i))
    return clusters


def weighted_row_mean(residuals, importances, weights, indices):
    """Per-row weighted mean across a subset of clients."""
    if not indices:
        return torch.zeros_like(residuals[0])
    eps = 1e-8
    acc = torch.zeros_like(residuals[0])
    wsum = torch.zeros((residuals[0].shape[0], 1), dtype=torch.float32)
    for idx in indices:
        w = float(weights[idx]) * importances[idx]
        acc = acc + w * residuals[idx]
        wsum = wsum + w
    return acc / wsum.clamp_min(eps)


def task_head_average(global_dict, key, client_states, weights):
    return sum(float(weights[i]) * client_states[i][key].float() for i in range(len(client_states)))
