"""Branch A — FedPLoRA-Hier++ aggregator.

Fixes the v3 regression with:
  1. Soft sigmoid gate (no hard quantile threshold).
  2. Optional data-driven clusters (spectral / kmeans) in addition to domain prior.
  3. Properly differentiated default vs personalized A_down paths.

Reads (via args):
  - v4_gate_kappa            float (>0)        sigmoid temperature (default 1.0)
  - v4_gate_power            float             gate^p exponent (default 1.0)
  - v4_cluster_mode          str               {prior, spectral, kmeans, none}
  - v4_cluster_k             int               cluster count when not prior
  - v4_lambda_min            float             lambda lower bound (default 0.3)
  - v4_lambda_max            float             lambda upper bound (default 0.9)
  - v4_residual_eps          float             small eps for row activity
  - v4_personalized_eval     bool (1/0)        ship per-client A in personalized_states
  - v4_default_uniform       bool (1/0)        default path uses unweighted mean

Writes (via args):
  - _fedplora_v4_stats                 dict with per-layer + _summary
  - _fedplora_personalized_shared_states dict[client_id] -> {key: A_down}
  - _fedplora_v4_client_clusters       dict[client_id] -> cluster_id

The server still loads a single `global_dict` into `global_model`; per-client
A_down lives in `_fedplora_personalized_shared_states` and is broadcast in
fed_train_sft's eval routine.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from methods.common_v4 import (
    M,
    client_weights,
    is_lora_a_param_name,
    is_task_head_param_name,
    kmeans_clusters_from_residuals,
    metadata_from_uploads,
    residual_state,
    soft_sigmoid_gate,
    spectral_clusters_from_residuals,
    task_head_average,
    weighted_row_mean,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Prior clusters (kept for backwards compatibility with v3 setup).
_DOMAIN_PRIOR_CLUSTERS = {
    "general":   "general_education",
    "education": "general_education",
    "math":      "capability",
    "code":      "capability",
    "medical":   "risk",
    "legal":     "risk",
    "finance":   "risk",
}


def _float_arg(args, name, default):
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError):
        return float(default)


def _int_arg(args, name, default):
    try:
        return int(getattr(args, name, default))
    except (TypeError, ValueError):
        return int(default)


def _bool_arg(args, name, default=False):
    raw = getattr(args, name, default)
    if isinstance(raw, bool):
        return raw
    try:
        return bool(int(raw))
    except (TypeError, ValueError):
        return bool(raw)


def _assign_clusters(residuals, domains, args):
    mode = str(getattr(args, "v4_cluster_mode", "prior") or "prior").lower()
    if mode in {"none", "off"}:
        return {i: i for i in range(len(domains))}, {i: [i] for i in range(len(domains))}

    if mode == "prior":
        labels = []
        for d in domains:
            labels.append(_DOMAIN_PRIOR_CLUSTERS.get(d, d or "unknown"))
        clusters = defaultdict(list)
        for i, lbl in enumerate(labels):
            clusters[lbl].append(i)
        return {i: labels[i] for i in range(len(labels))}, dict(clusters)

    k = _int_arg(args, "v4_cluster_k", 3)
    if mode == "spectral":
        clusters = spectral_clusters_from_residuals(residuals, k)
    elif mode == "kmeans":
        clusters = kmeans_clusters_from_residuals(residuals, k)
    else:
        # Unknown mode: treat as one global cluster
        clusters = {0: list(range(len(residuals)))}

    client_to_cluster = {}
    for cid, members in clusters.items():
        for m in members:
            client_to_cluster[m] = cid
    return client_to_cluster, clusters


def _summary(layer_stats, variant, num_clusters):
    total = sum(int(v.get("num_rows", 0)) for v in layer_stats.values())
    if total <= 0:
        return {"variant": variant, "num_clusters": int(num_clusters)}

    def wmean(field):
        return float(
            sum(float(v.get(field, 0.0)) * int(v.get("num_rows", 0)) for v in layer_stats.values())
            / total
        )

    return {
        "variant": variant,
        "num_lora_a_matrices": int(len(layer_stats)),
        "num_rows": int(total),
        "num_clusters": int(num_clusters),
        "mean_conflict": wmean("mean_conflict"),
        "max_conflict": float(max(float(v.get("max_conflict", 0.0)) for v in layer_stats.values())),
        "mean_gate": wmean("mean_gate"),
        "mean_residual_norm": wmean("mean_residual_norm"),
        "mean_lambda_g": wmean("mean_lambda_g"),
        "mean_common_norm": wmean("mean_common_norm"),
        "mean_cluster_norm": wmean("mean_cluster_norm"),
    }


def aggregate_models_v4_hier(global_model, client_uploads, args, *, variant_tag="hier"):
    """Branch A: residual + soft gate + (optional) data-driven cluster personalization."""
    global_dict = global_model.state_dict()
    client_states = [M.upload_package_state(u) for u in client_uploads]
    client_row_importance = [M.upload_package_row_importance(u) for u in client_uploads]
    if not client_states:
        return global_model

    weights = client_weights(client_uploads, args)
    args._fedplora_client_sizes = [M.upload_package_client_size(u) for u in client_uploads]
    if all(x is None for x in args._fedplora_client_sizes):
        args._fedplora_client_sizes = getattr(args, "_fedplora_client_sizes", None)

    client_ids, domains = metadata_from_uploads(client_uploads, args)

    # Compute residuals on the first lora_A key to feed clustering; cheaper than per-key.
    cluster_seed_key = next(
        (k for k in global_dict.keys() if is_lora_a_param_name(k)
         and all(k in st for st in client_states)),
        None,
    )
    if cluster_seed_key is None:
        return global_model
    seed_residuals = [
        client_states[i][cluster_seed_key].detach().cpu().float()
        - global_dict[cluster_seed_key].detach().cpu().float()
        for i in range(len(client_states))
    ]
    client_to_cluster, clusters = _assign_clusters(seed_residuals, domains, args)

    kappa = _float_arg(args, "v4_gate_kappa", 1.0)
    power = _float_arg(args, "v4_gate_power", 1.0)
    lam_min = _float_arg(args, "v4_lambda_min", 0.3)
    lam_max = _float_arg(args, "v4_lambda_max", 0.9)
    use_personalized = _bool_arg(args, "v4_personalized_eval", True)
    default_uniform = _bool_arg(args, "v4_default_uniform", True)

    if lam_max < lam_min:
        lam_min, lam_max = lam_max, lam_min

    personalized_states = {int(cid): {} for cid in client_ids} if use_personalized else {}
    layer_stats = {}

    for key in list(global_dict.keys()):
        if is_task_head_param_name(key) and all(key in st for st in client_states):
            head = task_head_average(global_dict, key, client_states, weights)
            global_dict[key] = head.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
            if use_personalized:
                for cid in client_ids:
                    personalized_states[int(cid)][key] = head.detach().cpu().clone()
            continue

        if not (is_lora_a_param_name(key) and all(key in st for st in client_states)):
            continue

        state = residual_state(client_states, client_row_importance, key, global_dict, args)
        ref = state["ref"]
        residuals = state["residuals"]
        importances = state["importances"]
        active = state["active"]
        conflict = state["conflict_raw"]

        # Soft sigmoid gate (no hard quantile threshold)
        gate = soft_sigmoid_gate(conflict, kappa=kappa)
        if (~active).any():
            gate = gate.clone()
            gate[~active] = 0.0
        if power != 1.0:
            gate = gate.clamp(0.0, 1.0).pow(power)
        lambda_g = lam_min + (lam_max - lam_min) * gate

        # Common residual: all clients, importance + sample size weighted
        R_common = weighted_row_mean(
            residuals,
            importances,
            weights,
            list(range(len(residuals))),
        )

        # Default A_down: by design uses unweighted full mean, so personalized != default
        if default_uniform:
            uniform_imp = [torch.ones_like(importances[0]) for _ in importances]
            uniform_weights = np.ones(len(residuals), dtype=np.float64) / max(len(residuals), 1)
            R_default = weighted_row_mean(
                residuals,
                uniform_imp,
                uniform_weights,
                list(range(len(residuals))),
            )
        else:
            R_default = R_common
        default_A = ref + (1.0 - gate) * R_common + lambda_g * gate * R_default
        global_dict[key] = default_A.to(
            device=global_dict[key].device, dtype=global_dict[key].dtype
        )

        # Per-cluster residual
        cluster_residuals = {
            cid: weighted_row_mean(residuals, importances, weights, members)
            for cid, members in clusters.items()
        }

        if use_personalized:
            for idx, client_id in enumerate(client_ids):
                cid = client_to_cluster.get(idx, idx)
                R_cluster = cluster_residuals.get(cid, R_common)
                A_p = ref + (1.0 - gate) * R_common + lambda_g * gate * R_cluster
                personalized_states[int(client_id)][key] = A_p.detach().cpu().clone()

        layer_stats[key] = {
            "num_rows": int(ref.shape[0]),
            "mean_conflict": float(conflict.mean().item()),
            "max_conflict": float(conflict.max().item()),
            "mean_gate": float(gate.mean().item()),
            "mean_lambda_g": float(lambda_g.mean().item()),
            "mean_residual_norm": float(state["norm_acc"].mean().item()),
            "mean_common_norm": float(torch.linalg.vector_norm(R_common).item()),
            "mean_cluster_norm": float(
                np.mean([
                    float(torch.linalg.vector_norm(R_cluster).item())
                    for R_cluster in cluster_residuals.values()
                ])
            ),
        }

    stats = dict(layer_stats)
    stats["_summary"] = _summary(layer_stats, variant_tag, num_clusters=len(clusters))
    setattr(args, "_fedplora_v4_stats", stats)
    setattr(args, "_fedplora_v4_client_clusters", client_to_cluster)
    setattr(args, "_fedplora_personalized_shared_states", personalized_states if use_personalized else {})

    global_model.load_state_dict(global_dict)
    return global_model


# Convenience entry points keyed by agg_type
def aggregate_models_v4_hier_soft_prior(global_model, client_uploads, args):
    setattr(args, "v4_cluster_mode", "prior")
    return aggregate_models_v4_hier(global_model, client_uploads, args, variant_tag="hier_soft_prior")


def aggregate_models_v4_hier_soft_spectral(global_model, client_uploads, args):
    setattr(args, "v4_cluster_mode", "spectral")
    return aggregate_models_v4_hier(global_model, client_uploads, args, variant_tag="hier_soft_spectral")


def aggregate_models_v4_hier_soft_pfl_eval(global_model, client_uploads, args):
    """A3: same as A1 with explicit personalized eval on by default (used as the bug-fix demo)."""
    setattr(args, "v4_cluster_mode", "prior")
    setattr(args, "v4_personalized_eval", True)
    setattr(args, "v4_default_uniform", True)
    return aggregate_models_v4_hier(global_model, client_uploads, args, variant_tag="hier_soft_pfl_eval")
