"""
FedPLoRA-Oneshot family.

This module keeps the original v2 entry point compatible with the existing
training script and adds three v3 variants:
- v3-lite: residual-space conflict gate with a single global A downlink.
- v3-cluster: common residual + domain-cluster residual personalized A.
- v3-rpca: low-rank common + sparse residual cluster personalization.
"""

from collections import defaultdict

import numpy as np
import torch

from methods import common as M
from methods.fedplora_oneshotv2 import aggregate_models_fedplora_oneshot
from utilities.utils import is_lora_a_param_name, is_task_head_param_name


_DOMAIN_PRIOR_CLUSTERS = {
    "general": "general_education",
    "education": "general_education",
    "math": "capability",
    "code": "capability",
    "medical": "risk",
    "legal": "risk",
    "finance": "risk",
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


def _clip(value, low, high):
    return max(low, min(high, value))


def _safe_quantile(values, q, default):
    if values is None or values.numel() == 0:
        return torch.tensor(default, dtype=torch.float32)
    q = _clip(float(q), 0.0, 1.0)
    values = values.detach().float().reshape(-1)
    if values.numel() == 1:
        return values[0]
    try:
        return torch.quantile(values, q)
    except Exception:
        sorted_values, _ = torch.sort(values)
        idx = int(round(q * float(sorted_values.numel() - 1)))
        return sorted_values[idx]


def _client_weights(client_uploads, args):
    client_sizes = [M.upload_package_client_size(m) for m in client_uploads]
    if all(x is None for x in client_sizes):
        client_sizes = getattr(args, "_fedplora_client_sizes", None)
    if client_sizes is None:
        return np.ones(len(client_uploads), dtype=np.float64) / max(len(client_uploads), 1)
    sizes = np.asarray(client_sizes, dtype=np.float64)
    total = float(sizes.sum())
    if total <= 0:
        return np.ones(len(client_uploads), dtype=np.float64) / max(len(client_uploads), 1)
    return sizes / total


def _metadata_from_uploads(client_uploads, args):
    runtime_ids = list(getattr(args, "_fedplora_round_client_ids", []) or [])
    domain_map = getattr(args, "_fedplora_client_domains", {}) or {}
    client_ids = []
    domains = []
    for idx, upload in enumerate(client_uploads):
        client_id = None
        domain = None
        if isinstance(upload, dict):
            client_id = upload.get("client_id", None)
            domain = upload.get("domain", None)
        if client_id is None and idx < len(runtime_ids):
            client_id = runtime_ids[idx]
        if client_id is None:
            client_id = idx
        try:
            client_id = int(client_id)
        except (TypeError, ValueError):
            client_id = idx
        if domain is None:
            domain = domain_map.get(client_id, domain_map.get(str(client_id), "unknown"))
        domain = str(domain or "unknown").strip().lower()
        client_ids.append(client_id)
        domains.append(domain)
    return client_ids, domains


def _parse_domain_cluster_map(raw):
    mapping = {}
    for item in str(raw or "").split(","):
        if ":" not in item:
            continue
        domain, cluster = item.split(":", 1)
        domain = domain.strip().lower()
        cluster = cluster.strip().lower()
        if domain and cluster:
            mapping[domain] = cluster
    return mapping


def _domain_to_cluster(domain, args):
    mode = str(getattr(args, "v3_cluster_mode", "domain_prior") or "domain_prior").lower()
    domain = str(domain or "unknown").strip().lower()
    if mode in {"none", "off", "global"}:
        return "global"
    if mode in {"domain", "domain_id", "per_domain"}:
        return domain or "unknown"
    custom = _parse_domain_cluster_map(getattr(args, "v3_domain_cluster_map", ""))
    if domain in custom:
        return custom[domain]
    if mode in {"domain_prior", "prior", "default"}:
        return _DOMAIN_PRIOR_CLUSTERS.get(domain, domain or "unknown")
    return domain or "unknown"


def _cluster_indices(domains, args):
    clusters = defaultdict(list)
    for idx, domain in enumerate(domains):
        clusters[_domain_to_cluster(domain, args)].append(idx)
    return dict(clusters)


def _aggregate_summary(layer_stats, variant, num_clusters=0):
    total_rows = sum(int(v.get("num_rows", 0)) for v in layer_stats.values())
    if total_rows <= 0:
        return {"variant": variant, "num_clusters": int(num_clusters)}

    def wmean(key):
        return float(
            sum(float(v.get(key, 0.0)) * int(v.get("num_rows", 0)) for v in layer_stats.values())
            / total_rows
        )

    return {
        "variant": variant,
        "num_lora_a_matrices": int(len(layer_stats)),
        "num_rows": int(total_rows),
        "num_clusters": int(num_clusters),
        "mean_conflict": wmean("mean_conflict"),
        "max_conflict": float(max(float(v.get("max_conflict", 0.0)) for v in layer_stats.values())),
        "high_conflict_row_frac": wmean("high_conflict_row_frac"),
        "mean_gate": wmean("mean_gate"),
        "mean_threshold": wmean("threshold"),
        "mean_residual_norm": wmean("mean_residual_norm"),
        "mean_common_ratio": wmean("mean_common_ratio"),
        "mean_sparse_ratio": wmean("mean_sparse_ratio"),
    }


def _task_head_average(global_dict, key, client_states, weights):
    return sum(float(weights[i]) * client_states[i][key].float() for i in range(len(client_states)))


def _row_importance(client_row_importance, idx, key, num_rows, args):
    imp = client_row_importance[idx].get(key, None)
    if imp is None:
        imp = torch.ones(num_rows, dtype=torch.float32)
    else:
        imp = imp.detach().cpu().float()
        if imp.numel() != num_rows:
            imp = torch.ones(num_rows, dtype=torch.float32)
    imp = imp.view(-1, 1).clamp_min(1e-8)
    clip = _float_arg(args, "oneshot_importance_clip", 5.0)
    if clip > 0:
        imp = imp.clamp(max=clip)
    power = _float_arg(args, "oneshot_importance_power", 1.0)
    if power != 1.0:
        imp = imp.pow(power)
    return imp


def _layer_residual_state(key, global_dict, client_states, client_row_importance, weights, args):
    initial_A = getattr(args, "_fedplora_initial_A", None)
    if not isinstance(initial_A, dict):
        initial_A = {}

    mats = [state[key].detach().cpu().float() for state in client_states]
    ref = initial_A.get(key, None)
    if ref is None or tuple(ref.shape) != tuple(mats[0].shape):
        ref = global_dict[key].detach().cpu().float()
    else:
        ref = ref.detach().cpu().float()

    eps = 1e-8
    residual_norm_power = _float_arg(args, "v3_residual_norm_power", 1.0)
    residual_eps = _float_arg(args, "v3_residual_eps", 1e-7)
    q = _float_arg(args, "v3_conflict_quantile", 0.80)
    temperature = max(_float_arg(args, "v3_gate_temperature", 0.05), eps)

    residuals = [Ai - ref for Ai in mats]
    num_rows = ref.shape[0]
    row_weights = []
    residual_acc = torch.zeros_like(ref, dtype=torch.float32)
    dir_acc = torch.zeros_like(ref, dtype=torch.float32)
    weight_sum = torch.zeros((num_rows, 1), dtype=torch.float32)
    norm_acc = torch.zeros((num_rows, 1), dtype=torch.float32)

    for idx, residual in enumerate(residuals):
        norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
        direction = residual / norm.clamp_min(eps)
        imp = _row_importance(client_row_importance, idx, key, num_rows, args)
        norm_factor = norm.clamp_min(eps)
        if residual_norm_power != 1.0:
            norm_factor = norm_factor.pow(residual_norm_power)
        row_weight = float(weights[idx]) * imp * norm_factor
        row_weights.append(row_weight)
        residual_acc = residual_acc + row_weight * residual
        dir_acc = dir_acc + row_weight * direction
        weight_sum = weight_sum + row_weight
        norm_acc = norm_acc + float(weights[idx]) * norm

    safe_weight = weight_sum.clamp_min(eps)
    global_residual = residual_acc / safe_weight
    mean_dir = dir_acc / safe_weight
    consensus = torch.linalg.vector_norm(mean_dir, dim=1, keepdim=True).clamp(0.0, 1.0)
    active = (norm_acc.squeeze(1) > residual_eps) & (weight_sum.squeeze(1) > eps)
    conflict = (1.0 - consensus).clamp(0.0, 1.0)
    if bool((~active).any()):
        conflict = conflict.clone()
        conflict[~active] = 0.0

    threshold = _safe_quantile(conflict[active], q, default=1.0) if bool(active.any()) else torch.tensor(1.0)
    raw_gate = torch.sigmoid((conflict - threshold) / temperature)
    gate = torch.where(conflict > threshold, raw_gate, torch.zeros_like(raw_gate))
    if bool((~active).any()):
        gate = gate.clone()
        gate[~active] = 0.0

    high_frac = float(((conflict.squeeze(1) > threshold) & active).float().mean().item())
    mean_residual_norm = float(norm_acc.mean().item())

    return {
        "ref": ref,
        "residuals": residuals,
        "row_weights": row_weights,
        "global_residual": global_residual,
        "conflict": conflict,
        "gate": gate,
        "threshold": float(threshold.item()),
        "active": active,
        "mean_residual_norm": mean_residual_norm,
        "stats": {
            "num_rows": int(num_rows),
            "mean_conflict": float(conflict.mean().item()),
            "max_conflict": float(conflict.max().item()),
            "high_conflict_row_frac": high_frac,
            "mean_gate": float(gate.mean().item()),
            "threshold": float(threshold.item()),
            "mean_residual_norm": mean_residual_norm,
            "mean_common_ratio": 1.0,
            "mean_sparse_ratio": 0.0,
        },
    }


def _weighted_residual_mean(residuals, row_weights, indices):
    if not indices:
        return torch.zeros_like(residuals[0])
    acc = torch.zeros_like(residuals[0], dtype=torch.float32)
    weight_sum = torch.zeros((residuals[0].shape[0], 1), dtype=torch.float32)
    for idx in indices:
        acc = acc + row_weights[idx] * residuals[idx]
        weight_sum = weight_sum + row_weights[idx]
    return acc / weight_sum.clamp_min(1e-8)


def _rpca_common_sparse(residuals, args):
    num_clients = len(residuals)
    shape = residuals[0].shape
    flat = torch.stack([r.reshape(-1) for r in residuals], dim=0).float()
    rank = _int_arg(args, "v3_rpca_rank", 1)
    rank = max(0, min(rank, flat.shape[0], flat.shape[1]))
    if rank <= 0 or num_clients <= 1:
        common = torch.zeros_like(flat)
        sparse = flat
    else:
        try:
            u, s, vh = torch.linalg.svd(flat, full_matrices=False)
            common = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
        except RuntimeError:
            common = flat.mean(dim=0, keepdim=True).repeat(num_clients, 1)
        sparse = flat - common

    sparse_q = _float_arg(args, "v3_sparse_quantile", 0.80)
    abs_sparse = sparse.abs().reshape(-1)
    threshold = _safe_quantile(abs_sparse, sparse_q, default=0.0).clamp_min(1e-12)
    mask = sparse.abs() >= threshold
    sparse = torch.where(mask, sparse, torch.zeros_like(sparse))

    common_residuals = [common[i].reshape(shape).contiguous() for i in range(num_clients)]
    sparse_residuals = [sparse[i].reshape(shape).contiguous() for i in range(num_clients)]
    sparse_ratio = float(mask.float().mean().item()) if mask.numel() else 0.0
    common_norm = float(torch.linalg.vector_norm(common).item())
    total_norm = float(torch.linalg.vector_norm(flat).item()) + 1e-8
    common_ratio = common_norm / total_norm
    return common_residuals, sparse_residuals, common_ratio, sparse_ratio


def _make_personalized_states(client_ids):
    return {int(client_id): {} for client_id in client_ids}


def _merge_task_head_into_personalized(personalized_states, key, value):
    for state in personalized_states.values():
        state[key] = value.detach().cpu().clone()


def _lambda_row(gate, args):
    lam_min = _float_arg(args, "v3_cluster_lambda_min", 0.2)
    lam_max = _float_arg(args, "v3_cluster_lambda_max", 1.0)
    if lam_max < lam_min:
        lam_min, lam_max = lam_max, lam_min
    return lam_min + (lam_max - lam_min) * gate


def _aggregate_v3(global_model, client_uploads, args, variant):
    global_dict = global_model.state_dict()
    client_states = [M.upload_package_state(m) for m in client_uploads]
    client_row_importance = [M.upload_package_row_importance(m) for m in client_uploads]
    if not client_states:
        return global_model

    weights = _client_weights(client_uploads, args)
    client_ids, domains = _metadata_from_uploads(client_uploads, args)
    clusters = _cluster_indices(domains, args)
    client_cluster = {
        int(client_ids[idx]): _domain_to_cluster(domains[idx], args)
        for idx in range(len(client_ids))
    }
    use_personalized = variant in {"cluster", "rpca"}
    personalized_states = _make_personalized_states(client_ids) if use_personalized else {}
    layer_stats = {}
    blend = _clip(_float_arg(args, "v3_conflict_blend", 1.0), 0.0, 1.0)

    for key in list(global_dict.keys()):
        if is_task_head_param_name(key) and all(key in state for state in client_states):
            agg_head = _task_head_average(global_dict, key, client_states, weights)
            global_dict[key] = agg_head.to(device=global_dict[key].device, dtype=global_dict[key].dtype)
            if use_personalized:
                _merge_task_head_into_personalized(personalized_states, key, global_dict[key])
            continue

        if not (is_lora_a_param_name(key) and all(key in state for state in client_states)):
            continue

        state = _layer_residual_state(
            key, global_dict, client_states, client_row_importance, weights, args
        )
        ref = state["ref"]
        gate = state["gate"]
        global_residual = state["global_residual"]

        if variant == "lite":
            candidate = ref + (1.0 - blend * gate) * global_residual
            layer_stats[key] = dict(state["stats"])
            global_dict[key] = candidate.to(device=global_dict[key].device, dtype=global_dict[key].dtype)
            continue

        low_conflict_common = (1.0 - gate) * global_residual
        lambda_gate = _lambda_row(gate, args) * gate

        if variant == "rpca":
            common_residuals, sparse_residuals, common_ratio, sparse_ratio = _rpca_common_sparse(
                state["residuals"], args
            )
            rpca_weights = [
                torch.ones((ref.shape[0], 1), dtype=torch.float32) * float(weights[i])
                for i in range(len(common_residuals))
            ]
            common_residual = _weighted_residual_mean(common_residuals, rpca_weights, list(range(len(common_residuals))))
            common_path = (1.0 - gate) * common_residual
            residual_pool = sparse_residuals
            residual_weights = rpca_weights
            stats = dict(state["stats"])
            stats["mean_common_ratio"] = float(common_ratio)
            stats["mean_sparse_ratio"] = float(sparse_ratio)
        else:
            common_path = low_conflict_common
            residual_pool = state["residuals"]
            residual_weights = state["row_weights"]
            stats = dict(state["stats"])
            stats["mean_common_ratio"] = float((1.0 - gate).mean().item())

        default_cluster_residual = _weighted_residual_mean(
            residual_pool, residual_weights, list(range(len(residual_pool)))
        )
        default_candidate = ref + common_path + lambda_gate * default_cluster_residual
        global_dict[key] = default_candidate.to(
            device=global_dict[key].device, dtype=global_dict[key].dtype
        )

        cluster_residuals = {
            cluster: _weighted_residual_mean(residual_pool, residual_weights, indices)
            for cluster, indices in clusters.items()
        }
        for idx, client_id in enumerate(client_ids):
            cluster = client_cluster.get(int(client_id), _domain_to_cluster(domains[idx], args))
            cluster_residual = cluster_residuals.get(cluster, default_cluster_residual)
            personalized_A = ref + common_path + lambda_gate * cluster_residual
            personalized_states[int(client_id)][key] = personalized_A.detach().cpu().clone()

        stats["num_clusters"] = int(len(clusters))
        stats["mean_cluster_gate"] = float(lambda_gate.mean().item())
        layer_stats[key] = stats

    stats = dict(layer_stats)
    stats["_summary"] = _aggregate_summary(layer_stats, variant, num_clusters=len(clusters))
    setattr(args, "_fedplora_v3_stats", stats)
    setattr(args, "_fedplora_v3_client_clusters", client_cluster)
    if use_personalized:
        setattr(args, "_fedplora_personalized_shared_states", personalized_states)
    else:
        setattr(args, "_fedplora_personalized_shared_states", {})

    global_model.load_state_dict(global_dict)
    return global_model


def aggregate_models_fedplora_v3_lite(global_model, client_uploads, args):
    """FedPLoRA-Oneshot v3-lite: residual conflict gate, one global A."""
    return _aggregate_v3(global_model, client_uploads, args, variant="lite")


def aggregate_models_fedplora_v3_cluster(global_model, client_uploads, args):
    """FedPLoRA-Oneshot v3-cluster: common residual + domain-cluster A downlink."""
    return _aggregate_v3(global_model, client_uploads, args, variant="cluster")


def aggregate_models_fedplora_v3_rpca(global_model, client_uploads, args):
    """FedPLoRA-Oneshot v3-rpca: low-rank common + sparse cluster residual."""
    return _aggregate_v3(global_model, client_uploads, args, variant="rpca")
