"""FedPLoRA v6: Domain Conflict Resolution via Subspace Consensus.

DCR is an A-only, one-shot aggregator. It avoids entry-wise averaging of LoRA A
under gauge ambiguity by aggregating row subspaces:

1. For each domain, stack client A matrices and take the top-r right singular
   vectors as a domain consensus subspace.
2. Stack domain bases and take the top-r global directions as the cross-domain
   consensus subspace.
3. Downlink either one global A (``v6_dcr_global``) or one domain-specific A
   per client (``v6_dcr_domain``).

The implementation writes personalized A_down tensors to
``args._fedplora_personalized_shared_states``; the training/eval loop broadcasts
those per-client shared states while B remains local.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import torch

from methods import common as M
from methods.fedplora_oneshot import _client_weights, _metadata_from_uploads
from utilities.utils import is_lora_a_param_name, is_task_head_param_name


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


def _norm_agg(agg_type):
    return str(agg_type or "").strip().lower().replace("-", "_")


def _task_head_average(global_dict, key, client_states, weights):
    return sum(float(weights[i]) * client_states[i][key].float() for i in range(len(client_states)))


def _make_personalized_states(client_ids):
    return {int(client_id): {} for client_id in client_ids}


def _merge_task_head_into_personalized(personalized_states, key, value):
    for state in personalized_states.values():
        state[key] = value.detach().cpu().clone()


def _orthonormal_rows(mat, eps=1e-8):
    """Return an r x d orthonormal-row basis spanning the rows of mat."""
    mat = mat.detach().cpu().float()
    if mat.numel() == 0:
        return mat.contiguous()
    try:
        q, _ = torch.linalg.qr(mat.transpose(0, 1), mode="reduced")
        out = q.transpose(0, 1).contiguous()
    except RuntimeError:
        out = mat / torch.linalg.vector_norm(mat, dim=1, keepdim=True).clamp_min(eps)
    return out[: mat.shape[0]].contiguous()


def _top_right_basis(stacked, rank):
    """Top-r right singular vectors of stacked row data, returned as r x d."""
    stacked = stacked.detach().cpu().float()
    rank = max(1, min(int(rank), min(stacked.shape)))
    try:
        _u, s, vh = torch.linalg.svd(stacked, full_matrices=False)
        basis = vh[:rank].contiguous()
    except RuntimeError:
        basis = _orthonormal_rows(stacked)[:rank].contiguous()
        s = torch.linalg.vector_norm(stacked, dim=1).sort(descending=True).values
    return basis, s.detach().cpu().float()


def _energy_rank(s, max_rank, tau):
    if s is None or s.numel() == 0:
        return max(1, int(max_rank))
    vals = s.detach().cpu().float().pow(2)
    total = float(vals.sum().item())
    if total <= 0:
        return max(1, int(max_rank))
    target = max(0.0, min(float(tau), 1.0)) * total
    csum = torch.cumsum(vals, dim=0)
    idx = int(torch.searchsorted(csum, torch.tensor(target), right=False).item()) + 1
    return max(1, min(int(max_rank), idx))


def _weighted_rows(A, importance=None, power=0.0, clip=0.0):
    """Optional row-importance weighting before subspace SVD."""
    A = A.detach().cpu().float()
    if importance is None or float(power) == 0.0:
        return A
    imp = importance.detach().cpu().float().reshape(-1)
    if imp.numel() != A.shape[0]:
        return A
    if clip and float(clip) > 0:
        imp = imp.clamp(max=float(clip))
    scale = imp.clamp_min(1e-8).pow(float(power)).sqrt().view(-1, 1)
    return A * scale


def _principal_angle_stats(bases):
    """Pairwise principal-angle diagnostics between orthonormal row bases."""
    domains = sorted(bases.keys())
    pairs = {}
    cos_means = []
    angle_means = []
    for d1, d2 in combinations(domains, 2):
        B1 = bases[d1].float()
        B2 = bases[d2].float()
        try:
            singular = torch.linalg.svdvals(B1 @ B2.transpose(0, 1)).clamp(0.0, 1.0)
        except RuntimeError:
            singular = torch.empty(0, dtype=torch.float32)
        if singular.numel() == 0:
            mean_cos = float("nan")
            mean_angle = float("nan")
        else:
            mean_cos = float(singular.mean().item())
            mean_angle = float(torch.rad2deg(torch.arccos(singular)).mean().item())
            cos_means.append(mean_cos)
            angle_means.append(mean_angle)
        pairs[f"{d1}__{d2}"] = {
            "mean_cos": mean_cos,
            "mean_angle_deg": mean_angle,
        }
    return {
        "pairs": pairs,
        "mean_pair_cos": float(np.mean(cos_means)) if cos_means else float("nan"),
        "mean_pair_angle_deg": float(np.mean(angle_means)) if angle_means else float("nan"),
    }


def _residual_basis(domain_basis, shared_basis, residual_rank):
    residual_rank = max(0, int(residual_rank))
    if residual_rank <= 0:
        return domain_basis.new_empty((0, domain_basis.shape[1]))
    if shared_basis.numel() == 0:
        return domain_basis[:residual_rank].contiguous()
    residual = domain_basis - (domain_basis @ shared_basis.transpose(0, 1)) @ shared_basis
    residual = residual[torch.linalg.vector_norm(residual, dim=1) > 1e-7]
    if residual.numel() == 0:
        return domain_basis.new_empty((0, domain_basis.shape[1]))
    rb = _orthonormal_rows(residual)
    return rb[:residual_rank].contiguous()


def _fill_to_rank(candidate, fallback, rank):
    """Orthonormalize candidate and append fallback/global rows if rank is short."""
    parts = []
    if candidate.numel() > 0:
        parts.append(candidate)
    if fallback.numel() > 0:
        parts.append(fallback)
    merged = torch.cat(parts, dim=0) if parts else fallback.new_zeros((rank, fallback.shape[1]))
    out = _orthonormal_rows(merged)
    if out.shape[0] >= rank:
        return out[:rank].contiguous()
    # Extremely degenerate case: repeat zero-padded rows then orthonormalize if possible.
    pad = torch.zeros((rank - out.shape[0], out.shape[1]), dtype=out.dtype)
    return torch.cat([out, pad], dim=0).contiguous()


def _shared_rank_for_domain(domain_basis, global_basis, global_spectrum, rank, args):
    policy = str(getattr(args, "v6_dcr_rc_policy", "auto") or "auto").lower()
    min_shared = max(0, min(rank, _int_arg(args, "v6_dcr_min_shared_rank", 1)))
    max_raw = _int_arg(args, "v6_dcr_max_shared_rank", rank)
    if max_raw <= 0:
        max_raw = rank
    max_shared = max(min_shared, min(rank, max_raw))

    if policy == "fixed":
        rc = _int_arg(args, "v6_dcr_shared_rank", max(1, rank // 2))
        if rc <= 0:
            rc = max(1, rank // 2)
        return max(min_shared, min(max_shared, rc))
    if policy == "energy":
        rc = _energy_rank(global_spectrum, rank, _float_arg(args, "v6_dcr_energy_tau", 0.80))
        return max(min_shared, min(max_shared, rc))

    # Auto: combine global spectrum with domain/global alignment. High-conflict
    # domains get fewer shared directions and more domain residual directions.
    base = _energy_rank(global_spectrum, rank, _float_arg(args, "v6_dcr_energy_tau", 0.80))
    try:
        cos = torch.linalg.svdvals(
            domain_basis @ global_basis[:rank].transpose(0, 1)
        ).clamp(0.0, 1.0)
        align = float(cos.mean().item()) if cos.numel() else 0.0
    except RuntimeError:
        align = 0.5
    conflict = 1.0 - max(0.0, min(1.0, align))
    strength = max(0.0, min(1.0, _float_arg(args, "v6_dcr_conflict_strength", 1.0)))
    rc = int(round(float(base) * (1.0 - strength * conflict)))
    return max(min_shared, min(max_shared, rc))


def _summary(layer_stats, mode):
    if not layer_stats:
        return {"variant": "dcr", "mode": mode}

    def mean(field):
        vals = [
            float(v.get(field, float("nan")))
            for v in layer_stats.values()
            if field in v and not np.isnan(float(v.get(field, float("nan"))))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    rc_vals = []
    for v in layer_stats.values():
        rc_vals.extend(float(x) for x in (v.get("domain_shared_ranks", {}) or {}).values())

    return {
        "variant": "dcr",
        "mode": mode,
        "num_lora_a_matrices": int(len(layer_stats)),
        "num_domains_mean": mean("num_domains"),
        "mean_domain_pair_cos": mean("mean_domain_pair_cos"),
        "mean_domain_pair_angle_deg": mean("mean_domain_pair_angle_deg"),
        "mean_global_energy_top_r": mean("global_energy_top_r"),
        "mean_shared_rank": float(np.mean(rc_vals)) if rc_vals else float("nan"),
    }


def _aggregate_layer_dcr(key, global_dict, client_states, client_row_importance, weights, domains, args):
    rank = int(global_dict[key].shape[0])
    imp_power = _float_arg(args, "v6_dcr_importance_power", 0.0)
    imp_clip = _float_arg(args, "v6_dcr_importance_clip", 5.0)
    domain_members = defaultdict(list)
    for idx, domain in enumerate(domains):
        domain_members[str(domain or "unknown").lower()].append(idx)

    domain_basis = {}
    domain_spectrum = {}
    domain_member_counts = {}
    for domain, indices in sorted(domain_members.items()):
        rows = []
        for idx in indices:
            imp = client_row_importance[idx].get(key, None)
            rows.append(_weighted_rows(client_states[idx][key], imp, imp_power, imp_clip))
        stacked = torch.cat(rows, dim=0)
        basis, spectrum = _top_right_basis(stacked, rank)
        domain_basis[domain] = _orthonormal_rows(basis)[:rank].contiguous()
        domain_spectrum[domain] = spectrum
        domain_member_counts[domain] = int(len(indices))

    stacked_domains = torch.cat([domain_basis[d] for d in sorted(domain_basis)], dim=0)
    global_basis, global_spectrum = _top_right_basis(stacked_domains, rank)
    global_basis = _orthonormal_rows(global_basis)[:rank].contiguous()

    mode = str(getattr(args, "v6_dcr_mode", "") or "").lower()
    if not mode:
        mode = "domain" if _norm_agg(getattr(args, "agg_type", "")) in {"v6_dcr_domain", "fedplora_dcr_domain"} else "global"
    use_domain = mode in {"domain", "personalized", "per_domain"}

    A_by_domain = {}
    shared_ranks = {}
    domain_alignment = {}
    if use_domain:
        for domain, basis in domain_basis.items():
            r_c = _shared_rank_for_domain(basis, global_basis, global_spectrum, rank, args)
            shared = global_basis[:r_c].contiguous()
            residual = _residual_basis(basis, shared, rank - r_c)
            candidate = torch.cat([shared, residual], dim=0) if residual.numel() else shared
            A_by_domain[domain] = _fill_to_rank(candidate, global_basis, rank)
            shared_ranks[domain] = int(r_c)
            try:
                cos = torch.linalg.svdvals(basis @ global_basis.transpose(0, 1)).clamp(0.0, 1.0)
                domain_alignment[domain] = float(cos.mean().item()) if cos.numel() else float("nan")
            except RuntimeError:
                domain_alignment[domain] = float("nan")

    pair_stats = _principal_angle_stats(domain_basis)
    total_energy = float(global_spectrum.pow(2).sum().item()) if global_spectrum.numel() else 0.0
    top_energy = float(global_spectrum[:rank].pow(2).sum().item()) if global_spectrum.numel() else 0.0
    layer_stat = {
        "num_domains": int(len(domain_basis)),
        "domain_member_counts": domain_member_counts,
        "domain_spectrum_top": {
            d: [float(x) for x in spec[: min(8, spec.numel())].tolist()]
            for d, spec in domain_spectrum.items()
        },
        "global_spectrum_top": [float(x) for x in global_spectrum[: min(16, global_spectrum.numel())].tolist()],
        "global_energy_top_r": float(top_energy / total_energy) if total_energy > 0 else float("nan"),
        "mean_domain_pair_cos": pair_stats["mean_pair_cos"],
        "mean_domain_pair_angle_deg": pair_stats["mean_pair_angle_deg"],
        "principal_angle_pairs": pair_stats["pairs"],
        "domain_shared_ranks": shared_ranks,
        "domain_global_alignment": domain_alignment,
    }
    return global_basis, A_by_domain, layer_stat


def aggregate_models_fedplora_dcr(global_model, client_uploads, args, mode=None):
    """Aggregate LoRA A with DCR and update global/personalized shared state."""
    global_dict = global_model.state_dict()
    client_states = [M.upload_package_state(m) for m in client_uploads]
    client_row_importance = [M.upload_package_row_importance(m) for m in client_uploads]
    if not client_states:
        return global_model

    if mode is not None:
        setattr(args, "v6_dcr_mode", str(mode))
    mode = str(getattr(args, "v6_dcr_mode", "") or "").lower()
    if not mode:
        mode = "domain" if _norm_agg(getattr(args, "agg_type", "")) in {"v6_dcr_domain", "fedplora_dcr_domain"} else "global"

    weights = _client_weights(client_uploads, args)
    client_ids, domains = _metadata_from_uploads(client_uploads, args)
    personalized_states = _make_personalized_states(client_ids) if mode in {"domain", "personalized", "per_domain"} else {}
    layer_stats = {}

    for key in list(global_dict.keys()):
        if is_task_head_param_name(key) and all(key in state for state in client_states):
            head = _task_head_average(global_dict, key, client_states, weights)
            global_dict[key] = head.to(device=global_dict[key].device, dtype=global_dict[key].dtype)
            if personalized_states:
                _merge_task_head_into_personalized(personalized_states, key, global_dict[key])
            continue

        if not (is_lora_a_param_name(key) and all(key in state for state in client_states)):
            continue

        global_basis, A_by_domain, stat = _aggregate_layer_dcr(
            key, global_dict, client_states, client_row_importance, weights, domains, args
        )
        global_dict[key] = global_basis.to(device=global_dict[key].device, dtype=global_dict[key].dtype)
        if personalized_states:
            for idx, client_id in enumerate(client_ids):
                domain = str(domains[idx] or "unknown").lower()
                A_down = A_by_domain.get(domain, global_basis)
                personalized_states[int(client_id)][key] = A_down.detach().cpu().clone()
        layer_stats[key] = stat

    stats = dict(layer_stats)
    stats["_summary"] = _summary(layer_stats, mode)
    setattr(args, "_fedplora_v6_dcr_stats", stats)
    setattr(args, "_fedplora_personalized_shared_states", personalized_states)
    setattr(
        args,
        "_fedplora_v6_client_domains",
        {int(client_ids[i]): str(domains[i] or "unknown").lower() for i in range(len(client_ids))},
    )

    global_model.load_state_dict(global_dict)
    return global_model
