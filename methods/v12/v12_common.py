"""Shared FedPLoRA-v12 helpers for global-routed B mixing.

v12 keeps v11 frozen and turns the 2026-07-08 analysis into lightweight,
separately testable branches:

* v12a: round-scheduled global-B mixing, for testing whether a low-μ early
  phase preserves B geometry before a higher-μ macro phase.
* v12b: NMI-guarded global-B mixing, a small adaptive controller that backs
  off when the pre-mix B geometry has already collapsed.

Both branches reuse v11's true rank-k A-delta sketch payload.  They do not
change v8/v9/v10/v11 implementations.
"""

from __future__ import annotations

import copy
import math
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch

from methods import common as M
from methods.lora_expert_baselines import (
    _common_lora_keys,
    _metadata_from_uploads,
    _resolve_assignments,
    _stats_for_assignments,
    _weighted_mean_tensors,
    aggregate_models_lora_expert_baseline,
)
from utilities.utils import is_lora_b_param_name

from methods.v11.v11_common import (
    aggregate_shared_a_correction,
    build_v11_upload_package,
    client_weights,
    norm_agg,
    reconstruct_client_states_for_a,
)


def clamp_mu(value, default: float = 0.4) -> float:
    try:
        mu = float(value)
    except Exception:
        mu = float(default)
    return min(1.0, max(0.0, mu))


def _round_progress(args) -> Tuple[int, int, float]:
    """Return zero-based round, total rounds, and [0,1] schedule progress."""
    round_idx = int(getattr(args, "_fedplora_current_round_idx", 0) or 0)
    total = max(1, int(getattr(args, "rounds", 1) or 1))
    warmup = max(0, int(getattr(args, "v12_mu_warmup_rounds", 0) or 0))
    if total <= 1 or round_idx < warmup:
        return round_idx, total, 0.0
    denom = max(1, total - warmup - 1)
    progress = float(round_idx - warmup) / float(denom)
    return round_idx, total, min(1.0, max(0.0, progress))


def scheduled_mu(args, _pre_summary=None) -> Tuple[float, Dict[str, object]]:
    """Resolve v12a current μ from schedule args."""
    start = clamp_mu(getattr(args, "v12_mu_start", 0.4), 0.4)
    end = clamp_mu(getattr(args, "v12_mu_end", 0.6), 0.6)
    schedule = str(getattr(args, "v12_mu_schedule", "linear") or "linear").lower()
    round_idx, total, progress = _round_progress(args)
    if schedule == "cosine":
        t = 0.5 - 0.5 * math.cos(math.pi * progress)
    elif schedule == "step":
        t = 0.0 if progress < 0.5 else 1.0
    elif schedule in {"constant", "fixed"}:
        t = 0.0
    else:
        schedule = "linear"
        t = progress
    mu = clamp_mu(start + (end - start) * t, start)
    return mu, {
        "v12_mu_policy": "round_schedule",
        "v12_mu_schedule": schedule,
        "v12_mu_start": float(start),
        "v12_mu_end": float(end),
        "v12_current_mu": float(mu),
        "v12_round_idx_0based": int(round_idx),
        "v12_round_idx_1based": int(round_idx + 1),
        "v12_total_rounds": int(total),
        "v12_mu_progress": float(progress),
        "v12_mu_warmup_rounds": int(getattr(args, "v12_mu_warmup_rounds", 0) or 0),
    }


def nmi_guard_mu(args, pre_summary=None) -> Tuple[float, Dict[str, object]]:
    """Resolve v12b μ by guarding against pre-mix geometry collapse."""
    pre_summary = pre_summary or {}
    low = clamp_mu(getattr(args, "v12_nmi_guard_low_mu", 0.4), 0.4)
    high = clamp_mu(getattr(args, "v12_nmi_guard_high_mu", 0.55), 0.55)
    threshold = float(getattr(args, "v12_nmi_guard_threshold", 0.75) or 0.75)
    nmi = pre_summary.get("domain_nmi", float("nan"))
    try:
        nmi_f = float(nmi)
    except Exception:
        nmi_f = float("nan")
    use_high = math.isfinite(nmi_f) and nmi_f >= threshold
    mu = high if use_high else low
    return mu, {
        "v12_mu_policy": "nmi_guard",
        "v12_nmi_guard_threshold": float(threshold),
        "v12_nmi_guard_low_mu": float(low),
        "v12_nmi_guard_high_mu": float(high),
        "v12_pre_mix_domain_nmi_for_guard": nmi_f,
        "v12_nmi_guard_used_high_mu": bool(use_high),
        "v12_current_mu": float(mu),
    }


def _global_b_state(
    client_states: Sequence[Dict[str, torch.Tensor]],
    b_keys: Sequence[str],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    global_b: Dict[str, torch.Tensor] = {}
    for key in b_keys:
        tensors = [st[key] for st in client_states if key in st]
        if len(tensors) != len(client_states):
            continue
        global_b[key] = _weighted_mean_tensors(tensors, weights).detach().cpu()
    return global_b


def _geometry_for_states(
    client_ids: Sequence[int],
    domains: Sequence[str],
    states_by_client: Dict[int, Dict[str, torch.Tensor]],
    b_keys: Sequence[str],
    args,
    *,
    algo: str,
) -> Dict[str, object]:
    """Compute B-subspace clustering stats for an arbitrary per-client B map."""
    states = [states_by_client.get(int(cid), {}) for cid in client_ids]
    if not states or not b_keys:
        return {"algorithm": algo, "_summary": {}}
    try:
        # Keep the user's route mode, but avoid accidental mutation of args.
        tmp_args = copy.copy(args)
        assign, _dist, info = _resolve_assignments(states, b_keys, domains, tmp_args)
        return _stats_for_assignments(algo, client_ids, domains, assign, info)
    except Exception as exc:
        return {
            "algorithm": algo,
            "_summary": {"geometry_error": repr(exc)},
            "client_clusters": {},
            "cluster_domain_hist": {},
        }


def _copy_prefixed(summary: Dict[str, object], src: Dict[str, object], prefix: str) -> None:
    for key in (
        "cluster_mode",
        "selected_k",
        "silhouette",
        "mean_pair_angle_deg",
        "mean_pair_cos",
        "domain_nmi",
        "domain_ari",
        "cluster_sizes",
    ):
        if key in src:
            summary[f"{prefix}_{key}"] = src[key]


def aggregate_models_v12_gmix(
    global_model,
    client_uploads,
    args,
    *,
    branch_name: str,
    mu_resolver: Callable[[object, Dict[str, object]], Tuple[float, Dict[str, object]]],
):
    """Aggregate v12 global-routed B mixing with true v11 A-sketch payloads."""
    client_states = [M.upload_package_state(m) for m in client_uploads]
    client_states_for_a = reconstruct_client_states_for_a(client_uploads, args)
    if not client_states:
        return global_model

    client_ids, domains = _metadata_from_uploads(client_uploads, args)
    weights = client_weights(client_uploads, args)
    b_keys = _common_lora_keys(client_states, is_lora_b_param_name)
    global_b = _global_b_state(client_states, b_keys, weights)

    global_model = aggregate_models_lora_expert_baseline(global_model, client_uploads, args)
    stats = getattr(args, "_lora_expert_stats", {}) or {}
    routed_states = getattr(args, "_lora_expert_personalized_local_states", {}) or {}
    pre_summary = dict(stats.get("_summary", {}) or {})

    mu, mu_info = mu_resolver(args, pre_summary)
    mu = clamp_mu(mu)
    mixed_states: Dict[int, Dict[str, torch.Tensor]] = {}
    mix_delta_norm_ratios = []
    global_delta_norm_ratios = []

    for client_id in client_ids:
        cid = int(client_id)
        routed = routed_states.get(cid, {})
        mixed: Dict[str, torch.Tensor] = {}
        for key in b_keys:
            if key not in routed or key not in global_b:
                continue
            routed_b = routed[key].detach().float().cpu()
            global_b_key = global_b[key].detach().float().cpu()
            mixed_b = mu * global_b_key + (1.0 - mu) * routed_b
            mixed[key] = mixed_b.detach().cpu()

            routed_norm = float(torch.linalg.vector_norm(routed_b).item())
            if routed_norm > 1e-12:
                mix_delta_norm_ratios.append(
                    float(torch.linalg.vector_norm(mixed_b - routed_b).item()) / routed_norm
                )
                global_delta_norm_ratios.append(
                    float(torch.linalg.vector_norm(global_b_key - routed_b).item()) / routed_norm
                )
        mixed_states[cid] = mixed

    post_stats = _geometry_for_states(
        client_ids,
        domains,
        mixed_states,
        b_keys,
        args,
        algo=f"{norm_agg(getattr(args, 'agg_type', ''))}_post_mix",
    )
    post_summary = dict(post_stats.get("_summary", {}) or {})

    global_dict = global_model.state_dict()
    for key in b_keys:
        all_b = [st[key] for st in mixed_states.values() if key in st]
        if all_b and key in global_dict:
            merged = torch.stack([b.float() for b in all_b], 0).mean(0)
            global_dict[key] = merged.to(
                device=global_dict[key].device,
                dtype=global_dict[key].dtype,
            )

    a_stats = aggregate_shared_a_correction(global_dict, client_states_for_a, weights, args)
    global_model.load_state_dict(global_dict)

    summary = stats.setdefault("_summary", {})
    summary.update(
        {
            "algorithm": norm_agg(getattr(args, "agg_type", "")) or "fedplora_v12_gmix",
            "v12_branch": branch_name,
            "v11_branch": branch_name,
            "v11_a_payload": "true_lowrank_delta",
            "v11_global_b_mix_mu": float(mu),
            "v12_current_mu": float(mu),
            "v11_global_branch": "sample_size_weighted_b_mean",
            "v11_routed_branch": pre_summary.get("cluster_mode", "b_subspace_auto"),
            "v11_mixing_rule": "B_client=mu*B_global+(1-mu)*B_routed",
            "v12_geometry_diagnostic": "pre_mix_uploads_and_post_mix_downlink_B",
            "v11_mean_mix_delta_norm_ratio": (
                float(np.mean(mix_delta_norm_ratios)) if mix_delta_norm_ratios else 0.0
            ),
            "v11_mean_global_delta_norm_ratio": (
                float(np.mean(global_delta_norm_ratios)) if global_delta_norm_ratios else 0.0
            ),
            "v12_mean_mix_delta_norm_ratio": (
                float(np.mean(mix_delta_norm_ratios)) if mix_delta_norm_ratios else 0.0
            ),
            "v12_mean_global_delta_norm_ratio": (
                float(np.mean(global_delta_norm_ratios)) if global_delta_norm_ratios else 0.0
            ),
            **mu_info,
            **a_stats,
        }
    )
    _copy_prefixed(summary, pre_summary, "v12_pre_mix")
    _copy_prefixed(summary, post_summary, "v12_post_mix")

    stats["algorithm"] = summary["algorithm"]
    stats["v12_post_mix_client_clusters"] = post_stats.get("client_clusters", {})
    stats["v12_post_mix_cluster_domain_hist"] = post_stats.get("cluster_domain_hist", {})
    stats["v12_post_mix_summary"] = post_summary
    args._lora_expert_personalized_local_states = mixed_states
    args._fedplora_personalized_shared_states = {}
    args._lora_expert_stats = stats
    return global_model

