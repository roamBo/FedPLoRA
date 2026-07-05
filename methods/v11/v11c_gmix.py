"""FedPLoRA-v11c: explicit global-B mixing plus controlled A correction.

The 2026-07-05 analysis hypothesizes that v10's gains over v8 came from an
accidental mixed-domain B pool.  v11c makes that mechanism explicit:

    B_client = mu * B_global + (1 - mu) * B_routed

while retaining v11's true sketch A-correction payload.  ``mu`` is controlled by
``--v11_global_b_mix_mu``.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from methods import common as M
from methods.lora_expert_baselines import (
    _common_lora_keys,
    _metadata_from_uploads,
    _weighted_mean_tensors,
    aggregate_models_lora_expert_baseline,
)
from utilities.utils import is_lora_b_param_name

from .v11_common import (
    aggregate_shared_a_correction,
    build_v11_upload_package,
    client_weights,
    norm_agg,
    reconstruct_client_states_for_a,
)


SUPPORTED_V11C_AGGS = {
    "fedplora_v11c_gmix",
    "fedplora_v11_gmix",
    "v11c_gmix",
    "v11_gmix",
}


def is_fedplora_v11c_agg(agg_type) -> bool:
    return norm_agg(agg_type) in SUPPORTED_V11C_AGGS


def build_fedplora_v11c_upload_package(*args, **kwargs):
    return build_v11_upload_package(*args, **kwargs)


def _clamp_mu(value) -> float:
    try:
        mu = float(value)
    except Exception:
        mu = 0.4
    return min(1.0, max(0.0, mu))


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


def aggregate_models_fedplora_v11c(global_model, client_uploads, args):
    client_states = [M.upload_package_state(m) for m in client_uploads]
    client_states_for_a = reconstruct_client_states_for_a(client_uploads, args)
    if not client_states:
        return global_model

    client_ids, _domains = _metadata_from_uploads(client_uploads, args)
    weights = client_weights(client_uploads, args)
    b_keys = _common_lora_keys(client_states, is_lora_b_param_name)
    global_b = _global_b_state(client_states, b_keys, weights)

    global_model = aggregate_models_lora_expert_baseline(global_model, client_uploads, args)
    routed_states = getattr(args, "_lora_expert_personalized_local_states", {}) or {}

    mu = _clamp_mu(getattr(args, "v11_global_b_mix_mu", 0.4))
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

    stats = getattr(args, "_lora_expert_stats", {}) or {}
    summary = stats.setdefault("_summary", {})
    summary.update(
        {
            "algorithm": norm_agg(getattr(args, "agg_type", "")) or "fedplora_v11c_gmix",
            "v11_branch": "global_B_mixing",
            "v11_a_payload": "true_lowrank_delta",
            "v11_global_b_mix_mu": float(mu),
            "v11_global_branch": "sample_size_weighted_b_mean",
            "v11_routed_branch": summary.get("cluster_mode", "b_subspace_auto"),
            "v11_mixing_rule": "B_client=mu*B_global+(1-mu)*B_routed",
            "v11_mean_mix_delta_norm_ratio": (
                float(np.mean(mix_delta_norm_ratios)) if mix_delta_norm_ratios else 0.0
            ),
            "v11_mean_global_delta_norm_ratio": (
                float(np.mean(global_delta_norm_ratios)) if global_delta_norm_ratios else 0.0
            ),
            **a_stats,
        }
    )
    stats["algorithm"] = summary["algorithm"]
    args._lora_expert_personalized_local_states = mixed_states
    args._fedplora_personalized_shared_states = {}
    args._lora_expert_stats = stats
    return global_model

