"""FedPLoRA-v9: global/routed dual-branch LoRA-B mixing.

The 2026-06-28 result audit showed a clear split: a single global B pool keeps
domain-macro accuracy higher, while routed B pools keep client-local
personalization higher. v9 materializes both branches in the existing
single-adapter framework by mixing B tensors under a shared A coordinate:

    Delta W_i = (lambda * B_global + (1 - lambda) * B_route_i) A

This is intentionally additive to v8. It reuses the v8/expert clustering
implementation, then replaces the routed client B states with the mixed B
states. Existing v8 behavior is untouched.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from methods import common as M
from methods.lora_expert_baselines import (
    _client_weight_vector,
    _common_lora_keys,
    _metadata_from_uploads,
    _weighted_mean_tensors,
    aggregate_models_lora_expert_baseline,
    build_lora_expert_upload_package,
)
from utilities.utils import is_lora_b_param_name


SUPPORTED_V9_AGGS = {
    "fedplora_v9_mix",
    "fedplora_v9_mix_ab",
    "v9_mix",
    "v9_mix_ab",
}


def _norm_agg(agg_type) -> str:
    return (agg_type or "").strip().lower().replace("-", "_")


def is_fedplora_v9_agg(agg_type) -> bool:
    return _norm_agg(agg_type) in SUPPORTED_V9_AGGS


def build_fedplora_v9_upload_package(*args, **kwargs):
    """Build a v9 client payload.

    Main v9 uses the same B-only upload schedule as v8. The ``*_ab`` variant is
    an A+B high-communication ablation controlled by the training entry point.
    """

    return build_lora_expert_upload_package(*args, **kwargs)


def _clamp_lambda(value) -> float:
    try:
        lam = float(value)
    except Exception:
        lam = 0.5
    return min(1.0, max(0.0, lam))


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


def aggregate_models_fedplora_v9(global_model, client_uploads, args):
    """Aggregate one v9 round and store mixed per-client B states.

    The function first delegates clustering/routing to the v8 expert aggregator.
    It then computes a sample-size weighted global B branch from the same uploads
    and mixes it with each routed B branch. Evaluation and checkpoints keep using
    the existing ``_lora_expert_personalized_local_states`` contract.
    """

    client_states = [M.upload_package_state(m) for m in client_uploads]
    client_ids, _domains = _metadata_from_uploads(client_uploads, args)
    weights = _client_weight_vector(client_uploads, args)
    b_keys = _common_lora_keys(client_states, is_lora_b_param_name)
    global_b = _global_b_state(client_states, b_keys, weights)

    global_model = aggregate_models_lora_expert_baseline(
        global_model, client_uploads, args
    )

    lam = _clamp_lambda(getattr(args, "v9_mix_lambda", 0.5))
    routed_states = getattr(args, "_lora_expert_personalized_local_states", {}) or {}
    mixed_states: Dict[int, Dict[str, torch.Tensor]] = {}
    mix_delta_norm_ratios = []

    for client_id in client_ids:
        cid = int(client_id)
        routed = routed_states.get(cid, {})
        mixed: Dict[str, torch.Tensor] = {}
        for key in b_keys:
            if key not in routed or key not in global_b:
                continue
            routed_b = routed[key].detach().float().cpu()
            global_b_key = global_b[key].detach().float().cpu()
            mixed_b = lam * global_b_key + (1.0 - lam) * routed_b
            mixed[key] = mixed_b.detach().cpu()

            denom = float(torch.linalg.vector_norm(routed_b).item())
            if denom > 1e-12:
                delta = float(torch.linalg.vector_norm(mixed_b - routed_b).item())
                mix_delta_norm_ratios.append(delta / denom)
        mixed_states[cid] = mixed

    # Keep the checkpoint/global model internally consistent with the effective
    # average downlinked B states; actual eval still uses per-client mixed B.
    global_dict = global_model.state_dict()
    for key in b_keys:
        all_b = [st[key] for st in mixed_states.values() if key in st]
        if all_b and key in global_dict:
            merged = torch.stack([b.float() for b in all_b], 0).mean(0)
            global_dict[key] = merged.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
    global_model.load_state_dict(global_dict)

    stats = getattr(args, "_lora_expert_stats", {}) or {}
    summary = stats.setdefault("_summary", {})
    summary.update(
        {
            "algorithm": _norm_agg(getattr(args, "agg_type", "")) or "fedplora_v9_mix",
            "v9_mix_lambda": float(lam),
            "v9_global_branch": "sample_size_weighted_b_mean",
            "v9_routed_branch": summary.get("cluster_mode", "b_subspace_auto"),
            "v9_mixing_rule": "B_client=lambda*B_global+(1-lambda)*B_routed",
            "v9_mean_mix_delta_norm_ratio": (
                float(np.mean(mix_delta_norm_ratios))
                if mix_delta_norm_ratios
                else 0.0
            ),
        }
    )
    stats["algorithm"] = summary["algorithm"]
    args._lora_expert_personalized_local_states = mixed_states
    args._fedplora_personalized_shared_states = {}
    args._lora_expert_stats = stats
    return global_model
