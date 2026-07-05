"""FedPLoRA-v11a: relaxed A-correction branch.

This branch exists to test the 2026-07-05 diagnosis that v10's A correction was
too conservative.  The algorithm keeps the v10 routed-B structure but moves the
implementation to v11 and uses a true low-rank A-delta payload.  Strength is
controlled by existing v10 flags, e.g. ``--v10_a_correction_alpha``,
``--v10_a_anchor_lambda``, ``--v10_a_prox_lambda`` and
``--v10_a_norm_clip_ratio``.
"""

from __future__ import annotations

from methods.lora_expert_baselines import aggregate_models_lora_expert_baseline

from .v11_common import (
    aggregate_shared_a_correction,
    build_v11_upload_package,
    client_weights,
    norm_agg,
    reconstruct_client_states_for_a,
)


SUPPORTED_V11A_AGGS = {
    "fedplora_v11a_relaxed_a",
    "fedplora_v11a",
    "v11a_relaxed_a",
    "v11a",
}


def is_fedplora_v11a_agg(agg_type) -> bool:
    return norm_agg(agg_type) in SUPPORTED_V11A_AGGS


def build_fedplora_v11a_upload_package(*args, **kwargs):
    return build_v11_upload_package(*args, **kwargs)


def aggregate_models_fedplora_v11a(global_model, client_uploads, args):
    client_states_for_a = reconstruct_client_states_for_a(client_uploads, args)
    if not client_states_for_a:
        return global_model

    weights = client_weights(client_uploads, args)
    global_model = aggregate_models_lora_expert_baseline(global_model, client_uploads, args)
    global_dict = global_model.state_dict()
    a_stats = aggregate_shared_a_correction(global_dict, client_states_for_a, weights, args)
    global_model.load_state_dict(global_dict)

    stats = getattr(args, "_lora_expert_stats", {}) or {}
    summary = stats.setdefault("_summary", {})
    summary.update(
        {
            "algorithm": norm_agg(getattr(args, "agg_type", "")) or "fedplora_v11a_relaxed_a",
            "v11_branch": "A_relaxation",
            "v11_a_payload": "true_lowrank_delta",
            "v11_b_branch": summary.get("cluster_mode", "b_subspace_auto"),
            "v11_training_rule": "train_A+B_with_configurable_A_anchor_B_prox",
            **a_stats,
        }
    )
    stats["algorithm"] = summary["algorithm"]
    args._lora_expert_stats = stats
    return global_model

