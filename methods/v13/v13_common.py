"""FedPLoRA-v13 one-shot branches for the 2026-07-11 audit.

The 20260711 one-shot analysis picked two useful, non-overlapping points:

* ``v13a_os``: the main one-shot branch.  It is the seed-42 winner
  (v11a alpha=1.0, no A/B regularizers, no global-B mixing μ) promoted to a
  named algorithm so future reruns do not rely on remembering a fragile flag
  bundle.
* ``v13b_os_bonly``: the low-communication routed-B attribution branch.  It
  keeps the v8 routed-B mechanism, but is named separately so communication
  accounting can use the one-shot/cached-A protocol without changing old v8
  results.

Both branches intentionally reuse the tested v11/v8 implementation paths and
only add thin wrappers plus runtime defaults.
"""

from __future__ import annotations

from typing import Dict

from methods.v8 import (
    aggregate_models_fedplora_v8,
    build_fedplora_v8_upload_package,
)
from methods.v11.v11a_relaxed_a import (
    aggregate_models_fedplora_v11a,
    build_fedplora_v11a_upload_package,
)


SUPPORTED_V13A_OS_AGGS = {
    "fedplora_v13a_os",
    "fedplora_v13a_oneshot",
    "v13a_os",
    "v13a_oneshot",
    "fedplora_os",
    "os_alpha100",
}

SUPPORTED_V13B_BONLY_AGGS = {
    "fedplora_v13b_os_bonly",
    "fedplora_v13b_bonly",
    "v13b_os_bonly",
    "v13b_bonly",
    "fedplora_os_bonly",
    "os_bonly",
}

SUPPORTED_V13_AGGS = SUPPORTED_V13A_OS_AGGS | SUPPORTED_V13B_BONLY_AGGS


def norm_agg(agg_type) -> str:
    return (agg_type or "").strip().lower().replace("-", "_")


def is_fedplora_v13a_os_agg(agg_type) -> bool:
    return norm_agg(agg_type) in SUPPORTED_V13A_OS_AGGS


def is_fedplora_v13b_os_bonly_agg(agg_type) -> bool:
    return norm_agg(agg_type) in SUPPORTED_V13B_BONLY_AGGS


def is_fedplora_v13_agg(agg_type) -> bool:
    return norm_agg(agg_type) in SUPPORTED_V13_AGGS


def _force_default(args, name: str, value) -> None:
    """Set a deterministic algorithm default and record the previous value."""

    prev = getattr(args, name, None)
    if prev != value:
        changes: Dict[str, Dict[str, object]] = getattr(
            args, "_fedplora_v13_forced_defaults", {}
        ) or {}
        changes[name] = {"previous": prev, "value": value}
        args._fedplora_v13_forced_defaults = changes
    setattr(args, name, value)


def apply_fedplora_v13_runtime_defaults(args):
    """Bake the 20260711 one-shot protocol into named v13 algorithms.

    This deliberately overrides legacy defaults, because the purpose of v13 is
    to remove "remember the right flag bundle" from the experiment protocol.
    """

    agg = norm_agg(getattr(args, "agg_type", ""))
    if is_fedplora_v13a_os_agg(agg):
        # Main branch: true A-delta sketch + routed B, no μ, no local geometry
        # regularizers, no norm clipping.  This is the strict v11a alpha100
        # condition from the 20260711 analysis.
        _force_default(args, "v10_a_correction_alpha", 1.0)
        _force_default(args, "v10_a_anchor_lambda", 0.0)
        _force_default(args, "v10_a_prox_lambda", 0.0)
        _force_default(args, "v10_b_prox_lambda", 0.0)
        _force_default(args, "v10_a_norm_clip_ratio", 0.0)
        _force_default(args, "v11_global_b_mix_mu", 0.0)
        _force_default(args, "expert_freeze_a", False)
        if not str(getattr(args, "expert_cluster_mode", "") or "").strip():
            _force_default(args, "expert_cluster_mode", "auto")
        _force_default(args, "expert_top_m", 0)
        args._fedplora_v13_branch = "v13a_os_true_a_sketch_routed_b_no_mu"
    elif is_fedplora_v13b_os_bonly_agg(agg):
        # Attribution branch: cached/shared A and routed-B only.  The new name
        # lets communication accounting reflect that no init-A downlink is paid
        # inside the one-shot routed-B round.
        _force_default(args, "expert_freeze_a", True)
        if not str(getattr(args, "expert_cluster_mode", "") or "").strip():
            _force_default(args, "expert_cluster_mode", "auto")
        _force_default(args, "expert_top_m", 0)
        _force_default(args, "v11_global_b_mix_mu", 0.0)
        args._fedplora_v13_branch = "v13b_os_bonly_cached_a_routed_b"
    return args


def build_fedplora_v13_upload_package(*args, **kwargs):
    cli_args = kwargs.get("args", None)
    if cli_args is not None:
        apply_fedplora_v13_runtime_defaults(cli_args)
        agg_type = getattr(cli_args, "agg_type", "")
    else:
        agg_type = ""
    if is_fedplora_v13b_os_bonly_agg(agg_type):
        return build_fedplora_v8_upload_package(*args, **kwargs)
    return build_fedplora_v11a_upload_package(*args, **kwargs)


def aggregate_models_fedplora_v13(global_model, client_uploads, args):
    apply_fedplora_v13_runtime_defaults(args)
    if is_fedplora_v13b_os_bonly_agg(getattr(args, "agg_type", "")):
        global_model = aggregate_models_fedplora_v8(global_model, client_uploads, args)
        stats = getattr(args, "_lora_expert_stats", {}) or {}
        summary = stats.setdefault("_summary", {})
        summary.update(
            {
                "algorithm": norm_agg(getattr(args, "agg_type", ""))
                or "fedplora_v13b_os_bonly",
                "v13_branch": "low_comm_routed_B_only",
                "v13_protocol": "cached_shared_A_plus_routed_B",
                "v13_a_payload": "none_cached_A",
                "v13_b_branch": summary.get("cluster_mode", "b_subspace_auto"),
                "v13_global_b_mix_mu": 0.0,
                "v13_comm_accounting": "B_plus_head_downlink_and_uplink",
            }
        )
        stats["algorithm"] = summary["algorithm"]
        args._lora_expert_stats = stats
        return global_model

    global_model = aggregate_models_fedplora_v11a(global_model, client_uploads, args)
    stats = getattr(args, "_lora_expert_stats", {}) or {}
    summary = stats.setdefault("_summary", {})
    summary.update(
        {
            "algorithm": norm_agg(getattr(args, "agg_type", ""))
            or "fedplora_v13a_os",
            "v13_branch": "main_true_A_sketch_routed_B_no_mu",
            "v13_protocol": "one_shot_A_delta_sketch_plus_routed_B",
            "v13_a_payload": "true_lowrank_delta",
            "v13_b_branch": summary.get("cluster_mode", "b_subspace_auto"),
            "v13_training_rule": "train_A+B_no_A_anchor_no_A_prox_no_B_prox",
            "v13_a_correction_alpha": float(
                getattr(args, "v10_a_correction_alpha", 1.0) or 0.0
            ),
            "v13_global_b_mix_mu": 0.0,
        }
    )
    stats["algorithm"] = summary["algorithm"]
    args._lora_expert_stats = stats
    return global_model

