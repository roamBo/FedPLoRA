"""FedPLoRA-v10: geometry-preserving A correction plus B-subspace routing.

The v9 audit showed a trilemma: B-only routed methods preserve domain geometry
but cap macro accuracy, while A+B training recovers macro but collapses B-domain
structure. v10 is an additive test of the missing edge: let clients train A, but
send only a controlled A correction and keep B routing as the personalization
carrier.

Two variants are exposed:
  - fedplora_v10_geom_a: full A correction with server-side anchor/clipping.
  - fedplora_v10_sketch_a: low-rank A-delta sketch reconstructed server-side.

Both reuse the v8/v9 B-subspace expert code and then overwrite the shared A with
an anchored correction against the round-start shared A.
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
from utilities.utils import (
    is_lora_a_param_name,
    is_lora_b_param_name,
)


SUPPORTED_V10_AGGS = {
    "fedplora_v10_geom_a",
    "fedplora_v10_sketch_a",
    "v10_geom_a",
    "v10_sketch_a",
}

V10_SKETCH_AGGS = {
    "fedplora_v10_sketch_a",
    "v10_sketch_a",
}


def _norm_agg(agg_type) -> str:
    return (agg_type or "").strip().lower().replace("-", "_")


def is_fedplora_v10_agg(agg_type) -> bool:
    return _norm_agg(agg_type) in SUPPORTED_V10_AGGS


def _is_sketch_agg(args) -> bool:
    return _norm_agg(getattr(args, "agg_type", "")) in V10_SKETCH_AGGS


def _lowrank_delta(delta: torch.Tensor, rank: int) -> torch.Tensor:
    """Return a rank-k reconstruction of a 2D delta tensor."""
    x = delta.detach().float().cpu()
    if x.ndim != 2 or x.numel() == 0:
        return x
    rank = max(1, min(int(rank or 1), min(x.shape)))
    try:
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        return (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :]
    except Exception:
        return x


def _row_norm_clip(candidate: torch.Tensor, reference: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    """Clip row norms relative to the shared-A reference while preserving direction."""
    ratio = float(clip_ratio or 0.0)
    if ratio <= 1.0 or candidate.ndim != 2 or reference.ndim != 2:
        return candidate
    cand = candidate.float()
    ref = reference.float()
    cand_norm = cand.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ref_norm = ref.norm(dim=1, keepdim=True).clamp_min(1e-12)
    min_norm = ref_norm / ratio
    max_norm = ref_norm * ratio
    target_norm = cand_norm.clamp(min=min_norm, max=max_norm)
    return cand * (target_norm / cand_norm)


def _a_ref_map(args) -> Dict[str, torch.Tensor]:
    ref = getattr(args, "_fedplora_global_A", None)
    return ref if isinstance(ref, dict) else {}


def build_fedplora_v10_upload_package(
    model,
    client_size=None,
    client_id=None,
    domain=None,
    *,
    args=None,
    client_index: int = 0,
    round_index: int = 0,
):
    """Build v10 payload.

    The current framework still serializes dense tensors, but the sketch variant
    replaces each local A with A_ref + rank-k(delta_A) and records an effective
    communication budget separately. This keeps training/eval compatible with
    the single-adapter PEFT model while testing the algorithmic effect of a cheap
    A correction.
    """

    payload = build_lora_expert_upload_package(
        model,
        client_size=client_size,
        client_id=client_id,
        domain=domain,
        args=args,
        client_index=client_index,
        round_index=round_index,
    )
    if args is None or not _is_sketch_agg(args):
        payload["v10_upload_scope"] = "a_b_full"
        return payload

    ref_map = _a_ref_map(args)
    rank = int(getattr(args, "v10_a_sketch_rank", 2) or 2)
    sd = payload.get("state_dict", {})
    for key, value in list(sd.items()):
        if not is_lora_a_param_name(key) or key not in ref_map:
            continue
        ref = ref_map[key].detach().float().cpu()
        delta = value.detach().float().cpu() - ref
        sd[key] = (ref + _lowrank_delta(delta, rank)).detach().cpu()
    payload["state_dict"] = sd
    payload["v10_upload_scope"] = f"a_delta_rank{max(1, rank)}_plus_b"
    return payload


def _aggregate_v10_shared_a(
    global_dict: Dict[str, torch.Tensor],
    client_states: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    args,
) -> Dict[str, float]:
    ref_map = _a_ref_map(args)
    alpha = float(getattr(args, "v10_a_correction_alpha", 0.35) or 0.35)
    alpha = min(1.0, max(0.0, alpha))
    clip_ratio = float(getattr(args, "v10_a_norm_clip_ratio", 1.5) or 1.5)
    sketch_rank = int(getattr(args, "v10_a_sketch_rank", 2) or 2)
    use_sketch = _is_sketch_agg(args)

    rel_update_norms = []
    row_cosines = []
    clipped_rows = 0
    total_rows = 0
    corrected_layers = 0
    for key in list(global_dict.keys()):
        if not is_lora_a_param_name(key):
            continue
        tensors = [st[key] for st in client_states if key in st]
        if len(tensors) != len(client_states):
            continue
        mean_a = _weighted_mean_tensors(tensors, weights).detach().float().cpu()
        ref = ref_map.get(key, global_dict[key].detach().cpu()).detach().float().cpu()
        if tuple(ref.shape) != tuple(mean_a.shape):
            ref = global_dict[key].detach().float().cpu()
        delta = mean_a - ref
        if use_sketch:
            delta = _lowrank_delta(delta, sketch_rank)
        corrected = ref + alpha * delta
        before = corrected
        corrected = _row_norm_clip(corrected, ref, clip_ratio)
        if corrected.ndim == 2 and before.ndim == 2:
            changed = (before.float().norm(dim=1) - corrected.float().norm(dim=1)).abs() > 1e-8
            clipped_rows += int(changed.sum().item())
            total_rows += int(changed.numel())
            ref_dir = ref.float() / ref.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
            new_dir = corrected.float() / corrected.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
            row_cosines.extend((ref_dir * new_dir).sum(dim=1).abs().clamp(max=1.0).tolist())
        denom = float(torch.linalg.vector_norm(ref).item())
        if denom > 1e-12:
            rel_update_norms.append(
                float(torch.linalg.vector_norm(corrected - ref).item()) / denom
            )
        global_dict[key] = corrected.to(
            device=global_dict[key].device,
            dtype=global_dict[key].dtype,
        )
        corrected_layers += 1

    return {
        "v10_a_corrected_layers": int(corrected_layers),
        "v10_a_correction_alpha": float(alpha),
        "v10_a_sketch_rank": int(sketch_rank) if use_sketch else 0,
        "v10_a_mean_rel_update_norm": (
            float(np.mean(rel_update_norms)) if rel_update_norms else 0.0
        ),
        "v10_a_mean_row_cos_to_ref": (
            float(np.mean(row_cosines)) if row_cosines else float("nan")
        ),
        "v10_a_clipped_row_frac": (
            float(clipped_rows) / float(total_rows) if total_rows else 0.0
        ),
    }


def aggregate_models_fedplora_v10(global_model, client_uploads, args):
    """Aggregate one v10 round.

    The B branch is produced by the existing expert router. The A branch is then
    replaced by an anchored correction against the round-start shared A, which is
    the part v9 could not test.
    """

    client_states = [M.upload_package_state(m) for m in client_uploads]
    if not client_states:
        return global_model

    weights = _client_weight_vector(client_uploads, args)
    global_model = aggregate_models_lora_expert_baseline(
        global_model,
        client_uploads,
        args,
    )
    global_dict = global_model.state_dict()
    a_stats = _aggregate_v10_shared_a(global_dict, client_states, weights, args)
    global_model.load_state_dict(global_dict)

    stats = getattr(args, "_lora_expert_stats", {}) or {}
    summary = stats.setdefault("_summary", {})
    summary.update(
        {
            "algorithm": _norm_agg(getattr(args, "agg_type", "")) or "fedplora_v10_geom_a",
            "v10_a_correction_mode": "lowrank_delta" if _is_sketch_agg(args) else "anchored_full_delta",
            "v10_b_branch": summary.get("cluster_mode", "b_subspace_auto"),
            "v10_training_rule": "train_A+B_with_A_anchor_and_B_prox",
            **a_stats,
        }
    )
    stats["algorithm"] = summary["algorithm"]
    args._lora_expert_stats = stats
    return global_model
