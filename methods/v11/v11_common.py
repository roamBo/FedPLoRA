"""Shared FedPLoRA-v11 helpers.

v11 keeps v10 frozen and moves the new 2026-07-05 branches into a fresh
namespace.  The core engineering difference is that sketch variants carry a
real rank-k A-delta payload instead of placing a dense reconstructed A in the
client ``state_dict``.  Server-side v11 aggregators reconstruct A only for the
controlled A-correction step, while B routing still sees the normal LoRA-B
payload.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from methods import common as M
from methods.lora_expert_baselines import (
    _client_weight_vector,
    _weighted_mean_tensors,
    build_lora_expert_upload_package,
)
from utilities.utils import is_lora_a_param_name


def norm_agg(agg_type) -> str:
    return (agg_type or "").strip().lower().replace("-", "_")


def lowrank_sketch(delta: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """Return a CPU rank-k SVD sketch for a 2D delta tensor."""
    x = delta.detach().float().cpu()
    if x.ndim != 2 or x.numel() == 0:
        return {"dense_fallback": x}
    rank = max(1, min(int(rank or 1), min(x.shape)))
    try:
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        return {
            "u": u[:, :rank].detach().cpu(),
            "s": s[:rank].detach().cpu(),
            "vh": vh[:rank, :].detach().cpu(),
        }
    except Exception:
        return {"dense_fallback": x}


def reconstruct_sketch(ref: torch.Tensor, sketch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruct A = ref + delta from a v11 sketch payload."""
    ref_cpu = ref.detach().float().cpu()
    if "dense_fallback" in sketch:
        delta = sketch["dense_fallback"].detach().float().cpu()
    else:
        u = sketch["u"].detach().float().cpu()
        s = sketch["s"].detach().float().cpu()
        vh = sketch["vh"].detach().float().cpu()
        delta = (u * s.unsqueeze(0)) @ vh
    if tuple(delta.shape) != tuple(ref_cpu.shape):
        return ref_cpu
    return (ref_cpu + delta).detach().cpu()


def row_norm_clip(candidate: torch.Tensor, reference: torch.Tensor, clip_ratio: float) -> torch.Tensor:
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


def a_ref_map(args) -> Dict[str, torch.Tensor]:
    ref = getattr(args, "_fedplora_global_A", None)
    return ref if isinstance(ref, dict) else {}


def sketch_bytes(sketch: Dict[str, torch.Tensor]) -> int:
    total = 0
    for tensor in sketch.values():
        if isinstance(tensor, torch.Tensor):
            total += int(tensor.numel() * tensor.element_size())
    return total


def build_v11_upload_package(
    model,
    client_size=None,
    client_id=None,
    domain=None,
    *,
    args=None,
    client_index: int = 0,
    round_index: int = 0,
):
    """Build a v11 upload with B/head tensors plus true rank-k A sketches."""
    payload = build_lora_expert_upload_package(
        model,
        client_size=client_size,
        client_id=client_id,
        domain=domain,
        args=args,
        client_index=client_index,
        round_index=round_index,
    )
    if args is None:
        payload["v11_upload_scope"] = "a_b_dense_no_args"
        return payload

    ref_map = a_ref_map(args)
    rank = int(getattr(args, "v10_a_sketch_rank", 2) or 2)
    sd = payload.get("state_dict", {})
    a_sketch = {}
    total_bytes = 0
    for key, value in list(sd.items()):
        if not is_lora_a_param_name(key):
            continue
        ref = ref_map.get(key)
        if ref is None or tuple(ref.shape) != tuple(value.shape):
            # Keep dense A only as a safety fallback for unusual model keys.
            continue
        delta = value.detach().float().cpu() - ref.detach().float().cpu()
        sk = lowrank_sketch(delta, rank)
        target_dtype = value.detach().cpu().dtype
        sk = {
            name: tensor.to(dtype=target_dtype) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in sk.items()
        }
        a_sketch[key] = sk
        total_bytes += sketch_bytes(sk)
        del sd[key]

    payload["state_dict"] = sd
    payload["v11_a_sketch"] = a_sketch
    payload["v11_a_sketch_rank"] = int(max(1, rank))
    payload["v11_a_sketch_bytes"] = int(total_bytes)
    payload["v11_upload_scope"] = f"true_a_delta_rank{max(1, rank)}_plus_b"
    return payload


def reconstruct_client_states_for_a(client_uploads, args) -> Sequence[Dict[str, torch.Tensor]]:
    """Return client states with A reconstructed from true v11 sketch payloads."""
    ref_map = a_ref_map(args)
    out = []
    for upload in client_uploads:
        state = {
            k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in M.upload_package_state(upload).items()
        }
        sketches = upload.get("v11_a_sketch", {}) if isinstance(upload, dict) else {}
        if isinstance(sketches, dict):
            for key, sketch in sketches.items():
                ref = ref_map.get(key)
                if ref is None or not isinstance(sketch, dict):
                    continue
                state[key] = reconstruct_sketch(ref, sketch)
        out.append(state)
    return out


def aggregate_shared_a_correction(
    global_dict: Dict[str, torch.Tensor],
    client_states_for_a: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    args,
) -> Dict[str, float]:
    """Apply v10-style anchored A correction from reconstructed v11 A states."""
    ref_map = a_ref_map(args)
    alpha = float(getattr(args, "v10_a_correction_alpha", 0.35) or 0.35)
    alpha = min(1.0, max(0.0, alpha))
    clip_ratio = float(getattr(args, "v10_a_norm_clip_ratio", 1.5) or 1.5)
    sketch_rank = int(getattr(args, "v10_a_sketch_rank", 2) or 2)

    rel_update_norms = []
    row_cosines = []
    clipped_rows = 0
    total_rows = 0
    corrected_layers = 0
    for key in list(global_dict.keys()):
        if not is_lora_a_param_name(key):
            continue
        tensors = [st[key] for st in client_states_for_a if key in st]
        if len(tensors) != len(client_states_for_a):
            continue
        mean_a = _weighted_mean_tensors(tensors, weights).detach().float().cpu()
        ref = ref_map.get(key, global_dict[key].detach().cpu()).detach().float().cpu()
        if tuple(ref.shape) != tuple(mean_a.shape):
            ref = global_dict[key].detach().float().cpu()
        delta = mean_a - ref
        corrected = ref + alpha * delta
        before = corrected
        corrected = row_norm_clip(corrected, ref, clip_ratio)
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
        "v11_a_corrected_layers": int(corrected_layers),
        "v11_a_correction_alpha": float(alpha),
        "v11_a_sketch_rank": int(sketch_rank),
        "v11_a_mean_rel_update_norm": (
            float(np.mean(rel_update_norms)) if rel_update_norms else 0.0
        ),
        "v11_a_mean_row_cos_to_ref": (
            float(np.mean(row_cosines)) if row_cosines else float("nan")
        ),
        "v11_a_clipped_row_frac": (
            float(clipped_rows) / float(total_rows) if total_rows else 0.0
        ),
        # Mirror v10 keys so existing analysis scripts can compare v10/v11 directly.
        "v10_a_corrected_layers": int(corrected_layers),
        "v10_a_correction_alpha": float(alpha),
        "v10_a_sketch_rank": int(sketch_rank),
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


def client_weights(client_uploads, args) -> np.ndarray:
    return _client_weight_vector(client_uploads, args)
