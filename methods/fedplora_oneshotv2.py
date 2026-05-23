"""
FedPLoRA-Oneshot: conflict-gated one-shot aggregation for cross-domain PFL.

Clients upload LoRA A, trainable heads, and row-importance statistics derived
from private B. The server never receives B. Compared with YOCO-style PCWA, this
aggregator keeps the initial shared A as the row-coordinate reference, estimates
row-level cross-client conflict, and falls back toward A0 on high-conflict rows.
QR is disabled by default because rotating A would invalidate local B row
coordinates after the single downlink.
"""

import numpy as np
import torch

from methods import common as M
from utilities.utils import is_lora_a_param_name, is_task_head_param_name


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


def _float_arg(args, name, default):
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError):
        return float(default)


def _aggregate_summary(layer_stats):
    total_rows = sum(int(v.get("num_rows", 0)) for v in layer_stats.values())
    if total_rows <= 0:
        return {}

    def wmean(key):
        return float(
            sum(float(v.get(key, 0.0)) * int(v.get("num_rows", 0)) for v in layer_stats.values())
            / total_rows
        )

    return {
        "num_lora_a_matrices": int(len(layer_stats)),
        "num_rows": int(total_rows),
        "mean_conflict": wmean("mean_conflict"),
        "max_conflict": float(max(float(v.get("max_conflict", 0.0)) for v in layer_stats.values())),
        "high_conflict_row_frac": wmean("high_conflict_row_frac"),
        "mean_consensus": wmean("mean_consensus"),
        "mean_ref_similarity": wmean("mean_ref_similarity"),
        "mean_init_gate": wmean("mean_init_gate"),
    }


def _aggregate_plain_fedavg_on_a(global_dict, client_states, client_uploads, args):
    """Ablation: sample-weighted FedAvg on LoRA A + heads (no conflict gate / row reweight)."""
    weights = _client_weights(client_uploads, args)
    n = len(client_states)
    for key in list(global_dict.keys()):
        if not all(key in st for st in client_states):
            continue
        if is_lora_a_param_name(key) or is_task_head_param_name(key):
            agg = sum(
                float(weights[i]) * client_states[i][key].float() for i in range(n)
            )
            global_dict[key] = agg.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
    setattr(
        args,
        "_fedplora_oneshot_conflict_stats",
        {
            "_summary": {
                "ablation": "plain_fedavg",
                "mean_conflict": float("nan"),
                "high_conflict_row_frac": 0.0,
            }
        },
    )
    return global_dict


def aggregate_models_fedplora_oneshot(global_model, client_uploads, args):
    """
    One-shot server rule:
    1. Canonicalize each A row with the initial A0 row direction.
    2. Estimate row conflict as 1 - ||weighted mean signed row direction||.
    3. Aggregate rows using sample size, private-B row importance, and A0 consensus.
    4. Blend high-conflict rows back to A0 to preserve local-B compatibility.
    """
    global_dict = global_model.state_dict()
    client_states = [M.upload_package_state(m) for m in client_uploads]
    if bool(getattr(args, "oneshot_ablation_plain_fedavg", False)):
        global_dict = _aggregate_plain_fedavg_on_a(
            global_dict, client_states, client_uploads, args
        )
        global_model.load_state_dict(global_dict)
        return global_model
    client_row_importance = [M.upload_package_row_importance(m) for m in client_uploads]
    n_clients = len(client_states)
    if n_clients == 0:
        return global_model

    weights = _client_weights(client_uploads, args)
    initial_A = getattr(args, "_fedplora_initial_A", None)
    if not isinstance(initial_A, dict):
        initial_A = {}

    eps = 1e-8
    consensus_power = _float_arg(
        args, "oneshot_consensus_power", getattr(args, "gp_consensus_power", 2.0)
    )
    importance_power = _float_arg(args, "oneshot_importance_power", 1.0)
    conflict_threshold = min(max(_float_arg(args, "oneshot_conflict_threshold", 0.35), 0.0), 1.0)
    conflict_blend = min(max(_float_arg(args, "oneshot_conflict_blend", 1.0), 0.0), 1.0)
    importance_clip = _float_arg(args, "oneshot_importance_clip", 5.0)
    scale_clip_ratio = _float_arg(args, "oneshot_scale_clip_ratio", 0.0)
    keep_init_on_conflict = not bool(
        getattr(args, "oneshot_no_keep_init_on_conflict", False)
    )
    orthogonalize = bool(getattr(args, "oneshot_orthogonalize", False))

    layer_stats = {}

    for key in list(global_dict.keys()):
        if is_task_head_param_name(key) and all(key in state for state in client_states):
            agg_head = sum(
                float(weights[i]) * client_states[i][key].float()
                for i in range(n_clients)
            )
            global_dict[key] = agg_head.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
            continue

        if not (is_lora_a_param_name(key) and all(key in state for state in client_states)):
            continue

        mats = [client_states[i][key].detach().cpu().float() for i in range(n_clients)]
        ref = initial_A.get(key, None)
        if ref is None or tuple(ref.shape) != tuple(mats[0].shape):
            ref = global_dict[key].detach().cpu().float()
        else:
            ref = ref.detach().cpu().float()

        ref_norm = torch.linalg.vector_norm(ref, dim=1, keepdim=True).clamp_min(eps)
        ref_dir = ref / ref_norm

        dir_acc = torch.zeros_like(ref, dtype=torch.float32)
        scale_acc = torch.zeros((ref.shape[0], 1), dtype=torch.float32)
        row_weight_sum = torch.zeros((ref.shape[0], 1), dtype=torch.float32)
        ref_sim_acc = torch.zeros((ref.shape[0], 1), dtype=torch.float32)

        for i, Ai in enumerate(mats):
            row_norm = torch.linalg.vector_norm(Ai, dim=1, keepdim=True).clamp_min(eps)
            row_dir = Ai / row_norm
            dot = (row_dir * ref_dir).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
            sign = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
            signed_dir = row_dir * sign
            ref_sim = dot.abs().clamp_min(eps)

            imp = client_row_importance[i].get(key, None)
            if imp is None:
                imp = torch.ones(ref.shape[0], dtype=torch.float32)
            else:
                imp = imp.detach().cpu().float()
                if imp.numel() != ref.shape[0]:
                    imp = torch.ones(ref.shape[0], dtype=torch.float32)
            imp = imp.view(-1, 1).clamp_min(eps)
            if importance_clip > 0:
                imp = imp.clamp(max=importance_clip)
            if importance_power != 1.0:
                imp = imp.pow(importance_power)

            row_weight = float(weights[i]) * imp * ref_sim.pow(consensus_power)
            dir_acc = dir_acc + row_weight * signed_dir
            scale_acc = scale_acc + row_weight * row_norm
            row_weight_sum = row_weight_sum + row_weight
            ref_sim_acc = ref_sim_acc + float(weights[i]) * ref_sim

        safe_weight = row_weight_sum.clamp_min(eps)
        mean_dir_vec = dir_acc / safe_weight
        consensus = torch.linalg.vector_norm(mean_dir_vec, dim=1, keepdim=True).clamp(
            0.0, 1.0
        )
        agg_dir = mean_dir_vec / consensus.clamp_min(eps)
        scale = scale_acc / safe_weight

        if scale_clip_ratio > 1.0:
            scale = torch.minimum(torch.maximum(scale, ref_norm / scale_clip_ratio), ref_norm * scale_clip_ratio)

        candidate = agg_dir * scale
        degenerate = consensus.squeeze(1) <= 1e-6
        if bool(degenerate.any()):
            candidate[degenerate] = ref[degenerate]

        conflict = (1.0 - consensus).clamp(0.0, 1.0)
        if keep_init_on_conflict and conflict_threshold < 1.0:
            gate = ((conflict - conflict_threshold) / max(1.0 - conflict_threshold, eps)).clamp(
                0.0, 1.0
            )
            gate = gate * conflict_blend
            candidate = (1.0 - gate) * candidate + gate * ref
        else:
            gate = torch.zeros_like(conflict)

        if orthogonalize and candidate.shape[0] <= candidate.shape[1]:
            row_scale = torch.linalg.vector_norm(candidate, dim=1, keepdim=True).clamp_min(eps)
            try:
                q, _ = torch.linalg.qr(candidate.transpose(0, 1), mode="reduced")
                candidate = q.transpose(0, 1).contiguous() * row_scale
            except RuntimeError:
                pass

        global_dict[key] = candidate.to(
            device=global_dict[key].device, dtype=global_dict[key].dtype
        )

        layer_stats[key] = {
            "num_rows": int(ref.shape[0]),
            "mean_conflict": float(conflict.mean().item()),
            "max_conflict": float(conflict.max().item()),
            "high_conflict_row_frac": float(
                (conflict.squeeze(1) > conflict_threshold).float().mean().item()
            ),
            "mean_consensus": float(consensus.mean().item()),
            "mean_ref_similarity": float(ref_sim_acc.mean().item()),
            "mean_init_gate": float(gate.mean().item()),
        }

    stats = dict(layer_stats)
    stats["_summary"] = _aggregate_summary(layer_stats)
    setattr(args, "_fedplora_oneshot_conflict_stats", stats)

    global_model.load_state_dict(global_dict)
    return global_model
