"""
YOCO (NeurIPS 2025 / FedMLLM ``yoco`` branch) one-shot aggregation.

- ``conflict`` (default): FedMLLM ``fed_alg=conflict`` → ``aggregate_lora_weights`` (B-similarity
  weighted fusion of LoRA A/B; see ``methods/references/fedmllm_fed_global_yoco.py``).
- ``fedavg``: legacy sample-size weighted FedAvg on trainable LoRA + heads (pre-alignment behavior).

Local: ``--yoco_sparse_lambda`` on A; ``--yoco_sign_lambda`` on B (sign consistency vs round-start global B).
"""

import torch
import torch.nn.functional as F

from methods import common as M
from utilities.utils import (
    is_lora_a_param_name,
    is_lora_b_param_name,
    is_lora_param_name,
    is_task_head_param_name,
)


def _yoco_aggregate_mode(args):
    mode = str(getattr(args, "yoco_aggregate_mode", "conflict") or "conflict").lower()
    if mode in {"fedavg", "mean", "default"}:
        return "fedavg"
    if mode in {"conflict", "pcwa", "yoco"}:
        return "conflict"
    raise ValueError(
        f"Unknown yoco_aggregate_mode={mode!r}; use 'conflict' (FedMLLM) or 'fedavg' (legacy)."
    )


def _client_state_list(client_sources):
    out = []
    for src in client_sources:
        if isinstance(src, dict) and "state_dict" in src:
            out.append(src["state_dict"])
        else:
            out.append(M.obj_sd(src))
    return out


def _sample_weights(n, args):
    sizes = getattr(args, "_aggregate_client_sizes", None)
    if sizes is None or len(sizes) < n:
        return torch.ones(n, dtype=torch.float64) / max(n, 1)
    s = torch.tensor([float(sizes[i]) for i in range(n)], dtype=torch.float64)
    return s / s.sum().clamp_min(1e-12)


def _cosine_scalar(tensor1, tensor2):
    return F.cosine_similarity(tensor1.reshape(1, -1), tensor2.reshape(1, -1), dim=1).item()


def _normalize_weights(weights):
    w = torch.tensor(weights, dtype=torch.float64)
    return (w / w.sum().clamp_min(1e-12)).tolist()


def _cpu_float(t):
    """Client uploads are CPU; global_model.state_dict() may be on GPU — fuse on CPU."""
    return t.detach().float().cpu()


def _aggregate_lora_weights_conflict(global_dict, client_states, args, *, method="avgm"):
    """FedMLLM ``aggregate_lora_weights`` (B-similarity weights, fuse A and B)."""
    n = len(client_states)
    clients_this_round = list(range(n))
    sw = _sample_weights(n, args)
    sample_this_round = float(sum(sw[i] * 1.0 for i in range(n)))

    for key in list(global_dict.keys()):
        if is_lora_a_param_name(key):
            key_b = key.replace("lora_A", "lora_B")
            if key_b not in global_dict:
                continue
            if not all(key in st and key_b in st for st in client_states):
                continue

            lora_a_list = [_cpu_float(client_states[i][key]) for i in clients_this_round]
            lora_b_list = [_cpu_float(client_states[i][key_b]) for i in clients_this_round]
            g_a = _cpu_float(global_dict[key])
            g_b = _cpu_float(global_dict[key_b])

            similarities = []
            for i, b_i in enumerate(lora_b_list):
                sim_sum = sum(
                    _cosine_scalar(b_i, lora_b_list[j])
                    for j in clients_this_round
                    if j != i
                )
                similarities.append(sim_sum)
            norm_w = _normalize_weights(similarities)

            if method == "mean":
                fused_a = sum(w * a for w, a in zip(norm_w, lora_a_list))
                fused_b = sum(w * b for w, b in zip(norm_w, lora_b_list))
            else:
                fused_a = g_a + sum(
                    (a - g_a) * w for w, a in zip(norm_w, lora_a_list)
                )
                fused_b = g_b + sum(
                    (b - g_b) * w for w, b in zip(norm_w, lora_b_list)
                )
            global_dict[key] = fused_a.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
            global_dict[key_b] = fused_b.to(
                device=global_dict[key_b].device, dtype=global_dict[key_b].dtype
            )

        elif is_lora_b_param_name(key):
            continue

        elif is_task_head_param_name(key):
            if not all(key in st for st in client_states):
                continue
            if method == "mean":
                agg = (
                    sum(_cpu_float(client_states[i][key]) for i in clients_this_round)
                    / float(n)
                )
            else:
                agg = _cpu_float(global_dict[key])
                for i in clients_this_round:
                    c = _cpu_float(client_states[i][key])
                    agg = agg + (c - agg) * float(sw[i])
            global_dict[key] = agg.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )

    return global_dict


def _aggregate_fedavg(global_dict, client_states, args):
    n = len(client_states)
    sw = _sample_weights(n, args)
    for key in global_dict.keys():
        if not all(key in st for st in client_states):
            continue
        if is_lora_param_name(key) or is_task_head_param_name(key):
            stacked = torch.stack([client_states[i][key].float() for i in range(n)], dim=0)
            swf = sw.to(device=stacked.device, dtype=stacked.dtype)
            agg = (swf.view(n, *([1] * (stacked.ndim - 1))) * stacked).sum(dim=0)
            global_dict[key] = agg.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
    return global_dict


def aggregate_models_yoco(global_model, client_sources, args):
    """
    One-shot YOCO server step.

    ``client_sources``: list of CPU trainable state dicts (or modules) after local training.
    """
    global_dict = global_model.state_dict()
    client_states = _client_state_list(client_sources)
    if not client_states:
        return global_model

    mode = _yoco_aggregate_mode(args)
    if mode == "conflict":
        method = str(getattr(args, "yoco_conflict_method", "avgm") or "avgm").lower()
        if method not in {"mean", "avgm"}:
            method = "avgm"
        global_dict = _aggregate_lora_weights_conflict(
            global_dict, client_states, args, method=method
        )
    else:
        global_dict = _aggregate_fedavg(global_dict, client_states, args)

    global_model.load_state_dict(global_dict)
    setattr(args, "_yoco_aggregate_mode_used", mode)
    return global_model
