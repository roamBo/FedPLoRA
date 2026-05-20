"""
FlexLoRA (NeurIPS 2024): sample-weighted synthesis of per-client low-rank updates
followed by truncated SVD back to rank r.

Reference: https://proceedings.neurips.cc/paper_files/paper/2024/hash/1a134b50202088aa8c595cc99b310e5a-Abstract-Conference.html
"""

import torch

from methods import common as M
from methods.flora import (
    _flora_factorize_delta_sum,
    _flora_resolve_svd_device,
    _flora_short_layer_name,
)
from utilities.utils import is_lora_a_param_name, is_task_head_param_name


def _sample_weights(n, args):
    sizes = getattr(args, "_aggregate_client_sizes", None)
    if sizes is None or len(sizes) < n:
        return torch.ones(n, dtype=torch.float64) / max(n, 1)
    s = torch.tensor([float(sizes[i]) for i in range(n)], dtype=torch.float64)
    return s / s.sum().clamp_min(1e-12)


def aggregate_models_flexlora(global_model, client_models, args):
    """Weighted sum of client ΔW = B_i A_i, then rank-r SVD (same protocol as Flora, weighted)."""
    global_dict = global_model.state_dict()
    n = len(client_models)
    if n == 0:
        return global_model

    sw = _sample_weights(n, args)
    r_target = int(getattr(args, "lora_r", 8))
    eps = 1e-6
    svd_device = _flora_resolve_svd_device(global_model, args)

    lora_keys = [
        k
        for k in global_dict.keys()
        if is_lora_a_param_name(k)
        and k.replace("lora_A", "lora_B") in global_dict
        and all(
            k in M.client_sd(client_models, i)
            and k.replace("lora_A", "lora_B") in M.client_sd(client_models, i)
            for i in range(n)
        )
    ]
    print(
        f"[flexlora] aggregating {len(lora_keys)} LoRA layers × {n} clients "
        f"(ΔW=SVD on {svd_device})",
        flush=True,
    )

    for idx, kA in enumerate(lora_keys):
        kB = kA.replace("lora_A", "lora_B")
        if (idx == 0) or (idx + 1) % 8 == 0 or (idx + 1) == len(lora_keys):
            print(
                f"[flexlora] layer {idx + 1}/{len(lora_keys)}: {_flora_short_layer_name(kA)}",
                flush=True,
            )

        delta_sum = None
        for i in range(n):
            sd = M.client_sd(client_models, i)
            d = sd[kB].float() @ sd[kA].float()
            w = float(sw[i].item())
            delta_sum = d * w if delta_sum is None else (delta_sum + w * d)

        A_new, B_new = _flora_factorize_delta_sum(delta_sum, r_target, eps, svd_device)
        if A_new is None or B_new is None:
            print(f"[flexlora][warn] skip {kA} (factorize failed)", flush=True)
            continue
        global_dict[kA] = A_new.to(
            device=global_dict[kA].device, dtype=global_dict[kA].dtype
        )
        global_dict[kB] = B_new.to(
            device=global_dict[kB].device, dtype=global_dict[kB].dtype
        )

    for k in global_dict.keys():
        if is_task_head_param_name(k) and M.all_clients_have_key(client_models, k):
            stacked = torch.stack(
                [M.client_sd(client_models, i)[k].float() for i in range(n)], dim=0
            )
            swf = sw.to(device=stacked.device, dtype=stacked.dtype)
            agg = (swf.view(n, *([1] * (stacked.ndim - 1))) * stacked).sum(dim=0)
            global_dict[k] = agg.to(
                device=global_dict[k].device, dtype=global_dict[k].dtype
            )

    global_model.load_state_dict(global_dict)
    print("[flexlora] aggregation done.", flush=True)
    return global_model
