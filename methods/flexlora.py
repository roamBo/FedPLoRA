"""
FlexLoRA (NeurIPS 2024): sample-weighted synthesis of per-client low-rank updates
followed by truncated SVD back to rank r.

Reference: https://proceedings.neurips.cc/paper_files/paper/2024/hash/1a134b50202088aa8c595cc99b310e5a-Abstract-Conference.html
"""

import torch

from methods import common as M
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

    for kA in list(global_dict.keys()):
        if not is_lora_a_param_name(kA):
            continue
        kB = kA.replace("lora_A", "lora_B")
        if kB not in global_dict or not all(
            kA in M.client_sd(client_models, i) and kB in M.client_sd(client_models, i)
            for i in range(n)
        ):
            continue

        delta_sum = None
        for i in range(n):
            sd = M.client_sd(client_models, i)
            d = sd[kB].float() @ sd[kA].float()
            w = float(sw[i].item())
            delta_sum = d * w if delta_sum is None else (delta_sum + w * d)

        try:
            U, S, Vh = torch.linalg.svd(delta_sum, full_matrices=False)
            r = int(min(r_target, int(S.numel()), U.shape[1], Vh.shape[0]))
            if r < 1:
                continue
            sqrt_e = torch.sqrt(S[:r].clamp_min(eps))
            B_new = U[:, :r] @ torch.diag(sqrt_e)
            A_new = torch.diag(sqrt_e) @ Vh[:r, :]
            global_dict[kA] = A_new.to(
                device=global_dict[kA].device, dtype=global_dict[kA].dtype
            )
            global_dict[kB] = B_new.to(
                device=global_dict[kB].device, dtype=global_dict[kB].dtype
            )
        except Exception:
            pass

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
    return global_model
