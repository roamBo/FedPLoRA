"""
Flora (NeurIPS 2024): approximate global low-rank update by summing per-client ΔW_i = B_i A_i,
then truncated SVD back to rank r (conventional rank-r projection; not full stacked adapters).
"""

import torch

from methods import common as M
from utilities.utils import is_lora_a_param_name, is_task_head_param_name


def _flora_resolve_svd_device(global_model, args):
    pref = str(getattr(args, "flora_svd_device", "auto") or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            return next(global_model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def _flora_short_layer_name(key: str) -> str:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            tail = ".".join(parts[i : i + 4])
            return tail if len(tail) < 80 else tail[:77] + "..."
    return parts[-4] if len(parts) >= 4 else key[-60:]


def _flora_factorize_delta_sum(delta_sum, r_target, eps, svd_device):
    """
    Rank-r LoRA factors from aggregated ΔW. Uses randomized SVD on large matrices (fast on GPU).
    """
    delta_sum = delta_sum.float().to(svd_device)
    m, n = delta_sum.shape
    r = int(min(r_target, m, n))
    if r < 1:
        return None, None

    use_lowrank = min(m, n) > max(2 * r_target, 64)
    if use_lowrank:
        U, S, V = torch.svd_lowrank(delta_sum, q=r, niter=4)
        sqrt_e = torch.sqrt(S[:r].clamp_min(eps))
        B_new = U[:, :r] @ torch.diag(sqrt_e)
        A_new = torch.diag(sqrt_e) @ V[:, :r].T
    else:
        U, S, Vh = torch.linalg.svd(delta_sum, full_matrices=False)
        r = int(min(r, int(S.numel()), U.shape[1], Vh.shape[0]))
        if r < 1:
            return None, None
        sqrt_e = torch.sqrt(S[:r].clamp_min(eps))
        B_new = U[:, :r] @ torch.diag(sqrt_e)
        A_new = torch.diag(sqrt_e) @ Vh[:r, :]
    return A_new, B_new


def aggregate_models_flora(global_model, client_models, args):
    global_dict = global_model.state_dict()
    n = len(client_models)
    if n == 0:
        return global_model

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
        f"[flora] aggregating {len(lora_keys)} LoRA layers × {n} clients "
        f"(ΔW=SVD on {svd_device}; large layers use svd_lowrank q={r_target})",
        flush=True,
    )

    for idx, kA in enumerate(lora_keys):
        kB = kA.replace("lora_A", "lora_B")
        if (idx == 0) or (idx + 1) % 8 == 0 or (idx + 1) == len(lora_keys):
            print(
                f"[flora] layer {idx + 1}/{len(lora_keys)}: {_flora_short_layer_name(kA)}",
                flush=True,
            )

        delta_sum = None
        for i in range(n):
            sd = M.client_sd(client_models, i)
            d = sd[kB].float() @ sd[kA].float()
            delta_sum = d if delta_sum is None else (delta_sum + d)
        delta_sum = delta_sum / float(n)

        A_new, B_new = _flora_factorize_delta_sum(delta_sum, r_target, eps, svd_device)
        if A_new is None or B_new is None:
            print(f"[flora][warn] skip {kA} (factorize failed)", flush=True)
            continue
        global_dict[kA] = A_new.to(device=global_dict[kA].device, dtype=global_dict[kA].dtype)
        global_dict[kB] = B_new.to(device=global_dict[kB].device, dtype=global_dict[kB].dtype)

    for k in global_dict.keys():
        if is_task_head_param_name(k) and M.all_clients_have_key(client_models, k):
            agg = torch.stack(
                [M.client_sd(client_models, i)[k].float() for i in range(n)], dim=0
            ).mean(0)
            global_dict[k] = agg.to(device=global_dict[k].device, dtype=global_dict[k].dtype)

    global_model.load_state_dict(global_dict)
    print("[flora] aggregation done.", flush=True)
    return global_model
