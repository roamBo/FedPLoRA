"""
Flora (NeurIPS 2024): approximate global low-rank update by summing per-client ΔW_i = B_i A_i,
then truncated SVD back to rank r (conventional rank-r projection; not full stacked adapters).
"""

import torch

from methods import common as M
from utilities.utils import is_lora_a_param_name, is_task_head_param_name


def aggregate_models_flora(global_model, client_models, args):
    global_dict = global_model.state_dict()
    n = len(client_models)
    if n == 0:
        return global_model

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
            Bi = sd[kB].float()
            Ai = sd[kA].float()
            d = Bi @ Ai
            delta_sum = d if delta_sum is None else (delta_sum + d)
        delta_sum = delta_sum / float(n)

        try:
            U, S, Vh = torch.linalg.svd(delta_sum, full_matrices=False)
            r = int(min(r_target, int(S.numel()), U.shape[1], Vh.shape[0]))
            if r < 1:
                continue
            sqrt_e = torch.sqrt(S[:r].clamp_min(eps))
            B_new = U[:, :r] @ torch.diag(sqrt_e)
            A_new = torch.diag(sqrt_e) @ Vh[:r, :]
            global_dict[kA] = A_new.to(device=global_dict[kA].device, dtype=global_dict[kA].dtype)
            global_dict[kB] = B_new.to(device=global_dict[kB].device, dtype=global_dict[kB].dtype)
        except Exception:
            pass

    for k in global_dict.keys():
        if is_task_head_param_name(k) and M.all_clients_have_key(client_models, k):
            agg = torch.stack(
                [M.client_sd(client_models, i)[k].float() for i in range(n)], dim=0
            ).mean(0)
            global_dict[k] = agg.to(device=global_dict[k].device, dtype=global_dict[k].dtype)

    global_model.load_state_dict(global_dict)
    return global_model
