"""
HetLoRA (EMNLP 2024): Frobenius-norm weighted FedAvg on full LoRA (A and B).
Conventional default when ranks differ: pad not implemented — assumes homogeneous PEFT LoRA.
"""

import numpy as np
import torch

from methods import common as M
from utilities.utils import is_lora_param_name, is_task_head_param_name


def aggregate_models_hetlora(global_model, client_models, args):
    global_dict = global_model.state_dict()
    n = len(client_models)
    if n == 0:
        return global_model

    weights = []
    for i in range(n):
        sd = M.client_sd(client_models, i)
        fro = 0.0
        for k, v in sd.items():
            if is_lora_param_name(k) and v.dtype.is_floating_point:
                fro += float(torch.linalg.norm(v.float().reshape(-1)) ** 2)
        weights.append(max(fro, 1e-8))
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    keys = list(global_dict.keys())
    for k in keys:
        if not all(k in M.client_sd(client_models, i) for i in range(n)):
            continue
        if is_lora_param_name(k) or (
            is_task_head_param_name(k) and M.all_clients_have_key(client_models, k)
        ):
            agg = sum(
                float(w[i]) * M.client_sd(client_models, i)[k].float() for i in range(n)
            )
            global_dict[k] = agg.to(device=global_dict[k].device, dtype=global_dict[k].dtype)

    global_model.load_state_dict(global_dict)
    return global_model
