"""
LoRA-A2 (ACL 2025): alternating rounds — even rounds aggregate & sync B (A frozen locally);
odd rounds aggregate & sync A (B frozen). Rank pruning omitted (conventional full-rank train).
"""

import numpy as np
import torch

from methods import common as M
from utilities.utils import is_lora_a_param_name, is_lora_b_param_name, is_task_head_param_name


def aggregate_models_lora_a2(global_model, client_models, args):
    round_idx = int(getattr(args, "_lora_a2_agg_round", 0))
    global_dict = global_model.state_dict()
    n = len(client_models)
    if n == 0:
        return global_model

    client_sizes = getattr(args, "_runtime_client_sizes", None) or getattr(
        args, "_fedplora_client_sizes", None
    )
    if client_sizes is None:
        weights = np.ones(n, dtype=np.float64) / n
    else:
        s = np.asarray(client_sizes, dtype=np.float64)
        weights = s / s.sum()

    train_b_this_round = round_idx % 2 == 0

    for k in global_dict.keys():
        if is_task_head_param_name(k) and M.all_clients_have_key(client_models, k):
            agg = sum(
                weights[i] * M.client_sd(client_models, i)[k].float() for i in range(n)
            )
            global_dict[k] = agg.to(device=global_dict[k].device, dtype=global_dict[k].dtype)
        elif train_b_this_round and is_lora_b_param_name(k):
            if not M.all_clients_have_key(client_models, k):
                continue
            agg = sum(
                weights[i] * M.client_sd(client_models, i)[k].float() for i in range(n)
            )
            global_dict[k] = agg.to(device=global_dict[k].device, dtype=global_dict[k].dtype)
        elif (not train_b_this_round) and is_lora_a_param_name(k):
            if not M.all_clients_have_key(client_models, k):
                continue
            agg = sum(
                weights[i] * M.client_sd(client_models, i)[k].float() for i in range(n)
            )
            global_dict[k] = agg.to(device=global_dict[k].device, dtype=global_dict[k].dtype)

    global_model.load_state_dict(global_dict)
    return global_model
