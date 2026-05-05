"""
FedSA-LoRA (ICLR 2025): upload LoRA A (+ heads); FedAvg-style mean on A; B stays local.
Conventional default: sample-size weighted mean, no sign-alignment or QR (cf. FedP-LoRA).
"""

import numpy as np
import torch

from methods import common as M
from utilities.utils import is_lora_a_param_name, is_task_head_param_name


def aggregate_models_fedsa_lora(global_model, client_uploads, args):
    global_dict = global_model.state_dict()
    client_states = [M.upload_package_state(m) for m in client_uploads]
    client_sizes = [M.upload_package_client_size(m) for m in client_uploads]
    if all(x is None for x in client_sizes):
        client_sizes = getattr(args, "_fedplora_client_sizes", None)
    if client_sizes is None:
        weights = np.ones(len(client_states), dtype=np.float64) / max(len(client_states), 1)
    else:
        sizes = np.asarray(client_sizes, dtype=np.float64)
        weights = sizes / sizes.sum()

    for k in global_dict.keys():
        if is_task_head_param_name(k) and all(k in st for st in client_states):
            agg = sum(weights[i] * client_states[i][k].float() for i in range(len(client_states)))
            global_dict[k] = agg.to(device=global_dict[k].device, dtype=global_dict[k].dtype)
        elif is_lora_a_param_name(k) and all(k in st for st in client_states):
            agg = sum(weights[i] * client_states[i][k].float() for i in range(len(client_states)))
            global_dict[k] = agg.to(device=global_dict[k].device, dtype=global_dict[k].dtype)

    global_model.load_state_dict(global_dict)
    return global_model
