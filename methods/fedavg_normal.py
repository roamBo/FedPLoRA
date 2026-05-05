"""FedAvg on LoRA A/B (and task heads when all clients send them)."""

import torch

from methods import common as M
from utilities.utils import is_task_head_param_name


def aggregate_models_normal(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora" in k:
            global_dict[k] = torch.stack(
                [M.client_sd(client_models, i)[k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if is_task_head_param_name(k) and M.all_clients_have_key(client_models, k):
            global_dict[k] = torch.stack(
                [M.client_sd(client_models, i)[k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)
    return global_model
