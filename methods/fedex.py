"""FedEx-style aggregation (task heads + LoRA low-rank correction)."""

import numpy as np
import torch

from methods import common as M
from utilities.utils import is_task_head_param_name


def aggregate_models_fedex(global_model, client_models, args):
    global_model = global_model.to("cuda") if torch.cuda.is_available() else global_model
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        if is_task_head_param_name(k) and M.all_clients_have_key(client_models, k):
            global_dict[k] = torch.stack(
                [M.client_sd(client_models, i)[k].float() for i in range(len(client_models))], 0
            ).mean(0)

    for client_model in client_models:
        csd = M.obj_sd(client_model)
        for k in global_dict.keys():
            if is_task_head_param_name(k) and k in csd:
                csd[k].copy_(global_dict[k].to(device=csd[k].device, dtype=csd[k].dtype))

    for name, module in global_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_A_keys = name + ".lora_A.default.weight"
            lora_B_keys = name + ".lora_B.default.weight"

            lora_A_weights = torch.stack(
                [M.client_sd(client_models, i)[lora_A_keys].detach() for i in range(len(client_models))]
            )
            lora_B_weights = torch.stack(
                [M.client_sd(client_models, i)[lora_B_keys].detach() for i in range(len(client_models))]
            )

            M_mat = sum(
                lora_B_weights[i] @ lora_A_weights[i] for i in range(len(client_models))
            ) / len(client_models)

            lora_A_avg = lora_A_weights.mean(0)
            lora_B_avg = lora_B_weights.mean(0)

            scaling_factor = (
                args.lora_alpha / np.sqrt(args.lora_r)
                if args.rslora
                else args.lora_alpha / args.lora_r
            )

            residue = M_mat - lora_B_avg @ lora_A_avg

            global_dict[name + ".lora_A.default.weight"] = lora_A_avg
            global_dict[name + ".lora_B.default.weight"] = lora_B_avg
            global_dict[name + ".base_layer.weight"] += torch.transpose(
                residue * scaling_factor, 1, 0
            )

    global_model.load_state_dict(global_dict)
    return global_model
