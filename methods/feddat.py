"""
FedDAT-inspired aggregation for text LoRA FL (AAAI 2024).

Uses sample-size weighted FedAvg on trainable LoRA + heads. Local training adds a
teacher proximal term (round-start global LoRA as frozen teacher) — see
utilities.train_eval._add_feddat_teacher_regularizer.

Reference: https://ojs.aaai.org/index.php/AAAI/article/view/29007
"""

import torch

from methods import common as M
from utilities.utils import is_lora_param_name, is_task_head_param_name


def _sample_weights(n, args):
    sizes = getattr(args, "_aggregate_client_sizes", None)
    if sizes is None or len(sizes) < n:
        return torch.ones(n, dtype=torch.float64) / max(n, 1)
    s = torch.tensor([float(sizes[i]) for i in range(n)], dtype=torch.float64)
    return s / s.sum().clamp_min(1e-12)


def aggregate_models_feddat(global_model, client_models, args):
    global_dict = global_model.state_dict()
    n = len(client_models)
    if n == 0:
        return global_model

    sw = _sample_weights(n, args)
    for key in global_dict.keys():
        if not all(key in M.client_sd(client_models, i) for i in range(n)):
            continue
        if is_lora_param_name(key) or is_task_head_param_name(key):
            stacked = torch.stack(
                [M.client_sd(client_models, i)[key].float() for i in range(n)], dim=0
            )
            swf = sw.to(device=stacked.device, dtype=stacked.dtype)
            agg = (swf.view(n, *([1] * (stacked.ndim - 1))) * stacked).sum(dim=0)
            global_dict[key] = agg.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )

    global_model.load_state_dict(global_dict)
    return global_model
