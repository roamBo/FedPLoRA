"""
FedALT (AAAI 2026; https://github.com/jmbian/FedALT).

Communication (per client per round):
- **Uplink:** trainable Individual LoRA ``lora_A`` + ``lora_B`` (+ task heads).
- **Downlink:** personalized Rest-of-World (RoTW) LoRA ``lora_A`` + ``lora_B`` for that
  client = mean of **other** clients' uploaded Individual LoRA (leave-one-out).

Server aggregation follows ``FedALT/server.py`` ``Server.aggregation`` (local
``lora_A``/``lora_B`` → per-client RoTW; optional route aggregation not used here).

This repo uses a **single** PEFT LoRA pair per layer (keys ``lora_A`` / ``lora_B``).
RoTW tensors are stored per client and loaded before local training; full dual-branch
forward + mixer (``lora_route``) can be added when a second adapter path is wired in
``utilities/models.py``.
"""

import torch

from methods import common as M
from utilities.utils import (
    get_trainable_param_names,
    is_lora_a_param_name,
    is_lora_b_param_name,
    is_lora_param_name,
    is_task_head_param_name,
)


def build_fedalt_upload_package(model, client_size=None):
    """Client → server: Individual (local) LoRA A and B plus trainable heads."""
    sd = model.state_dict() if hasattr(model, "state_dict") else model
    trainable_names = (
        get_trainable_param_names(model) if hasattr(model, "named_parameters") else None
    )
    upload_sd = {}
    for k, v in sd.items():
        if is_task_head_param_name(k) and (
            trainable_names is None or k in trainable_names
        ):
            upload_sd[k] = v.detach().cpu().clone()
        elif is_lora_a_param_name(k) or is_lora_b_param_name(k):
            upload_sd[k] = v.detach().cpu().clone()
    return {
        "state_dict": upload_sd,
        "client_size": client_size,
    }


def _upload_states(client_uploads):
    return [M.upload_package_state(m) for m in client_uploads]


def _is_local_lora_key(key):
    return is_lora_a_param_name(key) or is_lora_b_param_name(key)


def aggregate_models_fedalt(global_model, client_uploads, args):
    """
    Leave-one-out RoTW per client (official FedALT server).

    Returns the same ``global_model`` (eval template); per-client RoTW state dicts are
    stored on ``args._fedalt_rotw_by_client`` as a list aligned with upload order.
    """
    client_states = _upload_states(client_uploads)
    n = len(client_states)
    if n == 0:
        args._fedalt_rotw_by_client = []
        return global_model

    rotw_list = [{} for _ in range(n)]
    if n == 1:
        args._fedalt_rotw_by_client = rotw_list
        return global_model

    param_names = [k for k in client_states[0].keys() if _is_local_lora_key(k)]

    for client_idx in range(n):
        others = [j for j in range(n) if j != client_idx]
        for param_name in param_names:
            stacked = torch.stack(
                [client_states[j][param_name].float() for j in others], dim=0
            )
            rotw_list[client_idx][param_name] = stacked.mean(0).cpu()

    args._fedalt_rotw_by_client = rotw_list

    # Keep a global template in sync (mean of all locals) for logging / non-personalized eval.
    global_dict = global_model.state_dict()
    for param_name in param_names:
        stacked = torch.stack(
            [client_states[j][param_name].float() for j in range(n)], dim=0
        )
        mean = stacked.mean(0)
        if param_name in global_dict:
            global_dict[param_name] = mean.to(
                device=global_dict[param_name].device,
                dtype=global_dict[param_name].dtype,
            )
    global_model.load_state_dict(global_dict)
    return global_model
