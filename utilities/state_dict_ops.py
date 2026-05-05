"""Partial state dict load/save utilities shared across tasks (FedPLoRA local vs shared)."""

import torch

from utilities.utils import (
    get_fedplora_shared_param_names,
    get_trainable_param_names,
    is_lora_b_param_name,
)


def extract_trainable_state_dict(model):
    trainable_names = get_trainable_param_names(model)
    out = {}
    for key, value in model.state_dict().items():
        if key in trainable_names:
            out[key] = value.detach().cpu().clone()
    return out


def load_partial_state_dict(model, partial_state_dict):
    current = model.state_dict()
    for key, value in partial_state_dict.items():
        if key in current:
            current[key] = value.to(device=current[key].device, dtype=current[key].dtype)
    model.load_state_dict(current)


def extract_fedplora_local_state(model):
    sd = model.state_dict()
    local_state = {}
    for key, value in sd.items():
        if is_lora_b_param_name(key):
            local_state[key] = value.detach().cpu().clone()
    return local_state


def load_fedplora_local_state(model, local_state):
    load_partial_state_dict(model, local_state)


def broadcast_fedplora_shared_state(model, shared_state_dict):
    current = model.state_dict()
    shared_names = get_fedplora_shared_param_names(model)
    for key, value in shared_state_dict.items():
        if key in current and key in shared_names:
            current[key] = value.to(device=current[key].device, dtype=current[key].dtype)
    model.load_state_dict(current)
