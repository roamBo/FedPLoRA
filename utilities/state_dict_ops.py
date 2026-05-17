"""Partial state dict load/save utilities shared across tasks (FedPLoRA local vs shared)."""

import torch

from utilities.utils import (
    get_fedplora_shared_param_names,
    get_trainable_param_names,
    is_lora_a_param_name,
    is_lora_b_param_name,
)


def extract_trainable_state_dict(model):
    trainable_names = get_trainable_param_names(model)
    out = {}
    for key, value in model.state_dict().items():
        if key in trainable_names:
            out[key] = value.detach().cpu().clone()
    return out


def extract_round_broadcast_state(model, agg_type=None):
    """CPU snapshot for in-memory FedAvg-style rounds (trainable LoRA + heads only)."""
    del agg_type  # reserved for method-specific broadcast extensions
    return extract_trainable_state_dict(model)


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


def extract_fedalt_local_state(model):
    """FedALT Individual LoRA: both A and B stay on the client."""
    sd = model.state_dict()
    return {
        key: value.detach().cpu().clone()
        for key, value in sd.items()
        if is_lora_a_param_name(key) or is_lora_b_param_name(key)
    }


def load_fedalt_local_state(model, local_state):
    load_partial_state_dict(model, local_state)


def load_fedalt_rotw_state(model, rotw_state):
    """Load personalized RoTW LoRA (frozen at train time in full FedALT forward)."""
    load_partial_state_dict(model, rotw_state)
