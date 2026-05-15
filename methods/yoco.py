"""
YOCO (NeurIPS 2025 / FedMLLM ``yoco`` branch) one-shot aggregation.

Aligned with ``YOCO/eval_*_alg.py`` + ``finetune/federated_learning/fed_global.py``
**default** branch (not ``conflict`` / PCWA): after each client trains locally,
the server performs **sample-size weighted FedAvg** on the **full trainable LoRA**
(``lora_A`` + ``lora_B``) and task heads — same uplink volume as ``normal``.

Local A sparsity: ``utilities.train_eval._add_yoco_sparse`` (``--yoco_sparse_lambda``).
Sign regularizer on B (``CPMTrainerSign`` in the official repo) is not wired here yet.
"""

import torch

from methods import common as M
from utilities.utils import (
    is_lora_a_param_name,
    is_lora_b_param_name,
    is_lora_param_name,
    is_task_head_param_name,
)


def _client_state_list(client_sources):
    out = []
    for src in client_sources:
        if isinstance(src, dict) and "state_dict" in src:
            out.append(src["state_dict"])
        else:
            out.append(M.obj_sd(src))
    return out


def _sample_weights(n, args):
    sizes = getattr(args, "_aggregate_client_sizes", None)
    if sizes is None or len(sizes) < n:
        return torch.ones(n, dtype=torch.float64) / max(n, 1)
    s = torch.tensor([float(sizes[i]) for i in range(n)], dtype=torch.float64)
    return s / s.sum().clamp_min(1e-12)


def aggregate_models_yoco(global_model, client_sources, args):
    """
    One-shot YOCO server step (FedMLLM ``global_aggregate`` default).

    ``client_sources``: list of ``nn.Module`` or CPU state_dicts after local training.
    """
    global_dict = global_model.state_dict()
    client_states = _client_state_list(client_sources)
    n = len(client_states)
    if n == 0:
        return global_model

    sw = _sample_weights(n, args)

    for key in global_dict.keys():
        if not all(key in st for st in client_states):
            continue
        if is_lora_param_name(key) or is_task_head_param_name(key):
            stacked = torch.stack(
                [client_states[i][key].float() for i in range(n)], dim=0
            )
            swf = sw.to(device=stacked.device, dtype=stacked.dtype)
            agg = (swf.view(n, *([1] * (stacked.ndim - 1))) * stacked).sum(dim=0)
            global_dict[key] = agg.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )

    global_model.load_state_dict(global_dict)
    return global_model
