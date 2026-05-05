"""
YOCO (NeurIPS 2025): one-shot; PCWA on client A matrices (conventional PCA weights on flattened A).
Local L1-on-A regularizer is applied in utilities/train_eval (see --yoco_sparse_lambda).
"""

import numpy as np
import torch

from methods import common as M
from utilities.utils import is_lora_a_param_name, is_task_head_param_name


def aggregate_models_yoco(global_model, client_uploads, args):
    """PCWA: weight each client by energy on top-k PCA directions of stacked flattened A."""
    global_dict = global_model.state_dict()
    client_states = [M.upload_package_state(m) for m in client_uploads]
    n = len(client_states)
    if n == 0:
        return global_model

    k_pc = int(getattr(args, "yoco_pcwa_components", min(3, max(1, n - 1))))

    for key in global_dict.keys():
        if is_task_head_param_name(key) and all(key in st for st in client_states):
            stacked = torch.stack([client_states[i][key].float() for i in range(n)], dim=0)
            global_dict[key] = stacked.mean(0).to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
        elif is_lora_a_param_name(key) and all(key in st for st in client_states):
            mats = [client_states[i][key].float() for i in range(n)]
            flat = torch.stack([m.reshape(-1) for m in mats], dim=0)
            if n == 1 or flat.shape[1] < 2:
                agg = flat.mean(0).reshape(mats[0].shape)
            else:
                mean_row = flat.mean(0, keepdim=True)
                centered = flat - mean_row
                try:
                    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
                    q = min(k_pc, Vh.shape[0])
                    scores = torch.zeros(n, dtype=flat.dtype)
                    for j in range(q):
                        pc = Vh[j, :]
                        scores = scores + (centered * pc.unsqueeze(0)).sum(dim=1).abs()
                    w = scores.clamp_min(1e-8)
                    w = w / w.sum()
                except Exception:
                    w = torch.ones(n, dtype=flat.dtype) / n
                agg = sum(w[i] * mats[i] for i in range(n))
            global_dict[key] = agg.to(device=global_dict[key].device, dtype=global_dict[key].dtype)

    global_model.load_state_dict(global_dict)
    return global_model
