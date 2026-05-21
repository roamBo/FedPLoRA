"""Branch B — FedPLoRA-SVD aggregator (FedSVD-inspired).

This is a *stub* implementation. Branch B requires modifying LoRA initialization
to make A_0 row-orthonormal (see `utilities/v4_orth_init.py`, TODO) and then
optionally refactoring stacked client A via SVD before downlink.

Reads (via args):
  - v4_svd_orth_init     bool  apply QR-based orthogonalization to A_0
  - v4_svd_refactor      bool  perform SVD on stacked client A and rebuild
  - v4_svd_procrustes    bool  Procrustes-align refactored A to clients' weighted mean
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from methods.common_v4 import (
    M,
    client_weights,
    is_lora_a_param_name,
    is_task_head_param_name,
    metadata_from_uploads,
    task_head_average,
)


def aggregate_models_v4_svd(global_model, client_uploads, args):
    """Stub: combines weighted mean + optional SVD refactor + Procrustes alignment."""
    global_dict = global_model.state_dict()
    client_states = [M.upload_package_state(u) for u in client_uploads]
    if not client_states:
        return global_model
    weights = client_weights(client_uploads, args)
    do_refactor = bool(int(getattr(args, "v4_svd_refactor", 1) or 0))
    do_procrustes = bool(int(getattr(args, "v4_svd_procrustes", 1) or 0))

    for key in list(global_dict.keys()):
        if is_task_head_param_name(key) and all(key in st for st in client_states):
            head = task_head_average(global_dict, key, client_states, weights)
            global_dict[key] = head.to(device=global_dict[key].device, dtype=global_dict[key].dtype)
            continue
        if not (is_lora_a_param_name(key) and all(key in st for st in client_states)):
            continue

        mats = [client_states[i][key].detach().cpu().float() for i in range(len(client_states))]
        N, r, d = len(mats), mats[0].shape[0], mats[0].shape[1]
        weighted_mean = sum(float(weights[i]) * mats[i] for i in range(N))

        if not do_refactor:
            global_dict[key] = weighted_mean.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
            continue

        stacked = torch.cat(mats, dim=0)                         # (N*r, d)
        try:
            U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
            A_new = Vh[:r]                                        # (r, d) orthonormal rows
            # Sign disambiguation
            sign_score = (A_new * weighted_mean).sum(dim=1, keepdim=True)
            signs = torch.sign(sign_score)
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            A_new = A_new * signs
            # Rescale to match Frobenius norm of weighted mean (preserve LoRA scaling)
            target_norm = torch.linalg.matrix_norm(weighted_mean)
            current_norm = torch.linalg.matrix_norm(A_new).clamp_min(1e-8)
            A_new = A_new * (target_norm / current_norm)

            if do_procrustes:
                # R = argmin || R A_new - weighted_mean ||_F  s.t. R^T R = I
                M_proc = weighted_mean @ A_new.transpose(0, 1)
                Up, _, Vhp = torch.linalg.svd(M_proc, full_matrices=False)
                R = Up @ Vhp
                A_new = R @ A_new
        except RuntimeError:
            A_new = weighted_mean

        global_dict[key] = A_new.to(device=global_dict[key].device, dtype=global_dict[key].dtype)

    global_model.load_state_dict(global_dict)
    return global_model
