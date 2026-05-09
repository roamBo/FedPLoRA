"""
FedP-LoRA (FedPLoRA): upload LoRA A + heads + row stats; aggregate A only; B stays local.

See utilities.state_dict_ops for local/shared state movement.
"""

import numpy as np
import torch

from methods import common as M
from utilities.utils import (
    get_trainable_param_names,
    is_lora_a_param_name,
    is_task_head_param_name,
)


def build_fedplora_upload_package(model, client_size=None):
    """
    Client -> server payload: LoRA A, trainable task heads, row-importance from private B.
    """
    sd = model.state_dict() if hasattr(model, "state_dict") else model
    trainable_names = (
        get_trainable_param_names(model) if hasattr(model, "named_parameters") else None
    )
    upload_sd = {}
    row_importance = {}

    for k, v in sd.items():
        if is_task_head_param_name(k) and (
            trainable_names is None or k in trainable_names
        ):
            upload_sd[k] = v.detach().cpu().clone()
        elif is_lora_a_param_name(k):
            upload_sd[k] = v.detach().cpu().clone()
            kB = k.replace("lora_A", "lora_B")
            if kB in sd:
                A = v.detach().float()
                B = sd[kB].detach().float()
                imp = torch.linalg.vector_norm(B, dim=0) * torch.linalg.vector_norm(A, dim=1)
                imp = imp / imp.mean().clamp_min(1e-8)
                row_importance[k] = imp.detach().cpu()

    return {
        "state_dict": upload_sd,
        "row_importance": row_importance,
        "client_size": client_size,
    }


def aggregate_models_fedplora(global_model, client_models, args):
    """
    Server aggregation: sign-aligned weighted A rows, momentum, QR re-orthogonalization.
    """
    global_dict = global_model.state_dict()
    client_states = [M.upload_package_state(m) for m in client_models]
    client_row_importance = [M.upload_package_row_importance(m) for m in client_models]

    client_sizes = [M.upload_package_client_size(m) for m in client_models]
    if all(x is None for x in client_sizes):
        client_sizes = getattr(args, "_fedplora_client_sizes", None)
    if client_sizes is None:
        weights = np.ones(len(client_states), dtype=np.float64) / len(client_states)
    else:
        sizes = np.asarray(client_sizes, dtype=np.float64)
        weights = sizes / sizes.sum()

    prev_A = {k: v.detach().clone() for k, v in global_dict.items() if is_lora_a_param_name(k)}

    eps = 1e-8
    consensus_power = float(getattr(args, "gp_consensus_power", 2.0))
    agg_momentum = float(getattr(args, "gp_agg_momentum", 0.5))
    abl_no_consensus = bool(getattr(args, "fedplora_ablation_no_consensus", False))
    abl_no_momentum = bool(getattr(args, "fedplora_ablation_no_momentum", False))

    for k in global_dict.keys():
        if is_task_head_param_name(k) and all(k in state for state in client_states):
            agg_head = sum(weights[i] * client_states[i][k].float() for i in range(len(client_states)))
            global_dict[k] = agg_head.to(device=global_dict[k].device, dtype=global_dict[k].dtype)
        elif is_lora_a_param_name(k):
            kA = k
            A_prev = prev_A.get(kA, None)
            A_prev_dir = None
            if A_prev is not None:
                A_prev_f = A_prev.float()
                A_prev_dir = A_prev_f / A_prev_f.norm(dim=1, keepdim=True).clamp_min(eps)

            param_dev = global_dict[kA].device
            param_dtype = global_dict[kA].dtype
            A_acc = torch.zeros(global_dict[kA].shape, dtype=torch.float32, device="cpu")
            w_sum = torch.zeros((global_dict[kA].shape[0], 1), dtype=torch.float32, device="cpu")
            for i in range(len(client_states)):
                Ai = client_states[i][kA].float()
                A_dir = Ai / Ai.norm(dim=1, keepdim=True).clamp_min(eps)
                imp = client_row_importance[i].get(kA, None)
                if imp is None:
                    imp = torch.ones(Ai.shape[0], dtype=A_dir.dtype)
                else:
                    imp = imp.to(dtype=A_dir.dtype)

                if abl_no_consensus:
                    consensus = torch.ones(Ai.shape[0], device=A_dir.device, dtype=A_dir.dtype)
                elif A_prev_dir is not None:
                    ref = A_prev_dir.to(device=A_dir.device, dtype=A_dir.dtype)
                    dots = (A_dir * ref).sum(dim=1)
                    sign = torch.where(dots >= 0, 1.0, -1.0).unsqueeze(1)
                    A_dir = A_dir * sign
                    consensus = dots.abs().clamp_min(eps)
                else:
                    consensus = torch.ones(Ai.shape[0], device=A_dir.device, dtype=A_dir.dtype)

                cons_factor = (
                    torch.ones_like(consensus)
                    if abl_no_consensus
                    else torch.pow(consensus, consensus_power)
                )
                row_weight = float(weights[i]) * imp.clamp_min(eps) * cons_factor
                rw = row_weight.unsqueeze(1)
                A_acc = A_acc + rw * A_dir
                w_sum = w_sum + rw

            if torch.any(w_sum <= 0):
                continue

            A_mean = A_acc / w_sum.clamp_min(eps)
            if A_prev_dir is not None and not abl_no_momentum:
                A_mean = (agg_momentum * A_prev_dir.cpu()) + ((1.0 - agg_momentum) * A_mean)

            Q, _ = torch.linalg.qr(A_mean.transpose(0, 1), mode="reduced")
            A_ortho = Q.transpose(0, 1).contiguous()
            global_dict[kA] = A_ortho.to(device=param_dev, dtype=param_dtype)

    global_model.load_state_dict(global_dict)
    return global_model
