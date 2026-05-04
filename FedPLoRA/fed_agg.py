import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from data_utils import *
from models import *
from sklearn.metrics import matthews_corrcoef
import numpy as np
import torch.nn as nn
from utils import (
    get_fedplora_shared_param_names,
    get_trainable_param_names,
    is_fedplora_agg,
    is_fedplora_oneshot_agg,
    is_fedplora_shared_param_name,
    is_lora_a_param_name,
    is_lora_b_param_name,
    is_task_head_param_name,
)


def _client_sd(client_models, i):
    """State dict from a client (nn.Module) or an already-stored dict."""
    m = client_models[i]
    return m.state_dict() if hasattr(m, "state_dict") else m


def _obj_sd(obj):
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def _upload_package_state(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def _upload_package_row_importance(obj):
    if isinstance(obj, dict):
        return obj.get("row_importance", {})
    return {}


def _upload_package_client_size(obj):
    if isinstance(obj, dict):
        return obj.get("client_size", None)
    return None


def _all_clients_have_key(client_models, key):
    for i in range(len(client_models)):
        if key not in _client_sd(client_models, i):
            return False
    return True


def build_fedplora_upload_package(model, client_size=None):
    """
    Build the client -> server payload for FedPLoRA.

    The server only receives:
    - LoRA A matrices
    - trainable task head parameters
    - row-importance scalars derived locally from private B

    The private B matrices never leave the client.
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
                # Rank-component importance of B[:, j] A[j, :].
                imp = torch.linalg.vector_norm(B, dim=0) * torch.linalg.vector_norm(
                    A, dim=1
                )
                imp = imp / imp.mean().clamp_min(1e-8)
                row_importance[k] = imp.detach().cpu()

    return {
        "state_dict": upload_sd,
        "row_importance": row_importance,
        "client_size": client_size,
    }


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


def aggregate_models_normal(global_model, client_models):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora" in k:  # Only aggregate LoRA parameters
            global_dict[k] = torch.stack(
                [_client_sd(client_models, i)[k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if is_task_head_param_name(k) and _all_clients_have_key(client_models, k):
            global_dict[k] = torch.stack(
                [_client_sd(client_models, i)[k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)

    return global_model


def aggregate_models_ffa(global_model, client_models):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora_B" in k:  # Only aggregate LoRA B parameters
            global_dict[k] = torch.stack(
                [_client_sd(client_models, i)[k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if is_task_head_param_name(k) and _all_clients_have_key(client_models, k):
            global_dict[k] = torch.stack(
                [_client_sd(client_models, i)[k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)

    return global_model


def aggregate_models_fedex(global_model, client_models, args):

    global_model = (
        global_model.to("cuda") if torch.cuda.is_available() else global_model
    )
    global_dict = global_model.state_dict()

    for k in global_dict.keys():

        if is_task_head_param_name(k) and _all_clients_have_key(client_models, k):
            global_dict[k] = torch.stack(
                [_client_sd(client_models, i)[k].float() for i in range(len(client_models))], 0
            ).mean(0)

    for client_model in client_models:
        csd = _obj_sd(client_model)
        for k in global_dict.keys():
            if is_task_head_param_name(k) and k in csd:
                csd[k].copy_(global_dict[k].to(device=csd[k].device, dtype=csd[k].dtype))

    for name, module in global_model.named_modules():

        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):

            lora_A_keys = name + ".lora_A.default.weight"
            lora_B_keys = name + ".lora_B.default.weight"
            base_layer_keys = name + ".base_layer.weight"

            lora_A_weights = torch.stack(
                [
                    _client_sd(client_models, i)[lora_A_keys].detach()
                    for i in range(len(client_models))
                ]
            )
            lora_B_weights = torch.stack(
                [
                    _client_sd(client_models, i)[lora_B_keys].detach()
                    for i in range(len(client_models))
                ]
            )

            # M shape: (d, k)
            M = sum(
                lora_B_weights[i] @ lora_A_weights[i] for i in range(len(client_models))
            ) / len(client_models)

            lora_A_avg = lora_A_weights.mean(0)
            lora_B_avg = lora_B_weights.mean(0)

            scaling_factor = (
                args.lora_alpha / np.sqrt(args.lora_r)
                if args.rslora
                else args.lora_alpha / args.lora_r
            )

            residue = M - lora_B_avg @ lora_A_avg

            global_dict[name + ".lora_A.default.weight"] = lora_A_avg
            global_dict[name + ".lora_B.default.weight"] = lora_B_avg
            global_dict[name + ".base_layer.weight"] += torch.transpose(
                residue * scaling_factor, 1, 0
            )

    global_model.load_state_dict(global_dict)

    return global_model


def aggregate_models_gp_lora(global_model, client_models, args):
    """
    FedPLoRA / GP-LoRA server aggregation.

    The server only consumes upload payloads made of:
    - client LoRA A matrices
    - trainable task-head weights
    - per-row importance scalars

    Private LoRA B matrices remain local and are never read by the server.

    Aggregation:
    1. sign-align each client A to the previous global A row-wise;
    2. weight each row by sample size, row importance and consensus with A_prev;
    3. average row directions;
    4. blend with previous global A using server momentum;
    5. re-orthonormalize rows with QR.
    """

    global_dict = global_model.state_dict()
    client_states = [_upload_package_state(m) for m in client_models]
    client_row_importance = [_upload_package_row_importance(m) for m in client_models]

    client_sizes = [_upload_package_client_size(m) for m in client_models]
    if all(x is None for x in client_sizes):
        client_sizes = getattr(args, "_gp_lora_client_sizes", None)
    if client_sizes is None:
        weights = np.ones(len(client_states), dtype=np.float64) / len(client_states)
    else:
        sizes = np.asarray(client_sizes, dtype=np.float64)
        weights = sizes / sizes.sum()

    # Snapshot previous global A for sign alignment of singular vectors.
    prev_A = {
        k: v.detach().clone()
        for k, v in global_dict.items()
        if is_lora_a_param_name(k)
    }

    eps = 1e-8
    consensus_power = float(getattr(args, "gp_consensus_power", 2.0))
    agg_momentum = float(getattr(args, "gp_agg_momentum", 0.5))

    for k in global_dict.keys():
        if is_task_head_param_name(k) and all(k in state for state in client_states):
            global_dict[k] = sum(
                weights[i] * client_states[i][k].float()
                for i in range(len(client_states))
            )
        elif is_lora_a_param_name(k):
            kA = k
            A_prev = prev_A.get(kA, None)
            A_prev_f = None
            A_prev_dir = None
            if A_prev is not None:
                A_prev_f = A_prev.float()
                A_prev_dir = A_prev_f / A_prev_f.norm(dim=1, keepdim=True).clamp_min(
                    eps
                )

            A_acc = torch.zeros_like(global_dict[kA].float())
            w_sum = torch.zeros((global_dict[kA].shape[0], 1), dtype=torch.float32)
            for i in range(len(client_states)):
                Ai = client_states[i][kA].float()
                A_dir = Ai / Ai.norm(dim=1, keepdim=True).clamp_min(eps)
                imp = client_row_importance[i].get(kA, None)
                if imp is None:
                    imp = torch.ones(Ai.shape[0], dtype=A_dir.dtype)
                else:
                    imp = imp.to(dtype=A_dir.dtype)

                if A_prev_dir is not None:
                    ref = A_prev_dir.to(device=A_dir.device, dtype=A_dir.dtype)
                    dots = (A_dir * ref).sum(dim=1)
                    sign = torch.where(dots >= 0, 1.0, -1.0).unsqueeze(1)
                    A_dir = A_dir * sign
                    consensus = dots.abs().clamp_min(eps)
                else:
                    consensus = torch.ones(
                        Ai.shape[0], device=A_dir.device, dtype=A_dir.dtype
                    )

                row_weight = (
                    float(weights[i])
                    * imp.clamp_min(eps)
                    * torch.pow(consensus, consensus_power)
                )

                A_acc = A_acc + row_weight.unsqueeze(1) * A_dir.cpu()
                w_sum = w_sum + row_weight.unsqueeze(1).cpu()

            if torch.any(w_sum <= 0):
                continue

            A_mean = A_acc / w_sum.clamp_min(eps)
            if A_prev_dir is not None:
                A_mean = (agg_momentum * A_prev_dir.cpu()) + (
                    (1.0 - agg_momentum) * A_mean
                )

            Q, _ = torch.linalg.qr(A_mean.transpose(0, 1), mode="reduced")
            A_ortho = Q.transpose(0, 1).contiguous()
            global_dict[kA] = A_ortho.to(dtype=global_dict[kA].dtype)

    global_model.load_state_dict(global_dict)
    return global_model


def aggregate_models_fedplora_oneshot(global_model, client_models, args):
    """
    FedPLoRA-Oneshot server aggregation.

    One-shot setting:
    - clients train once from the same shared initialization;
    - the server receives only LoRA A, task-head weights and row statistics;
    - private LoRA B matrices remain local;
    - A rows with severe cross-domain conflict are gated toward the initial
      shared A instead of being force-averaged.

    This differs from multi-round FedPLoRA: there is no server momentum and no
    dependence on a previous aggregated model. The initial A acts as the only
    shared coordinate system.
    """
    if not is_fedplora_oneshot_agg(getattr(args, "agg_type", None)):
        raise ValueError("aggregate_models_fedplora_oneshot requires FedPLoRA-Oneshot")

    global_dict = global_model.state_dict()
    client_states = [_upload_package_state(m) for m in client_models]
    client_row_importance = [_upload_package_row_importance(m) for m in client_models]

    client_sizes = [_upload_package_client_size(m) for m in client_models]
    if all(x is None for x in client_sizes):
        client_sizes = getattr(args, "_gp_lora_client_sizes", None)
    if client_sizes is None:
        weights = np.ones(len(client_states), dtype=np.float64) / len(client_states)
    else:
        sizes = np.asarray(client_sizes, dtype=np.float64)
        weights = sizes / sizes.sum()

    init_A = getattr(args, "_gp_lora_initial_A", None)
    if not isinstance(init_A, dict) or not init_A:
        init_A = {
            k: v.detach().cpu().clone()
            for k, v in global_dict.items()
            if is_lora_a_param_name(k)
        }

    eps = 1e-8
    consensus_power = float(getattr(args, "oneshot_consensus_power", 2.0))
    conflict_threshold = float(getattr(args, "oneshot_conflict_threshold", 0.35))
    keep_init_on_conflict = bool(getattr(args, "oneshot_keep_init_on_conflict", True))
    orthogonalize = bool(getattr(args, "oneshot_orthogonalize", False))

    row_conflict_stats = {}

    for k in global_dict.keys():
        if is_task_head_param_name(k) and all(k in state for state in client_states):
            global_dict[k] = sum(
                weights[i] * client_states[i][k].float()
                for i in range(len(client_states))
            ).to(dtype=global_dict[k].dtype)
        elif is_lora_a_param_name(k):
            A_ref = init_A.get(k, None)
            if A_ref is None:
                continue
            A_ref_f = A_ref.float()
            A_ref_dir = A_ref_f / A_ref_f.norm(dim=1, keepdim=True).clamp_min(eps)

            A_acc = torch.zeros_like(global_dict[k].float())
            w_sum = torch.zeros((global_dict[k].shape[0], 1), dtype=torch.float32)
            norm_acc = torch.zeros((global_dict[k].shape[0], 1), dtype=torch.float32)
            signed_dirs = []

            for i in range(len(client_states)):
                Ai = client_states[i][k].float()
                A_norm = Ai.norm(dim=1, keepdim=True).clamp_min(eps)
                A_dir = Ai / A_norm
                ref = A_ref_dir.to(device=A_dir.device, dtype=A_dir.dtype)
                dots_to_ref = (A_dir * ref).sum(dim=1)
                sign = torch.where(dots_to_ref >= 0, 1.0, -1.0).unsqueeze(1)
                A_dir = A_dir * sign
                signed_dirs.append(A_dir.cpu())

                consensus = dots_to_ref.abs().clamp_min(eps)
                imp = client_row_importance[i].get(k, None)
                if imp is None:
                    imp = torch.ones(Ai.shape[0], dtype=A_dir.dtype)
                else:
                    imp = imp.to(dtype=A_dir.dtype)

                row_weight = (
                    float(weights[i])
                    * imp.clamp_min(eps)
                    * torch.pow(consensus, consensus_power)
                )
                A_acc = A_acc + row_weight.unsqueeze(1).cpu() * A_dir.cpu()
                w_sum = w_sum + row_weight.unsqueeze(1).cpu()
                norm_acc = norm_acc + row_weight.unsqueeze(1).cpu() * A_norm.cpu()

            if torch.any(w_sum <= 0):
                continue

            stacked = torch.stack(signed_dirs, dim=0)
            row_mean_dir = stacked.mean(dim=0)
            row_mean_norm = row_mean_dir.norm(dim=1)
            row_conflict = (1.0 - row_mean_norm).clamp(0.0, 1.0)
            row_gate = (1.0 - row_conflict).clamp(0.0, 1.0).pow(consensus_power)

            A_mean = A_acc / w_sum.clamp_min(eps)
            A_mean = A_mean / A_mean.norm(dim=1, keepdim=True).clamp_min(eps)
            A_scale = norm_acc / w_sum.clamp_min(eps)
            if keep_init_on_conflict:
                conflict_mask = (row_conflict > conflict_threshold).unsqueeze(1)
                gated = row_gate.unsqueeze(1) * A_mean + (
                    1.0 - row_gate.unsqueeze(1)
                ) * A_ref_dir.cpu()
                A_mean = torch.where(conflict_mask, gated, A_mean)

            if orthogonalize:
                Q, _ = torch.linalg.qr(A_mean.transpose(0, 1), mode="reduced")
                A_mean = Q.transpose(0, 1).contiguous()

            A_mean = A_mean / A_mean.norm(dim=1, keepdim=True).clamp_min(eps)
            A_mean = A_mean * A_scale
            global_dict[k] = A_mean.to(dtype=global_dict[k].dtype)
            row_conflict_stats[k] = {
                "mean_conflict": float(row_conflict.mean().item()),
                "max_conflict": float(row_conflict.max().item()),
                "conflict_rows": int((row_conflict > conflict_threshold).sum().item()),
                "total_rows": int(row_conflict.numel()),
            }

    setattr(args, "_fedplora_oneshot_conflict_stats", row_conflict_stats)
    global_model.load_state_dict(global_dict)
    return global_model
