import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from utilities.data_utils import *
from utilities.models import *
import argparse
import warnings
from sklearn.metrics import matthews_corrcoef
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
try:
    from opacus import PrivacyEngine
    from opacus.validators.module_validator import ModuleValidator
except Exception:
    PrivacyEngine = None
    ModuleValidator = None
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    Cider = None
import torch
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
from utilities.data_utils import *
import os
from copy import deepcopy
import json
from utilities.utils import (
    is_feddat_agg,
    is_fedplora_multiround_agg,
    is_fedplora_oneshot_agg,
    is_fedplora_oneshot_family_agg,
    is_fedplora_v10_family_agg,
    is_fedplora_v11_family_agg,
    is_fedplora_v12_family_agg,
    is_fedplora_v13_a_sketch_agg,
    is_yoco_agg,
)


def _fedplora_regularization_losses(model, args):
    """
    FedPLoRA local regularizers:
    - alignment of B_i A_local with B_i A_global
    - proximity of A_local to the broadcast global A
    - row-orthogonality for A as a stable shared basis
    """
    if not is_fedplora_multiround_agg(getattr(args, "agg_type", None)):
        return {}
    A_global_gpu = getattr(args, "_fedplora_global_A_gpu", None)
    A_global = (
        A_global_gpu
        if isinstance(A_global_gpu, dict) and A_global_gpu
        else getattr(args, "_fedplora_global_A", None)
    )
    if not isinstance(A_global, dict) or not A_global:
        return {}

    eps = 1e-8
    sd = model.state_dict()
    reg = {"align": None, "prox": None, "orth": None}
    for kA_local, A_local in sd.items():
        if "lora_A" not in kA_local or not kA_local.endswith("default.weight"):
            continue
        if kA_local not in A_global:
            continue
        A_ref = A_global[kA_local]
        if A_ref.device != A_local.device or A_ref.dtype != A_local.dtype:
            A_ref = A_ref.to(device=A_local.device, dtype=A_local.dtype)

        prox_term = torch.mean((A_local.float() - A_ref.float()) ** 2)
        reg["prox"] = prox_term if reg["prox"] is None else (reg["prox"] + prox_term)

        A_dir = A_local.float() / A_local.float().norm(dim=1, keepdim=True).clamp_min(
            eps
        )
        gram = A_dir @ A_dir.transpose(0, 1)
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        orth_term = torch.mean((gram - eye) ** 2)
        reg["orth"] = orth_term if reg["orth"] is None else (reg["orth"] + orth_term)

        kB_local = kA_local.replace("lora_A", "lora_B")
        if kB_local not in sd:
            continue
        B_local = sd[kB_local]
        dW_local = (B_local @ A_local).float()
        dW_ref = (B_local @ A_ref).float()

        v1 = dW_local.reshape(-1)
        v2 = dW_ref.reshape(-1)
        n1 = torch.linalg.vector_norm(v1) + eps
        n2 = torch.linalg.vector_norm(v2) + eps
        cos = (v1 @ v2) / (n1 * n2)
        term = 1.0 - cos
        reg["align"] = term if reg["align"] is None else (reg["align"] + term)

    return reg


def _add_fedplora_regularization(loss, model, args):
    if not is_fedplora_multiround_agg(getattr(args, "agg_type", None)):
        return loss

    reg = _fedplora_regularization_losses(model, args)
    if reg.get("align") is not None:
        loss = loss + (
            float(getattr(args, "gp_align_lambda", 0.01)) * reg["align"]
        )
    if reg.get("prox") is not None:
        loss = loss + (
            float(getattr(args, "gp_prox_lambda", 0.001)) * reg["prox"]
        )
    if reg.get("orth") is not None:
        loss = loss + (
            float(getattr(args, "gp_orth_lambda", 0.0001)) * reg["orth"]
        )
    return loss


def _add_yoco_sign_regularizer(loss, model, args):
    """B sign consistency vs round-start global LoRA (FedMLLM / YOCO-style)."""
    if not is_yoco_agg(getattr(args, "agg_type", None)):
        return loss
    ref = getattr(args, "_yoco_round_start_trainable", None)
    if not isinstance(ref, dict) or not ref:
        return loss
    lam = float(getattr(args, "yoco_sign_lambda", 0.0))
    if lam <= 0:
        return loss
    terms = []
    for key, p in model.named_parameters():
        if "lora_B" not in key or not key.endswith("default.weight") or not p.requires_grad:
            continue
        if key not in ref:
            continue
        b0 = ref[key]
        if b0.device != p.device or b0.dtype != p.dtype:
            b0 = b0.to(device=p.device, dtype=p.dtype)
        if tuple(b0.shape) != tuple(p.shape):
            continue
        # Penalize element-wise sign disagreement with broadcast global B at round start.
        terms.append(torch.relu(-torch.sign(p.float()) * torch.sign(b0.float())).mean())
    if terms:
        loss = loss + lam * torch.stack(terms).mean()
    return loss


def _add_yoco_sparse(loss, model, args):
    agg = getattr(args, "agg_type", None)
    if not (is_yoco_agg(agg) or is_fedplora_oneshot_family_agg(agg)):
        return loss
    lam = float(getattr(args, "yoco_sparse_lambda", 1e-4))
    terms = [
        p.abs().mean()
        for _n, p in model.named_parameters()
        if "lora_A" in _n and p.requires_grad
    ]
    if terms:
        loss = loss + lam * torch.stack(terms).mean()
    return loss


def _add_feddat_teacher_regularizer(loss, model, args):
    """Proximal term to round-start teacher LoRA (FedDAT-inspired)."""
    if not is_feddat_agg(getattr(args, "agg_type", None)):
        return loss
    teacher = getattr(args, "_feddat_teacher_state", None)
    if not isinstance(teacher, dict) or not teacher:
        return loss
    lam = float(getattr(args, "feddat_teacher_lambda", 0.01))
    if lam <= 0:
        return loss
    terms = []
    for key, p in model.named_parameters():
        if not p.requires_grad or key not in teacher:
            continue
        t = teacher[key]
        if t.device != p.device or t.dtype != p.dtype:
            t = t.to(device=p.device, dtype=p.dtype)
        if tuple(t.shape) != tuple(p.shape):
            continue
        terms.append(torch.mean((p.float() - t.float()) ** 2))
    if terms:
        loss = loss + lam * torch.stack(terms).mean()
    return loss


def _add_fedplora_oneshot_anchor(loss, model, args):
    if not is_fedplora_oneshot_family_agg(getattr(args, "agg_type", None)):
        return loss
    initial_gpu = getattr(args, "_fedplora_initial_A_gpu", None)
    initial_A = (
        initial_gpu
        if isinstance(initial_gpu, dict) and initial_gpu
        else getattr(args, "_fedplora_initial_A", None)
    )
    if not isinstance(initial_A, dict) or not initial_A:
        return loss

    anchor_lam = float(getattr(args, "oneshot_anchor_lambda", 1e-4))
    prox_lam = float(getattr(args, "oneshot_prox_lambda", 0.0))
    if anchor_lam <= 0 and prox_lam <= 0:
        return loss

    eps = 1e-8
    anchor_terms = []
    prox_terms = []
    for key, A_local in model.named_parameters():
        if "lora_A" not in key or not key.endswith("default.weight"):
            continue
        if not A_local.requires_grad:
            continue
        A0 = initial_A.get(key, None)
        if A0 is None:
            continue
        if A0.device != A_local.device or A0.dtype != A_local.dtype:
            A0 = A0.to(device=A_local.device, dtype=A_local.dtype)
        if tuple(A0.shape) != tuple(A_local.shape):
            continue

        if anchor_lam > 0:
            A_dir = A_local.float() / A_local.float().norm(dim=1, keepdim=True).clamp_min(eps)
            A0_dir = A0.float() / A0.float().norm(dim=1, keepdim=True).clamp_min(eps)
            # Sign-invariant row anchor keeps A/B row coordinates compatible without
            # forcing every client to learn the same row magnitude.
            row_cos = (A_dir * A0_dir).sum(dim=1).abs().clamp(max=1.0)
            anchor_terms.append((1.0 - row_cos).mean())

        if prox_lam > 0:
            prox_terms.append(torch.mean((A_local.float() - A0.float()) ** 2))

    if anchor_terms:
        loss = loss + anchor_lam * torch.stack(anchor_terms).mean()
    if prox_terms:
        loss = loss + prox_lam * torch.stack(prox_terms).mean()
    return loss


def _add_fedplora_v10_geometry_regularizer(loss, model, args):
    if not (
        is_fedplora_v10_family_agg(getattr(args, "agg_type", None))
        or is_fedplora_v11_family_agg(getattr(args, "agg_type", None))
        or is_fedplora_v12_family_agg(getattr(args, "agg_type", None))
        or is_fedplora_v13_a_sketch_agg(getattr(args, "agg_type", None))
    ):
        return loss

    a_ref = getattr(args, "_fedplora_v10_global_A_gpu", None)
    if not isinstance(a_ref, dict):
        a_ref = {}
    b_ref = getattr(args, "_fedplora_v10_round_start_B_gpu", None)
    if not isinstance(b_ref, dict):
        b_ref = {}

    anchor_lam = float(getattr(args, "v10_a_anchor_lambda", 1e-3) or 0.0)
    prox_lam = float(getattr(args, "v10_a_prox_lambda", 5e-4) or 0.0)
    b_prox_lam = float(getattr(args, "v10_b_prox_lambda", 1e-4) or 0.0)
    if anchor_lam <= 0 and prox_lam <= 0 and b_prox_lam <= 0:
        return loss

    eps = 1e-8
    a_anchor_terms = []
    a_prox_terms = []
    b_prox_terms = []
    for key, param in model.named_parameters():
        if "lora_A" in key and key.endswith("default.weight") and param.requires_grad:
            ref = a_ref.get(key)
            if ref is not None and tuple(ref.shape) == tuple(param.shape):
                if anchor_lam > 0:
                    p_dir = param.float() / param.float().norm(dim=1, keepdim=True).clamp_min(eps)
                    r_dir = ref.float() / ref.float().norm(dim=1, keepdim=True).clamp_min(eps)
                    row_cos = (p_dir * r_dir).sum(dim=1).abs().clamp(max=1.0)
                    a_anchor_terms.append((1.0 - row_cos).mean())
                if prox_lam > 0:
                    a_prox_terms.append(torch.mean((param.float() - ref.float()) ** 2))
        elif "lora_B" in key and key.endswith("default.weight") and param.requires_grad:
            ref = b_ref.get(key)
            if ref is not None and tuple(ref.shape) == tuple(param.shape) and b_prox_lam > 0:
                b_prox_terms.append(torch.mean((param.float() - ref.float()) ** 2))

    if a_anchor_terms:
        loss = loss + anchor_lam * torch.stack(a_anchor_terms).mean()
    if a_prox_terms:
        loss = loss + prox_lam * torch.stack(a_prox_terms).mean()
    if b_prox_terms:
        loss = loss + b_prox_lam * torch.stack(b_prox_terms).mean()
    return loss


def _fedplora_refresh_reg_tensor_gpu_cache(model, args):
    """
    `_fedplora_initial_A` / `_fedplora_global_A` live on CPU for checkpointing;
    copying them H2D inside every optimizer step dominated wall time. Materialize once
    per local training run.
    """
    args._fedplora_initial_A_gpu = None
    args._fedplora_global_A_gpu = None
    args._fedplora_v10_global_A_gpu = None
    args._fedplora_v10_round_start_B_gpu = None

    if is_fedplora_oneshot_family_agg(getattr(args, "agg_type", None)):
        init = getattr(args, "_fedplora_initial_A", None)
        if isinstance(init, dict) and init:
            gpu_map = {}
            for key, A_local in model.named_parameters():
                if key not in init:
                    continue
                if "lora_A" not in key or not key.endswith("default.weight"):
                    continue
                gpu_map[key] = init[key].to(device=A_local.device, dtype=A_local.dtype)
            if gpu_map:
                args._fedplora_initial_A_gpu = gpu_map

    if is_fedplora_multiround_agg(getattr(args, "agg_type", None)):
        Ag = getattr(args, "_fedplora_global_A", None)
        if isinstance(Ag, dict) and Ag:
            sd = model.state_dict()
            gpu_map = {}
            for kA_local, A_cpu in Ag.items():
                if kA_local not in sd or "lora_A" not in kA_local:
                    continue
                A_local = sd[kA_local]
                gpu_map[kA_local] = A_cpu.to(device=A_local.device, dtype=A_local.dtype)
            if gpu_map:
                args._fedplora_global_A_gpu = gpu_map

    if (
        is_fedplora_v10_family_agg(getattr(args, "agg_type", None))
        or is_fedplora_v11_family_agg(getattr(args, "agg_type", None))
        or is_fedplora_v12_family_agg(getattr(args, "agg_type", None))
        or is_fedplora_v13_a_sketch_agg(getattr(args, "agg_type", None))
    ):
        sd = model.state_dict()
        Ag = getattr(args, "_fedplora_global_A", None)
        if isinstance(Ag, dict) and Ag:
            gpu_map = {}
            for key, tensor in Ag.items():
                if key in sd and "lora_A" in key and key.endswith("default.weight"):
                    gpu_map[key] = tensor.to(device=sd[key].device, dtype=sd[key].dtype)
            if gpu_map:
                args._fedplora_v10_global_A_gpu = gpu_map
        Bg = getattr(args, "_fedplora_round_start_B", None)
        if isinstance(Bg, dict) and Bg:
            gpu_map = {}
            for key, tensor in Bg.items():
                if key in sd and "lora_B" in key and key.endswith("default.weight"):
                    gpu_map[key] = tensor.to(device=sd[key].device, dtype=sd[key].dtype)
            if gpu_map:
                args._fedplora_v10_round_start_B_gpu = gpu_map


def train_client(model, dataloader, args, client_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _fedplora_refresh_reg_tensor_gpu_cache(model, args)

    full_steps = len(dataloader) * args.local_epochs
    cap_steps = int(getattr(args, "train_max_steps_per_client", 0) or 0)
    steps_this_round = min(full_steps, cap_steps) if cap_steps > 0 else full_steps

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        trainable = list(model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    num_warmup_steps = int(steps_this_round * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=steps_this_round,
    )
    scaler = GradScaler()

    model.train()

    global_step = 0
    try:
        for epoch in range(args.local_epochs):
            for step, data in enumerate(
                tqdm(
                    dataloader,
                    leave=True,
                    dynamic_ncols=True,
                    desc=getattr(args, "_tqdm_desc", None),
                )
            ):
                if global_step >= steps_this_round:
                    break
                data = {k: v.to(device) for k, v in data.items()}

                with autocast():
                    outputs = model(**data)
                    loss = outputs.loss
                    loss = _add_fedplora_regularization(loss, model, args)
                    loss = _add_yoco_sparse(loss, model, args)
                    loss = _add_fedplora_oneshot_anchor(loss, model, args)
                    loss = _add_fedplora_v10_geometry_regularizer(loss, model, args)
                    loss = _add_yoco_sign_regularizer(loss, model, args)
                    loss = _add_feddat_teacher_regularizer(loss, model, args)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            if global_step >= steps_this_round:
                break

        return model.state_dict()
    finally:
        args._fedplora_initial_A_gpu = None
        args._fedplora_global_A_gpu = None
        args._fedplora_v10_global_A_gpu = None
        args._fedplora_v10_round_start_B_gpu = None


def compute_accuracy(model, dataloader):
    """Classification accuracy on a dataloader (no wandb)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            correct += int((preds == labels).sum().detach().cpu().item())
            total += int(labels.numel())
    return float(correct) / float(max(total, 1))


def evaluate_pfl_clients_acc_mean(client_models, val_dataloader, client_val_loaders=None):
    """
    Mean accuracy across clients.
    - If client_val_loaders is provided, evaluate client i on its own val shard.
    - Else evaluate all clients on the same global val_dataloader.
    """
    accs = []
    for i, model in enumerate(client_models):
        dl = client_val_loaders[i] if client_val_loaders is not None else val_dataloader
        accs.append(compute_accuracy(model, dl))
    return float(np.mean(accs)) if accs else 0.0


def compute_eval_metrics(model, dataloader, args):
    """Run evaluation on a dataloader; return scalar metrics (no wandb)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    eval_loss = 0.0
    all_predictions = []
    all_true_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            eval_loss += float(outputs.loss.detach().cpu().numpy())
            if args.task == "stsb":
                predictions = outputs.logits.squeeze().cpu().numpy()
                if np.ndim(predictions) == 0:
                    all_predictions.append(float(predictions))
                else:
                    all_predictions.extend(predictions.tolist())
            else:
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_predictions.extend(predictions.tolist())
            all_true_labels.extend(batch["labels"].cpu().numpy().tolist())

    n_batches = max(len(dataloader), 1)
    eval_loss /= n_batches
    metric1, metric2 = calculate_metrics(all_true_labels, all_predictions, args.task)
    return {
        "eval_loss": eval_loss,
        "metric1": float(metric1),
        "metric2": None if metric2 is None else float(metric2),
    }


def evaluate_pfl_clients(client_models, val_dataloader, args, client_sizes=None, client_val_loaders=None):
    """
    Personalized metrics: evaluate each client model on full val (scheme A) or
    per-client val shard (scheme B via client_val_loaders).
    Returns macro mean and sample-size-weighted mean for metric1 (and metric2 if all clients have it).
    """
    n = len(client_models)
    m1_list = []
    m2_list = []
    weights = (
        np.asarray(client_sizes, dtype=np.float64)
        if client_sizes is not None
        else np.ones(n, dtype=np.float64) / n
    )
    weights = weights / weights.sum()

    for i, model in enumerate(client_models):
        dl = client_val_loaders[i] if client_val_loaders is not None else val_dataloader
        m = compute_eval_metrics(model, dl, args)
        m1_list.append(m["metric1"])
        m2_list.append(m["metric2"])

    m1_macro = float(np.mean(m1_list))
    m1_weighted = float(np.sum(weights * np.asarray(m1_list)))
    m2_macro = None
    m2_weighted = None
    if all(x is not None for x in m2_list):
        m2_arr = np.asarray([float(x) for x in m2_list])
        m2_macro = float(np.mean(m2_arr))
        m2_weighted = float(np.sum(weights * m2_arr))

    return {
        "pfl_metric1_macro": m1_macro,
        "pfl_metric1_weighted": m1_weighted,
        "pfl_metric2_macro": m2_macro,
        "pfl_metric2_weighted": m2_weighted,
        "pfl_per_client_metric1": m1_list,
        "pfl_per_client_metric2": m2_list,
    }


def calculate_metrics(all_true_labels, all_predictions, task):
    if task == "cola":
        return accuracy_score(all_true_labels, all_predictions), matthews_corrcoef(
            all_true_labels, all_predictions
        )
    elif task in ["sst2", "qnli", "rte", "wnli"]:
        return accuracy_score(all_true_labels, all_predictions), None
    elif task == "mrpc":
        return f1_score(all_true_labels, all_predictions), accuracy_score(
            all_true_labels, all_predictions
        )
    elif task == "stsb":
        return (
            pearsonr(all_true_labels, all_predictions)[0],
            spearmanr(all_true_labels, all_predictions)[0],
        )
    elif task == "qqp":
        return accuracy_score(all_true_labels, all_predictions), f1_score(
            all_true_labels, all_predictions
        )
    elif task in ["mnli_matched", "mnli_mismatched"]:
        return accuracy_score(all_true_labels, all_predictions), None
    else:
        raise ValueError(f"Unknown task: {task}")


def task_metric_names(task):
    """
    Human-readable names for (metric1, metric2) — same pairing as calculate_metrics.
    Second name is None when that task has no metric2.
    """
    if task == "cola":
        return "accuracy", "mcc"
    if task in ["sst2", "qnli", "rte", "wnli", "mnli_matched", "mnli_mismatched"]:
        return "accuracy", None
    if task == "mrpc":
        return "f1", "accuracy"
    if task == "stsb":
        return "pearson_r", "spearman_r"
    if task == "qqp":
        return "accuracy", "f1"
    raise ValueError(f"Unknown task: {task}")


def evaluate_global_model(
    global_model, dataloader, args, max_metric1, max_metric2
):
    m = compute_eval_metrics(global_model, dataloader, args)
    eval_loss = m["eval_loss"]
    metric1 = m["metric1"]
    metric2 = m["metric2"]

    if metric1 > max_metric1:
        max_metric1 = metric1

    if metric2 is not None and metric2 > max_metric2:
        max_metric2 = metric2

    print(f"{args.task} - Eval Loss: {eval_loss:.4f}, Metric 1: {metric1:.4f}")
    if metric2 is not None:
        print(f"{args.task} - Metric 2: {metric2:.4f}")
    print(f"{args.task} - Max Metric 1: {max_metric1:.4f}")
    if max_metric2 is not None:
        print(f"{args.task} - Max Metric 2: {max_metric2:.4f}")

    return max_metric1, max_metric2, m


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_client_e2e(model, train_dataset, val_dataset, tokenizer, args):
    num_epochs = args.local_epochs  # or whatever number of epochs you want
    per_device_train_batch_size = args.batch_size
    num_training_steps = len(train_dataset) * num_epochs // per_device_train_batch_size
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warmup

    optimizer = torch.optim.AdamW(model.parameters())

    # Define training arguments
    training_args = TrainingArguments(
        # Directory to save the model
        output_dir="./models_trained/gpt4/dump/models/gpt2-e2e-lora_gpt4",
        overwrite_output_dir=True,
        logging_dir="./models_trained/gpt4/dump/logs/gpt2-e2e-lora_gpt4",  # Directory for logs
        per_device_train_batch_size=args.batch_size,  # Adjust based on your GPU capacity
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",
        num_train_epochs=num_epochs,  # Number of training epochs
        learning_rate=args.lr,  # Learning rate for LoRA parameters
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        report_to="none",
        run_name="fed-lora",
        logging_steps=100,  # Log every 100 steps
    )

    class _YocoTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            loss = _add_fedplora_regularization(loss, model, args)
            loss = _add_yoco_sparse(loss, model, args)
            return (loss, outputs) if return_outputs else loss

    # Initialize the trainer
    trainer = _YocoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        optimizers=(
            optimizer,
            get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps),
        ),
    )

    # Train the model
    trainer.train()
    return model


def gen_and_save(model, dataloader, tokenizer, args):
    device = args.device
    model.to(device)
    model.eval()

    all_predictions = []

    all_inputs = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):

            inputs = {k: v.to(device) for k, v in batch.items()}

            # Generate predictions (starting from after the MR)
            generated = model.generate(
                input_ids=inputs["input_ids"],  # Input MR as prompt
                attention_mask=inputs["attention_mask"],
                max_length=inputs["input_ids"].shape[1]
                + 50,  # Allow space for generation after MR
                num_return_sequences=1,
                no_repeat_ngram_size=4,
                do_sample=True,
                num_beams=10,
                penalty_alpha=0.9,
                pad_token_id=tokenizer.eos_token_id,  # Ensure padding works correctly
            )
            # Decode the generated predictions, excluding the input MR tokens
            # We slice the generated tokens to remove the input MR part

            input_seq = tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
            predictions = [
                tokenizer.decode(
                    generated[i][len(inputs["input_ids"][i]) :],
                    skip_special_tokens=True,
                )
                for i in range(generated.shape[0])
            ]
            # Collect predictions and references
            all_inputs.extend(input_seq)
            all_predictions.extend(predictions)
            # all_references.extend(references)

    return all_predictions, all_inputs


def process_lists(input_list, second_list, third_list):
    result1 = []
    result2 = []
    result3 = []
    current_group = []
    current_item = None
    second_list_index = 0

    for item in input_list:
        if item != current_item:
            if current_group:
                result1.append(current_group)
                result2.append(current_item)
                result3.append(third_list[second_list_index - 1])
            current_item = item
            current_group = [second_list[second_list_index]]
            second_list_index += 1
        else:
            if second_list_index < len(second_list):
                current_group.append(second_list[second_list_index])
                second_list_index += 1

    if current_group:
        result1.append(current_group)

    return result1, result2, result3


def evaluate_e2e_save_text(model, test_data, tokenizer, args):

    def preprocess_function2(examples):
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]

        # Combine the input-output pair into a single text
        model_inputs = [f"{input_} ->" for input_, target in zip(inputs, targets)]

        # Tokenize the combined inputs
        tokenized_inputs = tokenizer(
            model_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Labels are the same as input_ids but shift them for next-token prediction
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

        # Set the labels to -100 where attention mask is 0 (this will ignore padding in loss computation)
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100

        return tokenized_inputs

    tokenized_test_dataset = test_data.map(preprocess_function2, batched=True)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(
        ["meaning_representation", "human_reference"]
    )
    tokenized_test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    test_dataloader = create_dataloader(tokenized_test_dataset, args)
    all_predictions, all_inputs = gen_and_save(model, test_dataloader, tokenizer, args)
    all_references = test_data[0 : len(all_predictions)]["human_reference"]

    all_references_new, all_inputs_new, all_predictions_new = process_lists(
        all_inputs, all_references, all_predictions
    )

    path_pred = args.run_dir + "/predictions.txt"
    path_ref = args.run_dir + "/refs_exact.txt"

    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    with open(path_pred, "w") as file:
        for item in all_predictions_new:
            file.write(item.strip() + "\n")

    with open(path_ref, "w") as file:
        for str_list in all_references_new:
            for item in str_list:
                file.write(item.strip() + "\n")

            file.write("\n")

    # Compute a lightweight metrics suite (string-based).
    # Notes:
    # - BLEU/NIST use whitespace tokenization.
    # - METEOR uses NLTK's reference-aware score (string inputs).
    # - ROUGE-L uses rouge_score.
    # - CIDEr uses pycocoevalcap (may be slower).
    preds = [p.strip() for p in all_predictions_new]
    refs = [[r.strip() for r in rs] for rs in all_references_new]
    try:
        bleu = corpus_bleu([[r.split() for r in rs] for rs in refs], [p.split() for p in preds])
    except Exception:
        bleu = None
    try:
        nist = corpus_nist([[r.split() for r in rs] for rs in refs], [p.split() for p in preds], n=5)
    except Exception:
        nist = None
    try:
        meteor = float(
            np.mean([meteor_score(rs, p) for rs, p in zip(refs, preds)])
        )
    except Exception:
        meteor = None
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rougeL = float(
            np.mean([scorer.score(" ".join(rs), p)["rougeL"].fmeasure for rs, p in zip(refs, preds)])
        )
    except Exception:
        rougeL = None
    try:
        if Cider is None:
            cider = None
        else:
            gts = {i: refs[i] for i in range(len(preds))}
            res = {i: [preds[i]] for i in range(len(preds))}
            cider = Cider().compute_score(gts, res)[0]
            cider = float(cider)
    except Exception:
        cider = None

    metrics = {"bleu": bleu, "nist": nist, "meteor": meteor, "rougeL": rougeL, "cider": cider}
    with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
