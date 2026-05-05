import argparse
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from utilities.data_utils import *
from methods.fedavg_normal import aggregate_models_normal
from methods.fedex import aggregate_models_fedex
from methods.ffa_lora import aggregate_models_ffa
from methods.fedp_lora import aggregate_models_gp_lora, build_fedplora_upload_package
from utilities.models import *
from utilities.state_dict_ops import broadcast_fedplora_shared_state
from utilities.train_eval import *
from utilities.utils import *

parser = argparse.ArgumentParser(description="Federated Learning with LoRA")

parser.add_argument(
    "--task", type=str, default="cola", help="GLUE task to fine-tune on"
)
parser.add_argument("--model", type=str, default="roberta-base", help="Model name")
parser.add_argument("--lora_r", type=int, default=4, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha value")
parser.add_argument(
    "--lora_dropout", type=float, default=0.1, help="LoRA dropout value"
)
parser.add_argument("--rslora", action="store_true", help="Use RSLoRA")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument(
    "--agg_type", type=str, default="fedex", help="Type of aggregation"
)
parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
parser.add_argument("--rounds", type=int, default=50, help="Number of rounds")
parser.add_argument(
    "--local_epochs", type=int, default=3, help="Number of local epochs"
)
parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
parser.add_argument(
    "--max_seq_length", type=int, default=512, help="Maximum sequence length"
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--partition",
    type=str,
    default="iid",
    choices=["iid", "dirichlet"],
    help="Client data split: iid (uniform random) or dirichlet (label skew)",
)
parser.add_argument(
    "--dirichlet_alpha",
    type=float,
    default=1.0,
    help="Dirichlet concentration when partition=dirichlet (smaller => stronger skew)",
)
parser.add_argument(
    "--dirichlet_alphas",
    type=str,
    default="",
    help="If set (e.g. 0.05,0.1,0.5,1,10), run one experiment per alpha with partition=dirichlet",
)
parser.add_argument(
    "--print_partition_stats",
    action="store_true",
    help="Print per-client label counts after Dirichlet split",
)
parser.add_argument(
    "--pfl_eval_split",
    type=str,
    default="global_val",
    choices=["global_val", "client_val"],
    help="Eval split for client-mean metrics: global_val (each client on full val) or client_val (each client on its own val shard).",
)
parser.add_argument(
    "--gp_align_lambda",
    type=float,
    default=0.01,
    help="FedPLoRA alignment regularization weight.",
)
parser.add_argument(
    "--gp_prox_lambda",
    type=float,
    default=0.001,
    help="FedPLoRA A-to-global proximal regularization weight.",
)
parser.add_argument(
    "--gp_orth_lambda",
    type=float,
    default=1e-4,
    help="FedPLoRA A orthogonality regularization weight.",
)
parser.add_argument(
    "--gp_consensus_power",
    type=float,
    default=2.0,
    help="Power used for consensus-aware row weighting during FedPLoRA aggregation.",
)
parser.add_argument(
    "--gp_agg_momentum",
    type=float,
    default=0.5,
    help="Server-side momentum when updating the global A basis in FedPLoRA.",
)

args = parser.parse_args()


def _print_round_metrics_gp_lora(
    round_idx, args, pfl, n1, n2, max_pfl_m1, max_pfl_m2
):
    """Terminal line: PFL client-mean metric1/metric2 + per-client (same semantics as calculate_metrics)."""
    pfl_m1 = pfl["pfl_metric1_macro"]
    pfl_m2 = pfl["pfl_metric2_macro"]
    parts = [
        f"Round {round_idx + 1}/{args.rounds}",
        f"PFL_mean_{n1}={pfl_m1:.8f}",
        f"max_PFL_mean_{n1}={max_pfl_m1:.8f}",
    ]
    if pfl_m2 is not None and n2 is not None:
        parts.append(f"PFL_mean_{n2}={pfl_m2:.8f}")
        max_m2s = (
            max_pfl_m2
            if max_pfl_m2 != float("-inf")
            else float("nan")
        )
        parts.append(f"max_PFL_mean_{n2}={max_m2s:.8f}")
    pc1 = ", ".join(
        f"c{i}={float(x):.6f}"
        for i, x in enumerate(pfl["pfl_per_client_metric1"])
    )
    parts.append(f"PFL_per_client_{n1}: {pc1}")
    if (
        pfl_m2 is not None
        and n2 is not None
        and pfl["pfl_per_client_metric2"]
        and all(x is not None for x in pfl["pfl_per_client_metric2"])
    ):
        pc2 = ", ".join(
            f"c{i}={float(x):.6f}"
            for i, x in enumerate(pfl["pfl_per_client_metric2"])
        )
        parts.append(f"PFL_per_client_{n2}: {pc2}")
    print(" | ".join(parts))


def federated_learning(task):

    train_data, val_data, test_data = load_and_preprocess_data(task)

    num_labels = len(set(train_data["labels"]))

    if args.task == "stsb":
        num_labels = 1

    client_dataloaders = create_client_dataloaders(train_data, args)
    client_sizes = [len(dl.dataset) for dl in client_dataloaders]
    val_dataloader = create_dataloader(val_data, args)
    client_val_loaders = None
    if getattr(args, "pfl_eval_split", "global_val") == "client_val":
        client_val_loaders = create_client_val_dataloaders(val_data, args)

    max_pfl_m1 = float("-inf")
    max_pfl_m2 = float("-inf")
    max_cm_m1 = float("-inf")
    max_cm_m2 = float("-inf")
    n1, n2 = task_metric_names(args.task)

    if args.agg_type == "ffa":
        global_model = create_peft_FFA_model(num_labels, args)
    else:
        global_model = create_peft_model(num_labels, args)

    comm_info = estimate_round_communication_bytes(
        global_model.state_dict(),
        args.agg_type,
        trainable_param_names=get_trainable_param_names(global_model),
    )
    down_b = comm_info["down_bytes_per_client"]
    up_b = comm_info["up_bytes_per_client"]
    comm_per_round = args.num_clients * (down_b + up_b)
    print(
        f"[setup] task={args.task} agg_type={args.agg_type} "
        f"comm_down_bytes_per_client={down_b} comm_up_bytes_per_client={up_b} "
        f"comm_bytes_per_round={comm_per_round}"
    )

    client_models = []

    for i in range(args.num_clients):

        if args.agg_type == "ffa":
            client_model = create_peft_FFA_model(num_labels, args)
        else:
            client_model = create_peft_model(num_labels, args)

        client_models.append(client_model)

    if is_fedplora_agg(args.agg_type):
        init_gp_lora_adapters(global_model)
        for m in client_models:
            m.load_state_dict(global_model.state_dict())

    for round_idx in range(args.rounds):
        print(f"Round {round_idx + 1}/{args.rounds}")

        gp_global_state = None

        if is_fedplora_agg(args.agg_type):
            args._gp_lora_client_sizes = client_sizes
            gp_global_state = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
                if is_fedplora_shared_param_name(k)
            }
            args._gp_lora_global_A = {
                k: v.detach().cpu().clone()
                for k, v in gp_global_state.items()
                if is_lora_a_param_name(k)
            }

        for i in range(args.num_clients):
            client_model = client_models[i]
            args._tqdm_desc = f"R{round_idx + 1}/{args.rounds} client{i + 1}/{args.num_clients}"

            if is_fedplora_agg(args.agg_type):
                broadcast_fedplora_shared_state(client_model, gp_global_state)
            else:
                client_model.load_state_dict(global_model.state_dict())
            train_client(client_model, client_dataloaders[i], args, client_idx=i)

        args._tqdm_desc = None

        if args.agg_type == "normal":
            global_model = aggregate_models_normal(global_model, client_models)
        elif args.agg_type == "fedex":
            global_model = aggregate_models_fedex(
                global_model, client_models, args
            )
        elif is_fedplora_agg(args.agg_type):
            client_uploads = [
                build_fedplora_upload_package(client_models[i], client_sizes[i])
                for i in range(args.num_clients)
            ]
            global_model = aggregate_models_gp_lora(
                global_model, client_uploads, args
            )
            global_state = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
                if is_fedplora_shared_param_name(k)
            }
            for client_model in client_models:
                broadcast_fedplora_shared_state(client_model, global_state)
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)

        if is_fedplora_agg(args.agg_type):
            pfl = evaluate_pfl_clients(
                client_models,
                val_dataloader,
                args,
                client_sizes=client_sizes,
                client_val_loaders=client_val_loaders,
            )
            pfl_m1 = pfl["pfl_metric1_macro"]
            pfl_m2 = pfl["pfl_metric2_macro"]
            if pfl_m1 > max_pfl_m1:
                max_pfl_m1 = pfl_m1
            if pfl_m2 is not None and pfl_m2 > max_pfl_m2:
                max_pfl_m2 = pfl_m2

            _print_round_metrics_gp_lora(
                round_idx,
                args,
                pfl,
                n1,
                n2,
                max_pfl_m1,
                max_pfl_m2,
            )
        else:
            pfl = evaluate_pfl_clients(
                client_models,
                val_dataloader,
                args,
                client_sizes=client_sizes,
                client_val_loaders=client_val_loaders,
            )
            cm_m1 = pfl["pfl_metric1_macro"]
            cm_m2 = pfl["pfl_metric2_macro"]
            if cm_m1 > max_cm_m1:
                max_cm_m1 = cm_m1
            if cm_m2 is not None and cm_m2 > max_cm_m2:
                max_cm_m2 = cm_m2
            print_round_metrics_client_mean(
                round_idx,
                args.rounds,
                pfl,
                n1,
                n2,
                max_cm_m1,
                max_cm_m2,
                tag="CM",
            )


# Main execution
if __name__ == "__main__":
    alphas_str = (args.dirichlet_alphas or "").strip()
    if alphas_str:
        alphas = [float(x.strip()) for x in alphas_str.split(",") if x.strip()]
        for alpha in alphas:
            args.dirichlet_alpha = alpha
            args.partition = "dirichlet"
            log_file, orig_out, orig_err, _ = setup_run_logging(args)
            try:
                print(f"=== run dirichlet_alpha={alpha} ===")
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                federated_learning(args.task)
            finally:
                restore_logging(log_file, orig_out, orig_err)
    else:
        log_file, orig_out, orig_err, _ = setup_run_logging(args)
        try:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            federated_learning(args.task)
        finally:
            restore_logging(log_file, orig_out, orig_err)
