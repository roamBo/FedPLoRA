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
import argparse
import warnings
from sklearn.metrics import matthews_corrcoef
import numpy as np
from train_eval import *
from fed_agg import *
import json
from utils import setup_run_logging, restore_logging, is_fedplora_agg

parser = argparse.ArgumentParser(description="Federated Learning with LoRA")

parser.add_argument(
    "--agg_type", type=str, default="fedex", help="Type of aggregation"
)
parser.add_argument("--rounds", type=int, default=6, help="Number of rounds")
parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
parser.add_argument(
    "--local_epochs", type=int, default=3, help="Number of local epochs"
)
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--lora_r", type=int, default=4, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha value")
parser.add_argument(
    "--lora_dropout", type=float, default=0.1, help="LoRA dropout value"
)
parser.add_argument("--rslora", action="store_true", help="Use RSLoRA")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
parser.add_argument(
    "--max_seq_length", type=int, default=128, help="Maximum sequence length"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
parser.add_argument("--idx", type=int, default=0, help="Index of the save folder")
parser.add_argument("--log", action="store_true", help="Log the results")
parser.add_argument("--run_dir", type=str, help="Directory to store logs")
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

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

warnings.filterwarnings("ignore")


def get_next_run_number(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1

    existing_runs = [int(d) for d in os.listdir(base_dir) if d.isdigit()]
    return max(existing_runs, default=0) + 1


def save_args(args, directory):
    args_file = os.path.join(directory, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)


def federated_learning(task):

    train_data, val_data, test_data, tokenizer = create_e2e_data()
    client_data = create_client_dataloaders_nlg(train_data, args)
    client_sizes = [len(cd) for cd in client_data]

    if args.agg_type == "ffa":
        global_model = create_peft_gpt2_model_e2e_ffa(args)
    else:
        global_model = create_peft_gpt2_model_e2e(args)

    client_models = []
    for _ in range(args.num_clients):
        if args.agg_type == "ffa":
            client_models.append(create_peft_gpt2_model_e2e_ffa(args))
        else:
            client_models.append(create_peft_gpt2_model_e2e(args))

    if is_fedplora_agg(args.agg_type):
        init_gp_lora_adapters(global_model)
        for m in client_models:
            m.load_state_dict(global_model.state_dict())

    for round in range(args.rounds):
        print(f"Round {round + 1}/{args.rounds}")

        # Broadcast
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
        else:
            gp_global_state = None

        # Train on clients
        for client in range(args.num_clients):
            client_model = client_models[client]
            if is_fedplora_agg(args.agg_type):
                broadcast_fedplora_shared_state(client_model, gp_global_state)
            else:
                client_model.load_state_dict(global_model.state_dict())

            client_models[client] = train_client_e2e(
                client_model, client_data[client], val_data, tokenizer, args
            )

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
            global_model = aggregate_models_gp_lora(global_model, client_uploads, args)
            global_state = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
                if is_fedplora_shared_param_name(k)
            }
            for client_model in client_models:
                broadcast_fedplora_shared_state(client_model, global_state)
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)

        args.idx = round + 1

        # Evaluate: client-mean metrics on test set (per-client then macro mean).
        per_client = []
        base_dir = "text_store_new/" + args.agg_type
        run_number = get_next_run_number(base_dir)
        for i, m in enumerate(client_models):
            if args.log:
                run_dir = os.path.join(base_dir, str(run_number), f"client{i}")
                os.makedirs(run_dir, exist_ok=True)
                save_args(args, run_dir)
                args.run_dir = run_dir
            else:
                args.run_dir = os.path.join("log", "e2e_artifacts", args.agg_type, f"round{round+1}", f"client{i}")
                os.makedirs(args.run_dir, exist_ok=True)
            metrics = evaluate_e2e_save_text(m, test_data, tokenizer, args)
            per_client.append(metrics)

        def _mean(key):
            vals = [x.get(key) for x in per_client if x.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        mean_metrics = {k: _mean(k) for k in ["bleu", "nist", "meteor", "rougeL", "cider"]}
        print(f"E2E_client_mean_metrics: {mean_metrics}")

    return global_model


# Main execution
if __name__ == "__main__":
    task = "e2e"
    # Add a task name for shared logging helpers.
    args.task = task
    log_file, orig_out, orig_err, _ = setup_run_logging(args, filename_prefix="gpt2")
    try:
        model = federated_learning(task)
    finally:
        restore_logging(log_file, orig_out, orig_err)
