import argparse
import json
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm

from data_utils import (
    build_domain_benchmark_from_jsonl,
    create_domain_client_dataloaders,
    create_domain_eval_dataloader,
    group_rows_by_domain,
    load_domain_sft_benchmark,
)
from fed_agg import (
    aggregate_models_ffa,
    aggregate_models_fedex,
    aggregate_models_gp_lora,
    aggregate_models_normal,
    build_fedplora_upload_package,
    broadcast_fedplora_shared_state,
    extract_fedplora_local_state,
    load_fedplora_local_state,
    load_partial_state_dict,
)
from models import (
    create_peft_causal_lm_model,
    create_peft_causal_lm_ffa_model,
    init_gp_lora_adapters,
)
from train_eval import train_client
from utils import (
    estimate_round_communication_bytes,
    get_fedplora_shared_param_names,
    get_trainable_param_names,
    is_fedplora_agg,
    is_fedplora_shared_param_name,
    is_lora_a_param_name,
    restore_logging,
    setup_run_logging,
)


parser = argparse.ArgumentParser(description="Federated SFT with FedPLoRA")
parser.add_argument("--model", type=str, required=True, help="Base causal LM model path or HF id")
parser.add_argument("--benchmark_dir", type=str, default="", help="Path to prepared benchmark split dir, e.g. data/domain_benchmark/seed_42")
parser.add_argument("--benchmark_jsonl", type=str, default="", help="Optional raw JSONL path; if provided with --build_benchmark, the benchmark will be built automatically")
parser.add_argument("--build_benchmark", action="store_true", help="Build benchmark from --benchmark_jsonl before training")
parser.add_argument("--benchmark_output_dir", type=str, default="data/domain_benchmark", help="Where to save built benchmark")
parser.add_argument("--num_clients_per_domain", type=int, default=5, help="Clients per domain when building benchmark")
parser.add_argument("--min_samples_per_client", type=int, default=50, help="Minimum samples per client when building benchmark")
parser.add_argument("--agg_type", type=str, default="gp_lora", help="Aggregation type")
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--num_clients", type=int, default=0, help="If 0, infer from benchmark")
parser.add_argument("--local_epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--rslora", action="store_true")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--warmup_ratio", type=float, default=0.03)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--trust_remote_code", action="store_true")
parser.add_argument(
    "--torch_dtype",
    type=str,
    default="auto",
    choices=["auto", "bfloat16", "float16", "float32"],
)
parser.add_argument(
    "--target_modules",
    type=str,
    default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    help="Comma-separated LoRA target modules for causal LM backbones",
)
parser.add_argument("--client_state_dir", type=str, default="artifacts/domain_client_states", help="Directory to store per-client local FedPLoRA states for large-model sequential training")
parser.add_argument("--save_client_state_to_disk", action="store_true", help="Persist per-client FedPLoRA local states to disk instead of keeping them in CPU memory")
parser.add_argument("--metrics_output_dir", type=str, default="artifacts/sft_metrics", help="Directory for round-wise evaluation metrics")
parser.add_argument("--gp_align_lambda", type=float, default=0.01)
parser.add_argument("--gp_prox_lambda", type=float, default=0.001)
parser.add_argument("--gp_orth_lambda", type=float, default=1e-4)
parser.add_argument("--gp_consensus_power", type=float, default=2.0)
parser.add_argument("--gp_agg_momentum", type=float, default=0.5)

args = parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_or_load_benchmark(args):
    if args.build_benchmark:
        if not args.benchmark_jsonl:
            raise ValueError("--build_benchmark requires --benchmark_jsonl")
        info = build_domain_benchmark_from_jsonl(
            input_path=args.benchmark_jsonl,
            output_dir=args.benchmark_output_dir,
            num_clients_per_domain=args.num_clients_per_domain,
            min_samples_per_client=args.min_samples_per_client,
            seed=args.seed,
        )
        split_dir = info["split_dir"]
    else:
        if not args.benchmark_dir:
            raise ValueError("provide --benchmark_dir or use --build_benchmark")
        split_dir = args.benchmark_dir
    return load_domain_sft_benchmark(split_dir), split_dir


def compute_lm_loss(model, dataloader, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.detach().cpu().item())
            steps += 1
    return total_loss / max(steps, 1)


def evaluate_domain_macro(client_models, domain_rows, tokenizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_domain = group_rows_by_domain(domain_rows)
    metrics = {}
    for domain, rows in sorted(by_domain.items()):
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        losses = [compute_lm_loss(model, dl, device) for model in client_models]
        metrics[domain] = float(np.mean(losses))
    macro = float(np.mean(list(metrics.values()))) if metrics else float("nan")
    worst = float(max(metrics.values())) if metrics else float("nan")
    return metrics, macro, worst


def _client_state_path(base_dir, client_id):
    return os.path.join(base_dir, f"client_{int(client_id):03d}.pt")


def _save_client_local_state(local_state, base_dir, client_id):
    os.makedirs(base_dir, exist_ok=True)
    path = _client_state_path(base_dir, client_id)
    torch.save(local_state, path)
    return path


def _load_client_local_state(base_dir, client_id):
    path = _client_state_path(base_dir, client_id)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def _ensure_sequential_fedplora_local_states(model, client_ids, args):
    state_dir = os.path.join(args.client_state_dir, f"seed_{args.seed}")
    if getattr(args, "save_client_state_to_disk", False):
        os.makedirs(state_dir, exist_ok=True)
        for client_id in client_ids:
            path = _client_state_path(state_dir, client_id)
            if not os.path.exists(path):
                torch.save(extract_fedplora_local_state(model), path)
        return {"mode": "disk", "state_dir": state_dir}

    initial = extract_fedplora_local_state(model)
    local_states = {
        int(client_id): {k: v.clone() for k, v in initial.items()} for client_id in client_ids
    }
    return {"mode": "memory", "local_states": local_states, "state_dir": state_dir}


def _get_client_local_state(client_store, client_id):
    if client_store["mode"] == "disk":
        state = _load_client_local_state(client_store["state_dir"], client_id)
        if state is None:
            raise FileNotFoundError(
                f"missing client local state for client_id={client_id}: "
                f"{_client_state_path(client_store['state_dir'], client_id)}"
            )
        return state
    return client_store["local_states"][int(client_id)]


def _set_client_local_state(client_store, client_id, local_state):
    if client_store["mode"] == "disk":
        _save_client_local_state(local_state, client_store["state_dir"], client_id)
    else:
        client_store["local_states"][int(client_id)] = local_state


def _evaluate_domain_macro_sequential(global_model, client_ids, client_store, domain_rows, tokenizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_domain = group_rows_by_domain(domain_rows)
    shared_state = {
        k: v.detach().cpu().clone()
        for k, v in global_model.state_dict().items()
        if is_fedplora_shared_param_name(k, get_trainable_param_names(global_model))
    }
    metrics = {}
    for domain, rows in sorted(by_domain.items()):
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        losses = []
        for client_id in client_ids:
            broadcast_fedplora_shared_state(global_model, shared_state)
            local_state = _get_client_local_state(client_store, client_id)
            load_fedplora_local_state(global_model, local_state)
            losses.append(compute_lm_loss(global_model, dl, device))
        metrics[domain] = float(np.mean(losses))
    macro = float(np.mean(list(metrics.values()))) if metrics else float("nan")
    worst = float(max(metrics.values())) if metrics else float("nan")
    return metrics, macro, worst


def _metrics_path(args, split_dir):
    split_tag = os.path.basename(os.path.normpath(split_dir))
    model_tag = os.path.basename(os.path.normpath(args.model.rstrip("/")))
    fname = (
        f"{args.agg_type}_{model_tag}_{split_tag}_"
        f"r{args.rounds}_e{args.local_epochs}_seed{args.seed}.json"
    )
    os.makedirs(args.metrics_output_dir, exist_ok=True)
    return os.path.join(args.metrics_output_dir, fname)


def _write_metrics_file(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def federated_sft(args):
    benchmark, split_dir = build_or_load_benchmark(args)
    print(f"[benchmark] loaded from {split_dir}")
    print(f"[benchmark] domains={sorted(benchmark['domain_stats'].keys())}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    client_ids, client_dataloaders = create_domain_client_dataloaders(
        benchmark["train"], tokenizer, args
    )
    client_sizes = [len(dl.dataset) for dl in client_dataloaders]
    args.num_clients = len(client_ids)

    if args.agg_type == "ffa":
        global_model = create_peft_causal_lm_ffa_model(args)
    else:
        global_model = create_peft_causal_lm_model(args)

    comm_info = estimate_round_communication_bytes(
        global_model.state_dict(),
        args.agg_type,
        trainable_param_names=get_trainable_param_names(global_model),
    )
    print(
        f"[setup] agg_type={args.agg_type} num_clients={args.num_clients} "
        f"comm_down_bytes_per_client={comm_info['down_bytes_per_client']} "
        f"comm_up_bytes_per_client={comm_info['up_bytes_per_client']}"
    )

    if is_fedplora_agg(args.agg_type):
        init_gp_lora_adapters(global_model)
        client_store = _ensure_sequential_fedplora_local_states(
            global_model, client_ids, args
        )
    else:
        client_store = None

    best_domain_macro = float("inf")
    best_worst_domain = float("inf")
    metrics_history = {
        "args": vars(args).copy(),
        "benchmark_dir": split_dir,
        "rounds": [],
    }

    for round_idx in range(args.rounds):
        print(f"Round {round_idx + 1}/{args.rounds}")

        if is_fedplora_agg(args.agg_type):
            args._gp_lora_client_sizes = client_sizes
            shared_names = get_fedplora_shared_param_names(global_model)
            gp_global_state = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
                if k in shared_names
            }
            args._gp_lora_global_A = {
                k: v.detach().cpu().clone()
                for k, v in gp_global_state.items()
                if is_lora_a_param_name(k)
            }
        else:
            round_global_state = {
                k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()
            }

        client_states_for_agg = []
        for i, client_id in enumerate(client_ids):
            args._tqdm_desc = f"R{round_idx + 1}/{args.rounds} client{i + 1}/{args.num_clients}"
            if is_fedplora_agg(args.agg_type):
                broadcast_fedplora_shared_state(global_model, gp_global_state)
                local_state = _get_client_local_state(client_store, client_id)
                load_fedplora_local_state(global_model, local_state)
            else:
                load_partial_state_dict(global_model, round_global_state)
            train_client(global_model, client_dataloaders[i], args, client_idx=i)
            client_states_for_agg.append(
                {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
            )
            if is_fedplora_agg(args.agg_type):
                updated_local_state = extract_fedplora_local_state(global_model)
                _set_client_local_state(client_store, client_id, updated_local_state)

        args._tqdm_desc = None

        if args.agg_type == "normal":
            global_model = aggregate_models_normal(global_model, client_states_for_agg)
        elif args.agg_type == "fedex":
            global_model = aggregate_models_fedex(
                global_model, client_states_for_agg, args
            )
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_states_for_agg)
        elif is_fedplora_agg(args.agg_type):
            uploads = [
                build_fedplora_upload_package(client_states_for_agg[i], client_sizes[i])
                for i in range(args.num_clients)
            ]
            global_model = aggregate_models_gp_lora(global_model, uploads, args)
        else:
            raise ValueError(f"Unknown agg_type: {args.agg_type}")

        if is_fedplora_agg(args.agg_type):
            domain_metrics, domain_macro, worst_domain = _evaluate_domain_macro_sequential(
                global_model,
                client_ids,
                client_store,
                benchmark["test_domain"],
                tokenizer,
                args,
            )
        else:
            state_dir = os.path.join(args.client_state_dir, f"eval_seed_{args.seed}")
            os.makedirs(state_dir, exist_ok=True)
            eval_client_ids = list(range(len(client_states_for_agg)))
            eval_store = {"mode": "disk", "state_dir": state_dir}
            for idx, state in enumerate(client_states_for_agg):
                torch.save(state, _client_state_path(state_dir, idx))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            by_domain = group_rows_by_domain(benchmark["test_domain"])
            metrics = {}
            for domain, rows in sorted(by_domain.items()):
                dl = create_domain_eval_dataloader(rows, tokenizer, args)
                losses = []
                for idx in eval_client_ids:
                    state = _get_client_local_state(eval_store, idx)
                    load_partial_state_dict(global_model, state)
                    losses.append(compute_lm_loss(global_model, dl, device))
                metrics[domain] = float(np.mean(losses))
            domain_metrics = metrics
            domain_macro = float(np.mean(list(metrics.values()))) if metrics else float("nan")
            worst_domain = float(max(metrics.values())) if metrics else float("nan")
        best_domain_macro = min(best_domain_macro, domain_macro)
        best_worst_domain = min(best_worst_domain, worst_domain)
        metrics_str = " | ".join(
            [f"{d}_loss={v:.4f}" for d, v in domain_metrics.items()]
        )
        print(
            f"[eval] round={round_idx + 1} domain_macro_loss={domain_macro:.4f} "
            f"best_domain_macro_loss={best_domain_macro:.4f} "
            f"worst_domain_loss={worst_domain:.4f} "
            f"best_worst_domain_loss={best_worst_domain:.4f} | {metrics_str}"
        )

        metrics_history["rounds"].append(
            {
                "round": round_idx + 1,
                "domain_macro_loss": domain_macro,
                "best_domain_macro_loss": best_domain_macro,
                "worst_domain_loss": worst_domain,
                "best_worst_domain_loss": best_worst_domain,
                "domain_metrics": domain_metrics,
            }
        )

    metrics_path = _metrics_path(args, split_dir)
    _write_metrics_file(metrics_path, metrics_history)
    print(f"[metrics] saved to {metrics_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    set_seed(args.seed)
    log_file, orig_out, orig_err, _ = setup_run_logging(args, filename_prefix="sft")
    try:
        federated_sft(args)
    finally:
        restore_logging(log_file, orig_out, orig_err)
