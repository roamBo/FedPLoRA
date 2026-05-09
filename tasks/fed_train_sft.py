import argparse
import json
import os
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from utilities.data_utils import (
    build_domain_benchmark_from_jsonl,
    create_domain_client_dataloaders,
    create_domain_eval_dataloader,
    group_rows_by_client,
    group_rows_by_domain,
    load_domain_sft_benchmark,
)
from methods.fedavg_normal import aggregate_models_normal
from methods.fedex import aggregate_models_fedex
from methods.fdlora import aggregate_models_fdlora
from methods.ffa_lora import aggregate_models_ffa
from methods.fedsa_lora import aggregate_models_fedsa_lora
from methods.flora import aggregate_models_flora
from methods.hetlora import aggregate_models_hetlora
from methods.lora_a2 import aggregate_models_lora_a2
from methods.fedp_lora import aggregate_models_fedplora, build_fedplora_upload_package
from methods.yoco import aggregate_models_yoco
from utilities.models import (
    create_peft_causal_lm_model,
    create_peft_causal_lm_ffa_model,
    init_fedplora_adapters,
)
from utilities.state_dict_ops import (
    broadcast_fedplora_shared_state,
    extract_fedplora_local_state,
    load_fedplora_local_state,
    load_partial_state_dict,
)
from utilities.train_eval import train_client
from utilities.utils import (
    estimate_round_communication_bytes,
    get_fedplora_shared_param_names,
    get_trainable_param_names,
    is_fdlora_agg,
    is_flora_agg,
    is_fedalt_agg,
    is_fedsa_lora_agg,
    is_fedplora_oneshot_agg,
    is_fedplora_shared_param_name,
    is_fedplora_multiround_agg,
    is_hetlora_agg,
    is_lora_a2_agg,
    is_lora_a_disk_agg,
    is_lora_a_param_name,
    is_yoco_agg,
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
parser.add_argument("--agg_type", type=str, default="fedplora", help="Aggregation type")
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
parser.add_argument(
    "--client_state_dir",
    type=str,
    default="artifacts/domain_client_states",
    help="Per-client state root (contains seed_*/). If left at default artifacts/domain_client_states, "
    "it is relocated to artifacts_{num_clients}c/domain_client_states after benchmark load.",
)
parser.add_argument("--save_client_state_to_disk", action="store_true", help="Persist per-client FedPLoRA local states to disk instead of keeping them in CPU memory")
parser.add_argument(
    "--metrics_output_dir",
    type=str,
    default="artifacts/sft_metrics",
    help="Round-wise metrics JSON directory. Default artifacts/sft_metrics is relocated to "
    "artifacts_{num_clients}c/sft_metrics when still the legacy path.",
)
parser.add_argument("--gp_align_lambda", type=float, default=0.01)
parser.add_argument("--gp_prox_lambda", type=float, default=0.001)
parser.add_argument("--gp_orth_lambda", type=float, default=1e-4)
parser.add_argument("--gp_consensus_power", type=float, default=2.0)
parser.add_argument("--gp_agg_momentum", type=float, default=0.5)
parser.add_argument(
    "--yoco_sparse_lambda",
    type=float,
    default=1e-4,
    help="YOCO: L1 penalty on trainable LoRA A (conventional sparse prior).",
)
parser.add_argument(
    "--yoco_pcwa_components",
    type=int,
    default=3,
    help="YOCO: number of principal directions for PCWA weights (<= n_clients-1).",
)
parser.add_argument(
    "--eval_max_batches",
    type=int,
    default=0,
    help="If > 0, cap batches per eval forward pass (per client×domain). 0 = full eval. Does not affect training.",
)
parser.add_argument(
    "--eval_personalization_metrics",
    action="store_true",
    help="Also report client-local test loss (in-domain) and off-domain test loss for personalization analysis.",
)
parser.add_argument(
    "--fedplora_ablation_no_consensus",
    action="store_true",
    help="FedPLoRA server only: disable sign-alignment and consensus-based row reweighting in aggregate_models_fedplora.",
)
parser.add_argument(
    "--fedplora_ablation_no_momentum",
    action="store_true",
    help="FedPLoRA server only: disable EMA-style blending with previous global A (gp_agg_momentum ignored for A).",
)

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


def compute_lm_loss(model, dataloader, device, max_batches=0):
    model.to(device)
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            if max_batches and steps >= max_batches:
                break
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


def _norm_path_key(path: str) -> str:
    return os.path.normpath(os.path.expanduser(path or ""))


def _relocate_legacy_artifact_dirs(args, num_clients: int) -> None:
    """
    If user kept CLI defaults artifacts/domain_client_states and artifacts/sft_metrics,
    move them under artifacts_{num_clients}c/ so parallel runs for 7c vs 35c do not clash.
    Subdirectory names domain_client_states / sft_metrics stay unchanged.
    """
    if num_clients <= 0:
        return
    root = f"artifacts_{int(num_clients)}c"
    legacy_cs = _norm_path_key("artifacts/domain_client_states")
    legacy_met = _norm_path_key("artifacts/sft_metrics")
    cur_cs = _norm_path_key(args.client_state_dir)
    cur_met = _norm_path_key(args.metrics_output_dir)
    if cur_cs == legacy_cs:
        args.client_state_dir = os.path.join(root, "domain_client_states")
    if cur_met == legacy_met:
        args.metrics_output_dir = os.path.join(root, "sft_metrics")


def _fedplora_disk_state_dir(args):
    """Stable absolute path so client .pt files are always found across rounds."""
    base = os.path.abspath(os.path.expanduser(args.client_state_dir))
    return os.path.join(base, f"seed_{args.seed}")


def _client_state_path(base_dir, client_id):
    return os.path.join(base_dir, f"client_{int(client_id):03d}.pt")


def _save_client_local_state(local_state, base_dir, client_id):
    os.makedirs(base_dir, exist_ok=True)
    path = _client_state_path(base_dir, client_id)
    # Atomic replace avoids truncated / missing files if the job is interrupted mid-write.
    tmp_path = path + ".tmp"
    torch.save(local_state, tmp_path)
    os.replace(tmp_path, path)
    return path


def _load_client_local_state(base_dir, client_id):
    path = _client_state_path(base_dir, client_id)
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu")


def _disk_assert_all_client_states(client_store, client_ids, context: str):
    if client_store["mode"] != "disk":
        return
    missing = []
    for cid in client_ids:
        p = _client_state_path(client_store["state_dir"], cid)
        if not os.path.isfile(p):
            missing.append((int(cid), p))
    if missing:
        detail = "; ".join(f"client_id={c} -> {p}" for c, p in missing)
        raise FileNotFoundError(f"{context}: missing disk client state ({detail})")


def _ensure_sequential_fedplora_local_states(model, client_ids, args):
    state_dir = _fedplora_disk_state_dir(args)
    if getattr(args, "save_client_state_to_disk", False):
        os.makedirs(state_dir, exist_ok=True)
        seed_sd = extract_fedplora_local_state(model)
        for client_id in client_ids:
            path = _client_state_path(state_dir, client_id)
            if not os.path.isfile(path):
                _save_client_local_state(seed_sd, state_dir, client_id)
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
    eval_cap = int(getattr(args, "eval_max_batches", 0) or 0)
    shared_state = {
        k: v.detach().cpu().clone()
        for k, v in global_model.state_dict().items()
        if is_fedplora_shared_param_name(k, get_trainable_param_names(global_model))
    }
    metrics = {}
    domains_sorted = sorted(by_domain.items())
    total_passes = len(domains_sorted) * len(client_ids)
    print(
        f"[eval] domain-macro sequential: {len(domains_sorted)} domains × {len(client_ids)} clients "
        f"({total_passes} forward passes); eval_max_batches={eval_cap or 'all'}",
        flush=True,
    )
    for domain, rows in domains_sorted:
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        losses = []
        for client_id in tqdm(
            client_ids,
            desc=f"eval {domain}",
            leave=False,
            dynamic_ncols=True,
        ):
            broadcast_fedplora_shared_state(global_model, shared_state)
            local_state = _get_client_local_state(client_store, client_id)
            load_fedplora_local_state(global_model, local_state)
            losses.append(compute_lm_loss(global_model, dl, device, max_batches=eval_cap))
        metrics[domain] = float(np.mean(losses))
    macro = float(np.mean(list(metrics.values()))) if metrics else float("nan")
    worst = float(max(metrics.values())) if metrics else float("nan")
    return metrics, macro, worst


def _client_id_to_home_domain(clients_manifest):
    return {int(c["client_id"]): str(c["domain"]) for c in clients_manifest}


def _evaluate_personalization_metrics(
    global_model,
    client_ids,
    client_store,
    benchmark,
    tokenizer,
    args,
):
    """
    In-domain: mean LM loss on each client's test_local, then macro over clients.
    Off-domain: for each client, mean loss on held-out test_domain splits of domains != home domain.
    In-domain (domain test): each client on test_domain rows of its home domain only (sanity vs test_local).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cap = int(getattr(args, "eval_max_batches", 0) or 0)
    shared_state = {
        k: v.detach().cpu().clone()
        for k, v in global_model.state_dict().items()
        if is_fedplora_shared_param_name(k, get_trainable_param_names(global_model))
    }
    id2home = _client_id_to_home_domain(benchmark["clients"])
    by_client_local = group_rows_by_client(benchmark["test_local"])
    by_domain_test = group_rows_by_domain(benchmark["test_domain"])

    local_losses = []
    for client_id in client_ids:
        rows = by_client_local.get(int(client_id), [])
        if not rows:
            continue
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        broadcast_fedplora_shared_state(global_model, shared_state)
        local_state = _get_client_local_state(client_store, client_id)
        load_fedplora_local_state(global_model, local_state)
        local_losses.append(
            compute_lm_loss(global_model, dl, device, max_batches=eval_cap)
        )
    client_local_macro = (
        float(np.mean(local_losses)) if local_losses else float("nan")
    )

    off_losses = []
    for client_id in client_ids:
        home = id2home.get(int(client_id))
        if not home:
            continue
        for domain, rows in sorted(by_domain_test.items()):
            if domain == home or not rows:
                continue
            dl = create_domain_eval_dataloader(rows, tokenizer, args)
            broadcast_fedplora_shared_state(global_model, shared_state)
            local_state = _get_client_local_state(client_store, client_id)
            load_fedplora_local_state(global_model, local_state)
            off_losses.append(
                compute_lm_loss(global_model, dl, device, max_batches=eval_cap)
            )
    off_domain_macro = float(np.mean(off_losses)) if off_losses else float("nan")

    in_dom_dt_losses = []
    for client_id in client_ids:
        home = id2home.get(int(client_id))
        if not home:
            continue
        rows = by_domain_test.get(home, [])
        if not rows:
            continue
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        broadcast_fedplora_shared_state(global_model, shared_state)
        local_state = _get_client_local_state(client_store, client_id)
        load_fedplora_local_state(global_model, local_state)
        in_dom_dt_losses.append(
            compute_lm_loss(global_model, dl, device, max_batches=eval_cap)
        )
    in_domain_domain_test_macro = (
        float(np.mean(in_dom_dt_losses)) if in_dom_dt_losses else float("nan")
    )

    return {
        "client_local_macro_loss": client_local_macro,
        "off_domain_macro_loss": off_domain_macro,
        "in_domain_domain_test_macro_loss": in_domain_domain_test_macro,
    }


def _evaluate_personalization_metrics_full_state(
    global_model,
    eval_store,
    eval_client_ids,
    benchmark,
    tokenizer,
    args,
):
    """Personalization metrics for full-state clients (normal / ffa / fedex / flora, etc.)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cap = int(getattr(args, "eval_max_batches", 0) or 0)
    id2home = _client_id_to_home_domain(benchmark["clients"])
    by_client_local = group_rows_by_client(benchmark["test_local"])
    by_domain_test = group_rows_by_domain(benchmark["test_domain"])

    local_losses = []
    for idx in eval_client_ids:
        rows = by_client_local.get(int(idx), [])
        if not rows:
            continue
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        state = _get_client_local_state(eval_store, idx)
        load_partial_state_dict(global_model, state)
        local_losses.append(
            compute_lm_loss(global_model, dl, device, max_batches=eval_cap)
        )
    client_local_macro = (
        float(np.mean(local_losses)) if local_losses else float("nan")
    )

    off_losses = []
    for idx in eval_client_ids:
        home = id2home.get(int(idx))
        if not home:
            continue
        for domain, rows in sorted(by_domain_test.items()):
            if domain == home or not rows:
                continue
            dl = create_domain_eval_dataloader(rows, tokenizer, args)
            state = _get_client_local_state(eval_store, idx)
            load_partial_state_dict(global_model, state)
            off_losses.append(
                compute_lm_loss(global_model, dl, device, max_batches=eval_cap)
            )
    off_domain_macro = float(np.mean(off_losses)) if off_losses else float("nan")

    in_dom_dt_losses = []
    for idx in eval_client_ids:
        home = id2home.get(int(idx))
        if not home:
            continue
        rows = by_domain_test.get(home, [])
        if not rows:
            continue
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        state = _get_client_local_state(eval_store, idx)
        load_partial_state_dict(global_model, state)
        in_dom_dt_losses.append(
            compute_lm_loss(global_model, dl, device, max_batches=eval_cap)
        )
    in_domain_domain_test_macro = (
        float(np.mean(in_dom_dt_losses)) if in_dom_dt_losses else float("nan")
    )

    return {
        "client_local_macro_loss": client_local_macro,
        "off_domain_macro_loss": off_domain_macro,
        "in_domain_domain_test_macro_loss": in_domain_domain_test_macro,
    }


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
    if (is_yoco_agg(args.agg_type) or is_fedplora_oneshot_agg(args.agg_type)) and args.rounds != 1:
        tag = "YOCO" if is_yoco_agg(args.agg_type) else "fedplora-oneshot (FedP-LoRA + YOCO)"
        print(f"[setup] {tag} is one-shot: forcing --rounds 1")
        args.rounds = 1

    benchmark, split_dir = build_or_load_benchmark(args)
    print(f"[benchmark] loaded from {split_dir}")
    print(f"[benchmark] domains={sorted(benchmark['domain_stats'].keys())}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    client_ids, client_dataloaders = create_domain_client_dataloaders(
        benchmark["train"], tokenizer, args
    )
    client_sizes = [len(dl.dataset) for dl in client_dataloaders]
    args.num_clients = len(client_ids)
    args._runtime_client_sizes = client_sizes
    _relocate_legacy_artifact_dirs(args, args.num_clients)
    print(
        f"[setup] client_state_dir={args.client_state_dir} "
        f"metrics_output_dir={args.metrics_output_dir}",
        flush=True,
    )

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

    if is_lora_a_disk_agg(args.agg_type):
        init_fedplora_adapters(global_model)
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
        "communication": {
            "agg_type": args.agg_type,
            "down_bytes_per_client": int(comm_info["down_bytes_per_client"]),
            "up_bytes_per_client": int(comm_info["up_bytes_per_client"]),
        },
        "rounds": [],
    }

    for round_idx in range(args.rounds):
        print(f"Round {round_idx + 1}/{args.rounds}")

        if (
            is_lora_a_disk_agg(args.agg_type)
            and getattr(args, "save_client_state_to_disk", False)
            and round_idx > 0
        ):
            _disk_assert_all_client_states(
                client_store,
                client_ids,
                context=f"start of round {round_idx + 1}",
            )

        if is_lora_a_disk_agg(args.agg_type):
            args._fedplora_client_sizes = client_sizes
            shared_names = get_fedplora_shared_param_names(global_model)
            gp_global_state = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
                if k in shared_names
            }
            args._fedplora_global_A = {
                k: v.detach().cpu().clone()
                for k, v in gp_global_state.items()
                if is_lora_a_param_name(k)
            }
        else:
            round_global_state = {
                k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()
            }

        client_states_for_agg = []
        fedplora_uploads = []
        args._lora_a2_train_round = round_idx
        for i, client_id in enumerate(client_ids):
            args._tqdm_desc = f"R{round_idx + 1}/{args.rounds} client{i + 1}/{args.num_clients}"
            if is_lora_a_disk_agg(args.agg_type):
                broadcast_fedplora_shared_state(global_model, gp_global_state)
                local_state = _get_client_local_state(client_store, client_id)
                load_fedplora_local_state(global_model, local_state)
            else:
                load_partial_state_dict(global_model, round_global_state)
            train_client(global_model, client_dataloaders[i], args, client_idx=i)
            if is_lora_a_disk_agg(args.agg_type):
                fedplora_uploads.append(
                    build_fedplora_upload_package(global_model, client_sizes[i])
                )
            else:
                client_states_for_agg.append(
                    {
                        k: v.detach().cpu().clone()
                        for k, v in global_model.state_dict().items()
                    }
                )
            if is_lora_a_disk_agg(args.agg_type):
                updated_local_state = extract_fedplora_local_state(global_model)
                _set_client_local_state(client_store, client_id, updated_local_state)

        args._tqdm_desc = None

        if is_lora_a_disk_agg(args.agg_type) and getattr(args, "save_client_state_to_disk", False):
            _disk_assert_all_client_states(
                client_store,
                client_ids,
                context=f"after round {round_idx + 1} local training (before aggregation)",
            )

        print(
            f"[round {round_idx + 1}] local training done; aggregating agg_type={args.agg_type} ...",
            flush=True,
        )
        if args.agg_type == "normal":
            global_model = aggregate_models_normal(global_model, client_states_for_agg)
        elif args.agg_type == "fedex":
            global_model = aggregate_models_fedex(
                global_model, client_states_for_agg, args
            )
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_states_for_agg)
        elif is_fedplora_multiround_agg(args.agg_type):
            global_model = aggregate_models_fedplora(global_model, fedplora_uploads, args)
        elif is_yoco_agg(args.agg_type) or is_fedplora_oneshot_agg(args.agg_type):
            global_model = aggregate_models_yoco(global_model, fedplora_uploads, args)
        elif is_fedsa_lora_agg(args.agg_type) or is_fedalt_agg(args.agg_type):
            global_model = aggregate_models_fedsa_lora(global_model, fedplora_uploads, args)
        elif is_hetlora_agg(args.agg_type):
            global_model = aggregate_models_hetlora(global_model, client_states_for_agg, args)
        elif is_flora_agg(args.agg_type):
            global_model = aggregate_models_flora(global_model, client_states_for_agg, args)
        elif is_lora_a2_agg(args.agg_type):
            args._lora_a2_agg_round = round_idx
            global_model = aggregate_models_lora_a2(global_model, client_states_for_agg, args)
        elif is_fdlora_agg(args.agg_type):
            global_model = aggregate_models_fdlora(global_model, client_states_for_agg, args)
        else:
            raise ValueError(f"Unknown agg_type: {args.agg_type}")

        print(f"[round {round_idx + 1}] aggregation done; running evaluation ...", flush=True)
        pfl_block = {}
        if is_lora_a_disk_agg(args.agg_type):
            domain_metrics, domain_macro, worst_domain = _evaluate_domain_macro_sequential(
                global_model,
                client_ids,
                client_store,
                benchmark["test_domain"],
                tokenizer,
                args,
            )
            if getattr(args, "eval_personalization_metrics", False):
                pfl_block = _evaluate_personalization_metrics(
                    global_model,
                    client_ids,
                    client_store,
                    benchmark,
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
            eval_cap = int(getattr(args, "eval_max_batches", 0) or 0)
            domains_sorted = sorted(by_domain.items())
            print(
                f"[eval] full-state clients: {len(domains_sorted)} domains × {len(eval_client_ids)} clients; "
                f"eval_max_batches={eval_cap or 'all'}",
                flush=True,
            )
            metrics = {}
            for domain, rows in domains_sorted:
                dl = create_domain_eval_dataloader(rows, tokenizer, args)
                losses = []
                for idx in tqdm(
                    eval_client_ids,
                    desc=f"eval {domain}",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    state = _get_client_local_state(eval_store, idx)
                    load_partial_state_dict(global_model, state)
                    losses.append(compute_lm_loss(global_model, dl, device, max_batches=eval_cap))
                metrics[domain] = float(np.mean(losses))
            domain_metrics = metrics
            domain_macro = float(np.mean(list(metrics.values()))) if metrics else float("nan")
            worst_domain = float(max(metrics.values())) if metrics else float("nan")
            if getattr(args, "eval_personalization_metrics", False):
                pfl_block = _evaluate_personalization_metrics_full_state(
                    global_model,
                    eval_store,
                    eval_client_ids,
                    benchmark,
                    tokenizer,
                    args,
                )
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
        if pfl_block:
            print(
                f"[eval] personalization round={round_idx + 1} "
                f"client_local_macro_loss={pfl_block['client_local_macro_loss']:.4f} "
                f"in_domain_domain_test_macro_loss={pfl_block['in_domain_domain_test_macro_loss']:.4f} "
                f"off_domain_macro_loss={pfl_block['off_domain_macro_loss']:.4f}",
                flush=True,
            )

        round_payload = {
            "round": round_idx + 1,
            "domain_macro_loss": domain_macro,
            "best_domain_macro_loss": best_domain_macro,
            "worst_domain_loss": worst_domain,
            "best_worst_domain_loss": best_worst_domain,
            "domain_metrics": domain_metrics,
        }
        round_payload.update(pfl_block)
        metrics_history["rounds"].append(round_payload)

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
