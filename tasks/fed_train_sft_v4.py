"""FedPLoRA-v4 SFT training entry point.

This script wraps v2's `fed_train_sft.py` by:

  1. Adding `--agg_type v4_*` dispatch for Branch A / C / D aggregators.
  2. Adding new CLI args for v4-specific knobs (gate kappa, cluster mode, etc).
  3. Hooking client-side regularizers from Branch C (B-sign + sparse A) into
     `train_client` via the existing v2 hook surface.
  4. Persisting `A_local` snapshots for Branch D.
  5. Writing v4 conflict / gate / cluster diagnostics into the round JSON.

Usage example:
    python tasks/fed_train_sft_v4.py \
        --model /path/to/Meta-Llama-3.1-8B \
        --benchmark_dir data/domain_benchmark_35c/seed_42 \
        --agg_type v4_hier_soft_prior --rounds 1 --local_epochs 1

The script imports v2 utilities (data loading, model factory, train_client,
eval routines) directly to avoid duplication. v4 lives next to v2 and adds
new aggregation choices without modifying any v2 file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

# Re-use v2 plumbing
from utilities.data_utils import (
    create_domain_client_dataloaders,
    create_domain_eval_dataloader,
    group_rows_by_domain,
    load_domain_sft_benchmark,
)
from utilities.models import create_peft_causal_lm_model, init_fedplora_adapters
from utilities.state_dict_ops import (
    broadcast_fedplora_shared_state,
    extract_fedplora_local_state,
    load_fedplora_local_state,
    load_partial_state_dict,
)
from utilities.utils import (
    estimate_round_communication_bytes,
    get_fedplora_shared_param_names,
    get_trainable_param_names,
    is_fedplora_shared_param_name,
    is_lora_a_param_name,
    restore_logging,
    setup_run_logging,
)
from utilities import train_eval as v2_train_eval
from methods.fedp_lora import build_fedplora_upload_package
from methods.fedplora_oneshot import aggregate_models_fedplora_oneshot
from methods.fedsa_lora import aggregate_models_fedsa_lora

# v4 aggregators
from methods.v4.fedplora_v4_hier import (
    aggregate_models_v4_hier,
    aggregate_models_v4_hier_soft_prior,
    aggregate_models_v4_hier_soft_spectral,
    aggregate_models_v4_hier_soft_pfl_eval,
)
from methods.v4.fedplora_v4_sign import (
    apply_sign_regularizers,
    maybe_init_bsign_anchor,
    update_bsign_anchor,
)
from methods.v4.fedplora_v4_mix import (
    build_mixed_A,
    search_per_domain_eta,
    snapshot_local_A,
)


# ---------------------------------------------------------------------------
# Patch v2's train_client to add v4 client-side regularizers
# ---------------------------------------------------------------------------

_orig_train_client = v2_train_eval.train_client


def _train_client_v4_sign(model, dataloader, args, client_idx=0):
    """Branch C only: v2 train loop + B-sign anchor updates (no lora_a2; removed from main repo)."""
    maybe_init_bsign_anchor(model, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    v2_train_eval._fedplora_refresh_reg_tensor_gpu_cache(model, args)

    full_steps = len(dataloader) * args.local_epochs
    cap_steps = int(getattr(args, "train_max_steps_per_client", 0) or 0)
    steps_this_round = min(full_steps, cap_steps) if cap_steps > 0 else full_steps

    trainable = [p for p in model.parameters() if p.requires_grad] or list(model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = v2_train_eval.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(steps_this_round * args.warmup_ratio),
        num_training_steps=steps_this_round,
    )
    scaler = v2_train_eval.GradScaler()
    model.train()
    global_step = 0
    try:
        for _ in range(args.local_epochs):
            for batch in v2_train_eval.tqdm(
                dataloader,
                leave=True,
                dynamic_ncols=True,
                desc=getattr(args, "_tqdm_desc", None),
            ):
                if global_step >= steps_this_round:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                with v2_train_eval.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss = v2_train_eval._add_fedplora_regularization(loss, model, args)
                    loss = v2_train_eval._add_yoco_sparse(loss, model, args)
                    loss = v2_train_eval._add_fedplora_oneshot_anchor(loss, model, args)
                    loss = apply_sign_regularizers(loss, model, args)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                update_bsign_anchor(model, args)
                global_step += 1
            if global_step >= steps_this_round:
                break
        return model.state_dict()
    finally:
        args._fedplora_initial_A_gpu = None
        args._fedplora_global_A_gpu = None


def _patched_train_client(model, dataloader, args, client_idx=0):
    """v4: reuse v2 train_client; sign/mix add Branch C/D hooks only."""
    agg_type = (getattr(args, "agg_type", "") or "").lower()
    if agg_type in {"v4_sign_v2agg", "v4_sign_full"}:
        return _train_client_v4_sign(model, dataloader, args, client_idx=client_idx)

    sd = _orig_train_client(model, dataloader, args, client_idx=client_idx)

    if agg_type.startswith("v4_mix_"):
        a_local = snapshot_local_A(model)
        save_dir = Path(getattr(args, "v4_mix_save_dir", "artifacts/v4_mix_a_local"))
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(a_local, save_dir / f"client_{client_idx:03d}_A_local.pt")
    return sd


v2_train_eval.train_client = _patched_train_client


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description="Federated SFT with FedPLoRA-v4")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--benchmark_dir", type=str, required=True)
    p.add_argument("--agg_type", type=str, required=True,
                   help="One of: fedplora_oneshot | fedalt | "
                        "v4_hier_soft_prior | v4_hier_soft_spectral | v4_hier_soft_pfl_eval | "
                        "v4_sign_v2agg | v4_sign_full | "
                        "v4_mix_fixed05 | v4_mix_per_domain | v4_mix_moe")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--num_clients", type=int, default=0)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--rslora", action="store_true")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                   choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    p.add_argument("--client_state_dir", type=str, default="artifacts/v4_client_states")
    p.add_argument("--save_client_state_to_disk", action="store_true")
    p.add_argument("--metrics_output_dir", type=str, default="artifacts/v4_sft_metrics")
    p.add_argument("--eval_max_batches", type=int, default=200)
    p.add_argument("--eval_seeds", type=str, default="42",
                   help="Comma-separated list of seeds to repeat the experiment.")

    # v2 regularizers (reused by v4)
    p.add_argument("--oneshot_anchor_lambda", type=float, default=1e-4)
    p.add_argument("--oneshot_prox_lambda",   type=float, default=0.0)
    p.add_argument("--oneshot_consensus_power", type=float, default=2.0)
    p.add_argument("--oneshot_importance_power", type=float, default=1.0)
    p.add_argument("--oneshot_importance_clip", type=float, default=5.0)
    p.add_argument("--oneshot_conflict_threshold", type=float, default=0.35)
    p.add_argument("--oneshot_conflict_blend", type=float, default=1.0)
    p.add_argument("--oneshot_scale_clip_ratio", type=float, default=0.0)
    p.add_argument("--oneshot_no_keep_init_on_conflict", action="store_true")
    p.add_argument("--oneshot_orthogonalize", action="store_true")
    p.add_argument("--yoco_sparse_lambda", type=float, default=0.0)

    # Branch A: Hier++
    p.add_argument("--v4_gate_kappa",        type=float, default=1.0)
    p.add_argument("--v4_gate_power",        type=float, default=1.0)
    p.add_argument("--v4_cluster_mode",      type=str,   default="prior",
                   choices=["prior", "spectral", "kmeans", "none"])
    p.add_argument("--v4_cluster_k",         type=int,   default=3)
    p.add_argument("--v4_lambda_min",        type=float, default=0.3)
    p.add_argument("--v4_lambda_max",        type=float, default=0.9)
    p.add_argument("--v4_personalized_eval", type=int,   default=1)
    p.add_argument("--v4_default_uniform",   type=int,   default=1)
    p.add_argument("--v4_residual_eps",      type=float, default=1e-7)

    # Branch C: Sign
    p.add_argument("--v4_bsign_lambda",       type=float, default=0.0)
    p.add_argument("--v4_bsign_gamma",        type=float, default=5.0)
    p.add_argument("--v4_bsign_anchor_steps", type=int,   default=1)
    p.add_argument("--v4_asparse_lambda",     type=float, default=0.0)

    # Branch D: Mix
    p.add_argument("--v4_mix_mode",        type=str,   default="fixed",
                   choices=["fixed", "per_domain", "moe"])
    p.add_argument("--v4_mix_eta",         type=float, default=0.5)
    p.add_argument("--v4_mix_save_dir",    type=str,   default="artifacts/v4_mix_a_local")
    p.add_argument("--v4_mix_search_grid", type=str,   default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--v4_mix_gate_hidden", type=int,   default=64)
    p.add_argument("--v4_mix_gate_epochs", type=int,   default=3)
    return p


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


def _dispatch_aggregate(agg_type, global_model, fedplora_uploads, args):
    """Map agg_type string to the actual aggregator. Returns updated global_model."""
    t = agg_type.lower()
    if t in {"fedplora_oneshot", "fedplora-oneshot"}:
        return aggregate_models_fedplora_oneshot(global_model, fedplora_uploads, args)
    if t in {"fedalt", "fedsa", "fedsa_lora"}:
        return aggregate_models_fedsa_lora(global_model, fedplora_uploads, args)

    # Branch A
    if t == "v4_hier_soft_prior":
        return aggregate_models_v4_hier_soft_prior(global_model, fedplora_uploads, args)
    if t == "v4_hier_soft_spectral":
        return aggregate_models_v4_hier_soft_spectral(global_model, fedplora_uploads, args)
    if t == "v4_hier_soft_pfl_eval":
        return aggregate_models_v4_hier_soft_pfl_eval(global_model, fedplora_uploads, args)

    # Branch C — server aggregation is identical to v2 oneshot, local regs do the work
    if t in {"v4_sign_v2agg", "v4_sign_full"}:
        return aggregate_models_fedplora_oneshot(global_model, fedplora_uploads, args)

    # Branch D — server aggregation reuses v2 oneshot; mixer applied during eval
    if t in {"v4_mix_fixed05", "v4_mix_per_domain", "v4_mix_moe"}:
        return aggregate_models_fedplora_oneshot(global_model, fedplora_uploads, args)

    raise ValueError(f"Unknown v4 agg_type: {agg_type}")


# ---------------------------------------------------------------------------
# Federated training loop (single seed)
# ---------------------------------------------------------------------------


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_lm_eval_stats(model, dataloader, device, max_batches=0):
    model.to(device)
    model.eval()
    total_loss, steps, total_correct, total_valid = 0.0, 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            if max_batches and steps >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            total_loss += float(outputs.loss.detach().cpu().item())
            preds = outputs.logits[..., :-1, :].argmax(dim=-1)
            shifted = labels[..., 1:].contiguous()
            mask = shifted.ne(-100)
            if mask.any():
                total_correct += int((preds[mask] == shifted[mask]).sum().cpu())
                total_valid += int(mask.sum().cpu())
            steps += 1
    mean_loss = total_loss / max(steps, 1)
    tok_acc = total_correct / max(total_valid, 1)
    try:
        ppl = float(math.exp(min(mean_loss, 80.0)))
    except OverflowError:
        ppl = float("inf")
    return {"loss": mean_loss, "token_accuracy": float(tok_acc), "perplexity": ppl,
            "n_eval_batches": int(steps)}


def federated_sft_single_seed(args):
    """Run one federation pass for one seed. Returns metrics dict."""
    benchmark = load_domain_sft_benchmark(args.benchmark_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    client_ids, client_dataloaders = create_domain_client_dataloaders(
        benchmark["train"], tokenizer, args
    )
    client_sizes = [len(dl.dataset) for dl in client_dataloaders]
    args.num_clients = len(client_ids)
    args._runtime_client_sizes = client_sizes
    args._fedplora_client_domains = {
        int(c["client_id"]): str(c.get("domain", "unknown"))
        for c in benchmark.get("clients", [])
    }

    global_model = create_peft_causal_lm_model(args)
    comm = estimate_round_communication_bytes(
        global_model.state_dict(), args.agg_type,
        trainable_param_names=get_trainable_param_names(global_model),
    )
    print(f"[v4] seed={args.seed} agg_type={args.agg_type} "
          f"comm_down={comm['down_bytes_per_client']}B comm_up={comm['up_bytes_per_client']}B")

    init_fedplora_adapters(global_model)
    initial_A = {
        k: v.detach().cpu().clone()
        for k, v in global_model.state_dict().items()
        if is_lora_a_param_name(k)
    }
    args._fedplora_initial_A = initial_A

    # Per-client local B store (in-memory for brevity; v2 supports disk persistence)
    local_states = {
        int(cid): {k: v.clone() for k, v in extract_fedplora_local_state(global_model).items()}
        for cid in client_ids
    }

    metrics = {"args": vars(args).copy(), "seed": int(args.seed),
               "benchmark_dir": args.benchmark_dir, "rounds": [],
               "communication": dict(comm, agg_type=args.agg_type)}

    for round_idx in range(args.rounds):
        args._fedplora_client_sizes = client_sizes
        args._fedplora_round_client_ids = [int(x) for x in client_ids]
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

        fedplora_uploads = []
        for i, client_id in enumerate(client_ids):
            args._tqdm_desc = f"R{round_idx + 1} c{i + 1}/{args.num_clients}"
            broadcast_fedplora_shared_state(global_model, gp_global_state)
            load_fedplora_local_state(global_model, local_states[int(client_id)])
            v2_train_eval.train_client(global_model, client_dataloaders[i], args, client_idx=i)
            local_states[int(client_id)] = extract_fedplora_local_state(global_model)
            fedplora_uploads.append(build_fedplora_upload_package(
                global_model, client_sizes[i],
                client_id=int(client_id),
                domain=args._fedplora_client_domains.get(int(client_id), "unknown"),
            ))
        args._tqdm_desc = None

        print(f"[v4] round {round_idx + 1}: aggregating ({args.agg_type})")
        global_model = _dispatch_aggregate(args.agg_type, global_model, fedplora_uploads, args)

        # Per-round eval (domain macro on test_domain)
        domain_metrics, macro_acc, worst_acc, macro_ppl, worst_ppl = _evaluate(
            global_model, client_ids, local_states, benchmark, tokenizer, args
        )
        round_block = {
            "round": round_idx + 1,
            "domain_macro_token_accuracy": macro_acc,
            "worst_domain_token_accuracy": worst_acc,
            "domain_macro_perplexity": macro_ppl,
            "worst_domain_perplexity": worst_ppl,
            "domain_metrics": domain_metrics,
        }
        # Dump v4 diagnostics if present
        v4_stats = getattr(args, "_fedplora_v4_stats", None)
        if v4_stats:
            round_block["v4_stats_summary"] = v4_stats.get("_summary", {})
        metrics["rounds"].append(round_block)

    return metrics


def _evaluate(global_model, client_ids, local_states, benchmark, tokenizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_domain = group_rows_by_domain(benchmark["test_domain"])
    eval_cap = int(getattr(args, "eval_max_batches", 0) or 0)
    personalized = getattr(args, "_fedplora_personalized_shared_states", {}) or {}
    shared_state = {
        k: v.detach().cpu().clone()
        for k, v in global_model.state_dict().items()
        if is_fedplora_shared_param_name(k, get_trainable_param_names(global_model))
    }

    # Branch D mixer setup
    agg_type = (getattr(args, "agg_type", "") or "").lower()
    mix_active = agg_type.startswith("v4_mix_")
    mix_local_dir = Path(getattr(args, "v4_mix_save_dir", "artifacts/v4_mix_a_local"))
    mix_eta_by_domain = {}
    if mix_active:
        if args.v4_mix_mode == "per_domain":
            print("[v4-mix] per-domain eta search not yet wired to eval; using fixed eta")
        # `fixed` and `moe` paths fall back to a single eta for now
        for d in by_domain.keys():
            mix_eta_by_domain[d] = float(args.v4_mix_eta)

    metrics = {}
    for domain, rows in sorted(by_domain.items()):
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        stats_per_client = []
        for client_id in client_ids:
            client_shared = dict(shared_state)
            client_shared.update(personalized.get(int(client_id), {}))
            broadcast_fedplora_shared_state(global_model, client_shared)
            load_fedplora_local_state(global_model, local_states[int(client_id)])

            if mix_active:
                # Load A_local for this client and mix it with current A (= A_down).
                a_local_path = mix_local_dir / f"client_{client_id:03d}_A_local.pt"
                if a_local_path.exists():
                    a_local = torch.load(a_local_path, map_location="cpu")
                    A_down = {
                        k: v.detach().cpu().clone()
                        for k, v in global_model.state_dict().items()
                        if is_lora_a_param_name(k)
                    }
                    eta = mix_eta_by_domain.get(domain, float(args.v4_mix_eta))
                    A_eff = build_mixed_A(A_down, a_local, eta)
                    # Write A_eff back into the model
                    sd = global_model.state_dict()
                    for k, v in A_eff.items():
                        if k in sd:
                            sd[k] = v.to(device=sd[k].device, dtype=sd[k].dtype)
                    global_model.load_state_dict(sd)

            stats_per_client.append(
                compute_lm_eval_stats(global_model, dl, device, max_batches=eval_cap)
            )
        metrics[domain] = {
            "loss": float(np.mean([s["loss"] for s in stats_per_client])),
            "token_accuracy": float(np.mean([s["token_accuracy"] for s in stats_per_client])),
            "perplexity": float(np.mean([s["perplexity"] for s in stats_per_client])),
        }

    accs = [v["token_accuracy"] for v in metrics.values()]
    losses = [v["loss"] for v in metrics.values()]
    macro_acc = float(np.mean(accs)) if accs else float("nan")
    worst_acc = float(min(accs)) if accs else float("nan")
    macro_ppl = float(np.mean([v["perplexity"] for v in metrics.values()]))
    worst_ppl = float(max(v["perplexity"] for v in metrics.values()))
    return metrics, macro_acc, worst_acc, macro_ppl, worst_ppl


def main():
    args = build_parser().parse_args()
    log_file, orig_out, orig_err, log_path = setup_run_logging(
        args, log_dir="log_v4", filename_prefix="v4_sft"
    )
    try:
        seeds = [int(s) for s in str(args.eval_seeds).split(",") if s.strip()]
        per_seed_results = []
        for seed in seeds:
            args.seed = seed
            set_seed(seed)
            print(f"\n========== v4 run seed={seed} ==========")
            result = federated_sft_single_seed(args)
            per_seed_results.append(result)

        # Aggregate across seeds
        os.makedirs(args.metrics_output_dir, exist_ok=True)
        out_name = (f"{args.agg_type}_"
                    f"{Path(args.model).name}_seed_{','.join(str(s) for s in seeds)}_"
                    f"r{args.rounds}_e{args.local_epochs}.json")
        with open(os.path.join(args.metrics_output_dir, out_name), "w") as f:
            json.dump({"per_seed": per_seed_results,
                       "agg_type": args.agg_type,
                       "seeds": seeds}, f, indent=2)
        print(f"[v4] wrote metrics to {os.path.join(args.metrics_output_dir, out_name)}")
    finally:
        restore_logging(log_file, orig_out, orig_err)


if __name__ == "__main__":
    main()
