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
    tensor_to_list,
)
from utilities import train_eval as v2_train_eval
from methods.fedp_lora import build_fedplora_upload_package
from methods.fedplora_oneshot import (
    aggregate_models_fedplora_oneshot,
    aggregate_models_fedplora_v3_rpca,
)
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
from methods.v4.fedplora_v4_mix import build_mixed_A, snapshot_local_A
from methods.v4.fedplora_v4_svd import aggregate_models_v4_svd
from methods.v4.fedplora_v4_anchor import aggregate_models_v4_anchor
from methods.v4.fedplora_v4_adarank import aggregate_models_v4_adarank
from utilities.v4_orth_init import orthogonalize_lora_A_in_model
from utilities.v4_run_checkpoint import (
    init_v4_client_store,
    maybe_apply_default_save_run_checkpoint_dir,
    persist_client_local_state,
    save_v4_run_checkpoint,
    try_resume_eval_only_from_post_agg,
    try_skip_if_run_fully_complete,
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
    if agg_type.startswith("v5_") and "_route_" in agg_type:
        a_local = snapshot_local_A(model)
        save_dir = Path(getattr(args, "v4_mix_save_dir", "artifacts/v5_route_a_local"))
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
                        "v4_mix_fixed05 | v4_mix_per_domain | v4_mix_moe | "
                        "v5_route_mix_align | v5_rpca_route_mix_align")
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
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default on (match §11.1). 8B @ max_seq_length=2048 needs this on a single A100.",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                   choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    p.add_argument("--client_state_dir", type=str, default="artifacts/v4_client_states")
    p.add_argument("--save_client_state_to_disk", action="store_true")
    p.add_argument("--metrics_output_dir", type=str, default="artifacts/v4_sft_metrics")
    p.add_argument("--eval_max_batches", type=int, default=50,
                   help="Max eval batches per client per domain (200 for final stats; 50 for pilot).")
    p.add_argument("--eval_seeds", type=str, default="42",
                   help="Comma-separated training seeds; EACH seed re-runs full 35-client "
                   "local train + agg + eval (expensive). Use '42' for pilot; "
                   "'42,1234,9999' only for final paper stats.")

    # 防白训：与 v2 fed_train_sft.py 同机制（默认 auto ../trained_models/<stem>/）
    p.add_argument("--save_run_checkpoint_dir", type=str, default="",
                   help="Run bundle root; empty = auto under ../trained_models/<stem>/")
    p.add_argument("--trained_models_root", type=str, default="",
                   help="Override parent dir for auto checkpoint bundles")
    p.add_argument("--no_auto_save_run_checkpoint", action="store_true",
                   help="Disable automatic save_run_checkpoint_dir")
    p.add_argument("--force_retrain", action="store_true",
                   help="Ignore existing checkpoints and retrain from scratch")
    p.add_argument("--skip_post_agg_snapshots", action="store_true",
                   help="Do not write snapshots/round_XXX_post_agg/ (disables eval-only resume)")
    p.add_argument("--eval_only_from_checkpoint", type=str, default="",
                   help="Eval-only from a bundle dir (final or snapshots/round_XXX_post_agg)")
    p.add_argument("--log_dir", type=str, default="log_v4",
                   help="Directory for tee logs (LW runs: log_LWv4)")
    p.add_argument("--log_filename_prefix", type=str, default="v4_sft",
                   help="Log filename prefix (LW runs: LWv4_sft)")

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
    p.add_argument("--v4_mix_val_scope", type=str, default="local",
                   choices=["local", "domain", "global"],
                   help="Validation scope for v4_mix_per_domain eta search. "
                   "local=client private val; domain=small domain anchor; global=all val rows.")
    p.add_argument("--v4_mix_search_max_batches", type=int, default=4,
                   help="Max val batches per eta candidate; keeps adaptive mixing cheap.")

    # Optional post-aggregation local alignment. This is local-only computation:
    # no extra upload/downlink, but often fixes A_down/B_local coordinate mismatch.
    p.add_argument("--v4_post_align_steps", type=int, default=0,
                   help="After receiving A_down, run this many local B-only steps before eval.")
    p.add_argument("--v4_post_align_lr", type=float, default=0.0,
                   help="B-only alignment LR; <=0 uses 0.5 * --lr.")
    p.add_argument("--v4_post_align_prox_lambda", type=float, default=0.0,
                   help="Optional proximal regularizer to keep B near pre-alignment B.")

    # v5: validation-routed global/local A path + optional B-only post alignment.
    # This keeps the same one-shot A-only communication protocol while allowing
    # each client to safely reject a harmful global A in high-conflict domains.
    p.add_argument("--v5_route_val_scope", type=str, default="local",
                   choices=["local", "domain", "global"],
                   help="Validation scope for v5 route selection. local is the deployable "
                   "private-client setting; domain is a stronger public-anchor setting.")
    p.add_argument("--v5_route_search_grid", type=str,
                   default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                   help="Eta grid for A_eff = eta*A_down + (1-eta)*A_local.")
    p.add_argument("--v5_route_search_max_batches", type=int, default=4,
                   help="Max validation batches per eta candidate for v5 routing.")
    p.add_argument("--v5_route_tie_margin", type=float, default=0.0,
                   help="If >0, eta candidates within best_loss+margin are treated as ties.")
    p.add_argument("--v5_route_tie_breaker", type=str, default="best",
                   choices=["best", "global", "local", "mixed"],
                   help="Tie preference for v5 routing. best keeps strict validation optimum.")
    p.add_argument("--v5_route_post_align_steps", type=int, default=0,
                   help="After installing routed A_eff, run this many B-only local steps before eval.")
    p.add_argument("--v5_route_post_align_lr", type=float, default=0.0,
                   help="B-only LR for v5 routed alignment; <=0 uses 0.5 * --lr.")
    p.add_argument("--v5_route_post_align_prox_lambda", type=float, default=0.0,
                   help="Keep B near its pre-route-alignment value during v5 B-only alignment.")
    p.add_argument("--v3_conflict_quantile", type=float, default=0.80)
    p.add_argument("--v3_gate_temperature", type=float, default=0.05)
    p.add_argument("--v3_conflict_blend", type=float, default=1.0)
    p.add_argument("--v3_residual_norm_power", type=float, default=1.0)
    p.add_argument("--v3_residual_eps", type=float, default=1e-7)
    p.add_argument("--v3_cluster_mode", type=str, default="domain_prior")
    p.add_argument("--v3_cluster_lambda_min", type=float, default=0.2)
    p.add_argument("--v3_cluster_lambda_max", type=float, default=1.0)
    p.add_argument("--v3_rpca_rank", type=int, default=1)
    p.add_argument("--v3_sparse_quantile", type=float, default=0.80)
    p.add_argument("--v3_domain_cluster_map", type=str, default="")

    # Branch B: SVD
    p.add_argument("--v4_svd_orth_init", type=int, default=1,
                   help="1=QR-orthogonalize A_0 before local training (Branch B)")
    p.add_argument("--v4_svd_refactor", type=int, default=1,
                   help="1=SVD refactor on stacked client A at aggregation (B2)")
    p.add_argument("--v4_svd_procrustes", type=int, default=1)

    # Branch E: Anchor (stub → hier prior until anchor data wired)
    p.add_argument("--v4_anchor_gate_threshold", type=float, default=0.35)
    p.add_argument("--v4_anchor_cluster_lambda", type=float, default=0.5)
    p.add_argument("--v4_use_anchor", type=int, default=1)

    # Branch F: AdaRank (stub → oneshot until heterogeneous rank wired)
    p.add_argument("--v4_adarank_mode", type=str, default="full",
                   choices=["risk16", "full"])
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

    # v5 — same A-only server protocol; personalization happens after the single downlink.
    if t in {"v5_route_mix_align", "v5_route_mix_align_local", "v5_route_mix_align_domain"}:
        return aggregate_models_fedplora_oneshot(global_model, fedplora_uploads, args)
    if t == "v5_rpca_route_mix_align":
        return aggregate_models_fedplora_v3_rpca(global_model, fedplora_uploads, args)

    # Branch B — SVD
    if t in {"v4_svd_orth_only", "v4_svd_full"}:
        if t == "v4_svd_orth_only":
            args.v4_svd_refactor = 0
        else:
            args.v4_svd_refactor = 1
        return aggregate_models_v4_svd(global_model, fedplora_uploads, args)

    # Branch E — anchor (stub)
    if t in {"v4_anchor_gate", "v4_anchor_lambda"}:
        return aggregate_models_v4_anchor(global_model, fedplora_uploads, args)

    # Branch F — AdaRank (stub)
    if t in {"v4_adarank_risk16", "v4_adarank_full"}:
        return aggregate_models_v4_adarank(global_model, fedplora_uploads, args)

    raise ValueError(f"Unknown v4 agg_type: {agg_type}")


# ---------------------------------------------------------------------------
# Federated training loop (single seed)
# ---------------------------------------------------------------------------


def _metrics_args_snapshot(args) -> dict:
    """CLI args only — omit runtime attrs (_fedplora_initial_A, etc.) that hold tensors."""
    return {k: v for k, v in vars(args).items() if not str(k).startswith("_")}


def _write_v4_metrics_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tensor_to_list(payload), f, indent=2, ensure_ascii=False)


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


def _current_lora_a_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if "lora_A" in k and k.endswith("default.weight")
    }


def _install_lora_a_state(model, a_state):
    if not a_state:
        return
    sd = model.state_dict()
    for key, value in a_state.items():
        if key in sd:
            sd[key] = value.to(device=sd[key].device, dtype=sd[key].dtype)
    model.load_state_dict(sd)


def _load_v4_mix_local_a(mix_local_dir, client_id):
    path = Path(mix_local_dir) / f"client_{int(client_id):03d}_A_local.pt"
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu")


def _parse_float_grid(raw, default_eta=0.5):
    vals = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            vals.append(float(item))
        except ValueError:
            continue
    if not vals:
        vals = [float(default_eta)]
    return [min(max(v, 0.0), 1.0) for v in vals]


def _mix_search_rows(benchmark, domain, client_id, args):
    scope = str(getattr(args, "v4_mix_val_scope", "local") or "local").lower()
    return _rows_for_validation_scope(benchmark, domain, client_id, scope)


def _rows_for_validation_scope(benchmark, domain, client_id, scope):
    scope = str(scope or "local").lower()
    val_rows = benchmark.get("val", []) or []
    if scope == "global":
        return val_rows
    if scope == "domain":
        rows = [r for r in val_rows if str(r.get("domain", "")).lower() == str(domain).lower()]
        return rows or val_rows
    rows = [r for r in val_rows if int(r.get("client_id", -1)) == int(client_id)]
    if rows:
        return rows
    client_domain = None
    for c in benchmark.get("clients", []) or []:
        try:
            if int(c.get("client_id", -1)) == int(client_id):
                client_domain = str(c.get("domain", "")).lower()
                break
        except (TypeError, ValueError):
            continue
    if client_domain:
        rows = [r for r in val_rows if str(r.get("domain", "")).lower() == client_domain]
        if rows:
            return rows
    return rows or val_rows


def _v5_route_scope(args):
    agg_type = (getattr(args, "agg_type", "") or "").lower()
    if agg_type.endswith("_domain"):
        return "domain"
    if agg_type.endswith("_local"):
        return "local"
    return str(getattr(args, "v5_route_val_scope", "local") or "local").lower()


def _search_mix_eta_for_client_domain(
    global_model,
    tokenizer,
    benchmark,
    domain,
    client_id,
    shared_state,
    local_state,
    personalized,
    a_local,
    args,
    device,
):
    if not a_local:
        return float(getattr(args, "v4_mix_eta", 0.5))
    grid = _parse_float_grid(
        getattr(args, "v4_mix_search_grid", ""),
        default_eta=float(getattr(args, "v4_mix_eta", 0.5)),
    )
    rows = _mix_search_rows(benchmark, domain, client_id, args)
    if not rows:
        return float(getattr(args, "v4_mix_eta", 0.5))

    val_args = argparse.Namespace(**vars(args))
    val_args.eval_batch_size = int(getattr(args, "eval_batch_size", 0) or 0)
    dl = create_domain_eval_dataloader(rows, tokenizer, val_args)

    client_shared = dict(shared_state)
    client_shared.update(personalized.get(int(client_id), {}))
    broadcast_fedplora_shared_state(global_model, client_shared)
    load_fedplora_local_state(global_model, local_state)
    a_down = _current_lora_a_state(global_model)

    best_eta = float(getattr(args, "v4_mix_eta", 0.5))
    best_loss = float("inf")
    max_batches = int(getattr(args, "v4_mix_search_max_batches", 4) or 0)
    for eta in grid:
        A_eff = build_mixed_A(a_down, a_local, eta)
        _install_lora_a_state(global_model, A_eff)
        stats = compute_lm_eval_stats(global_model, dl, device, max_batches=max_batches)
        loss = float(stats.get("loss", float("inf")))
        if loss < best_loss:
            best_loss = loss
            best_eta = float(eta)

    # Restore the server/personalized A before the caller installs the selected A.
    broadcast_fedplora_shared_state(global_model, client_shared)
    load_fedplora_local_state(global_model, local_state)
    return best_eta


def _search_v5_route_for_client_domain(
    global_model,
    tokenizer,
    benchmark,
    domain,
    client_id,
    shared_state,
    local_state,
    personalized,
    a_local,
    args,
    device,
):
    """Client-side route search for v5.

    The search is local-only and does not change the communication protocol. It
    selects a single A path from a validation loss curve:
        A_eff = eta * A_down + (1 - eta) * A_local.
    eta=1.0 means trust the global A; eta=0.0 means private fallback.
    """
    scope = _v5_route_scope(args)
    if not a_local:
        return {
            "eta": 1.0,
            "loss": float("nan"),
            "route": "global",
            "scope": scope,
            "num_candidates": 0,
        }

    grid = _parse_float_grid(
        getattr(args, "v5_route_search_grid", ""),
        default_eta=float(getattr(args, "v4_mix_eta", 0.5)),
    )
    rows = _rows_for_validation_scope(benchmark, domain, client_id, scope)
    if not rows:
        return {
            "eta": float(getattr(args, "v4_mix_eta", 0.5)),
            "loss": float("nan"),
            "route": "mixed",
            "scope": scope,
            "num_candidates": 0,
        }

    val_args = argparse.Namespace(**vars(args))
    val_args.eval_batch_size = int(getattr(args, "eval_batch_size", 0) or 0)
    dl = create_domain_eval_dataloader(rows, tokenizer, val_args)

    client_shared = dict(shared_state)
    client_shared.update(personalized.get(int(client_id), {}))
    broadcast_fedplora_shared_state(global_model, client_shared)
    load_fedplora_local_state(global_model, local_state)
    a_down = _current_lora_a_state(global_model)

    max_batches = int(getattr(args, "v5_route_search_max_batches", 4) or 0)
    losses = []
    for eta in grid:
        A_eff = build_mixed_A(a_down, a_local, eta)
        _install_lora_a_state(global_model, A_eff)
        stats = compute_lm_eval_stats(global_model, dl, device, max_batches=max_batches)
        losses.append((float(eta), float(stats.get("loss", float("inf")))))

    best_loss = min(loss for _eta, loss in losses)
    margin = max(float(getattr(args, "v5_route_tie_margin", 0.0) or 0.0), 0.0)
    tied = [(eta, loss) for eta, loss in losses if loss <= best_loss + margin]
    tie_breaker = str(getattr(args, "v5_route_tie_breaker", "best") or "best").lower()
    if tie_breaker == "global":
        best_eta, best_loss = max(tied, key=lambda item: item[0])
    elif tie_breaker == "local":
        best_eta, best_loss = min(tied, key=lambda item: item[0])
    elif tie_breaker == "mixed":
        best_eta, best_loss = min(tied, key=lambda item: abs(item[0] - 0.5))
    else:
        best_eta, best_loss = min(losses, key=lambda item: item[1])

    # Restore the server/personalized A before the caller installs the selected A.
    broadcast_fedplora_shared_state(global_model, client_shared)
    load_fedplora_local_state(global_model, local_state)

    if best_eta >= 1.0 - 1e-8:
        route = "global"
    elif best_eta <= 1e-8:
        route = "local"
    else:
        route = "mixed"
    return {
        "eta": float(best_eta),
        "loss": float(best_loss),
        "route": route,
        "scope": scope,
        "num_candidates": int(len(losses)),
    }


def _post_align_local_b(global_model, client_ids, local_states, client_dataloaders, args):
    steps = int(getattr(args, "v4_post_align_steps", 0) or 0)
    if steps <= 0:
        return {"enabled": False, "steps": 0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = float(getattr(args, "v4_post_align_lr", 0.0) or 0.0)
    if lr <= 0:
        lr = 0.5 * float(getattr(args, "lr", 2e-4))
    prox_lam = float(getattr(args, "v4_post_align_prox_lambda", 0.0) or 0.0)
    personalized = getattr(args, "_fedplora_personalized_shared_states", {}) or {}
    shared_state = {
        k: v.detach().cpu().clone()
        for k, v in global_model.state_dict().items()
        if is_fedplora_shared_param_name(k, get_trainable_param_names(global_model))
    }

    old_flags = {name: p.requires_grad for name, p in global_model.named_parameters()}
    losses = []
    try:
        global_model.to(device)
        for idx, client_id in enumerate(client_ids):
            client_shared = dict(shared_state)
            client_shared.update(personalized.get(int(client_id), {}))
            broadcast_fedplora_shared_state(global_model, client_shared)
            load_fedplora_local_state(global_model, local_states[int(client_id)])

            for name, p in global_model.named_parameters():
                p.requires_grad = "lora_B" in name and name.endswith("default.weight")
            trainable = [p for p in global_model.parameters() if p.requires_grad]
            if not trainable:
                continue
            b_ref = {
                k: v.detach().clone()
                for k, v in global_model.state_dict().items()
                if "lora_B" in k and k.endswith("default.weight")
            }
            optimizer = torch.optim.AdamW(trainable, lr=lr)
            global_model.train()
            done = 0
            last_loss = None
            for batch in client_dataloaders[idx]:
                if done >= steps:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = global_model(**batch)
                loss = outputs.loss
                if prox_lam > 0:
                    prox_terms = []
                    for key, cur in global_model.state_dict().items():
                        if key in b_ref:
                            ref = b_ref[key].to(device=cur.device, dtype=cur.dtype)
                            prox_terms.append(torch.mean((cur.float() - ref.float()) ** 2))
                    if prox_terms:
                        loss = loss + prox_lam * torch.stack(prox_terms).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                last_loss = float(loss.detach().cpu().item())
                done += 1
            local_states[int(client_id)] = extract_fedplora_local_state(global_model)
            if last_loss is not None:
                losses.append(last_loss)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for name, p in global_model.named_parameters():
            if name in old_flags:
                p.requires_grad = old_flags[name]

    return {
        "enabled": True,
        "steps": int(steps),
        "lr": float(lr),
        "prox_lambda": float(prox_lam),
        "mean_final_loss": float(np.mean(losses)) if losses else float("nan"),
    }


def _align_current_client_b_once(global_model, dataloader, args):
    """B-only local alignment for the currently installed A_eff.

    This is intentionally client-local and should be called after v5 route
    selection installs A_eff. It returns a new local B state and never uploads it.
    """
    steps = int(getattr(args, "v5_route_post_align_steps", 0) or 0)
    if steps <= 0:
        return None, {"enabled": False, "steps": 0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = float(getattr(args, "v5_route_post_align_lr", 0.0) or 0.0)
    if lr <= 0:
        lr = 0.5 * float(getattr(args, "lr", 2e-4))
    prox_lam = float(getattr(args, "v5_route_post_align_prox_lambda", 0.0) or 0.0)

    old_flags = {name: p.requires_grad for name, p in global_model.named_parameters()}
    losses = []
    try:
        global_model.to(device)
        for name, p in global_model.named_parameters():
            p.requires_grad = "lora_B" in name and name.endswith("default.weight")
        trainable = [p for p in global_model.parameters() if p.requires_grad]
        if not trainable:
            return None, {"enabled": False, "steps": 0, "reason": "no_lora_b"}

        b_ref = {
            k: v.detach().clone()
            for k, v in global_model.state_dict().items()
            if "lora_B" in k and k.endswith("default.weight")
        }
        optimizer = torch.optim.AdamW(trainable, lr=lr)
        global_model.train()
        done = 0
        for batch in dataloader:
            if done >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = global_model(**batch)
            loss = outputs.loss
            if prox_lam > 0:
                prox_terms = []
                for key, cur in global_model.state_dict().items():
                    if key in b_ref:
                        ref = b_ref[key].to(device=cur.device, dtype=cur.dtype)
                        prox_terms.append(torch.mean((cur.float() - ref.float()) ** 2))
                if prox_terms:
                    loss = loss + prox_lam * torch.stack(prox_terms).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu().item()))
            done += 1
        return extract_fedplora_local_state(global_model), {
            "enabled": True,
            "steps": int(done),
            "lr": float(lr),
            "prox_lambda": float(prox_lam),
            "final_loss": float(losses[-1]) if losses else float("nan"),
        }
    finally:
        for name, p in global_model.named_parameters():
            if name in old_flags:
                p.requires_grad = old_flags[name]


def _build_round_block(global_model, round_idx, domain_metrics, macro_acc, worst_acc, macro_ppl, worst_ppl, args):
    round_block = {
        "round": round_idx + 1,
        "domain_macro_token_accuracy": macro_acc,
        "worst_domain_token_accuracy": worst_acc,
        "domain_macro_perplexity": macro_ppl,
        "worst_domain_perplexity": worst_ppl,
        "domain_metrics": domain_metrics,
    }
    v4_stats = getattr(args, "_fedplora_v4_stats", None)
    if v4_stats:
        round_block["v4_stats_summary"] = v4_stats.get("_summary", {})
    oneshot = getattr(args, "_fedplora_oneshot_conflict_stats", None)
    if oneshot and oneshot.get("_summary"):
        round_block["fedplora_oneshot_conflict"] = oneshot.get("_summary", {})
    mix_stats = getattr(args, "_fedplora_v4_mix_stats", None)
    if mix_stats:
        round_block["v4_mix_stats"] = mix_stats
    align_stats = getattr(args, "_fedplora_v4_post_align_stats", None)
    if align_stats:
        round_block["v4_post_align_stats"] = align_stats
    route_stats = getattr(args, "_fedplora_v5_route_stats", None)
    if route_stats:
        round_block["v5_route_stats"] = route_stats
    return round_block


def _run_eval_round(
    global_model,
    client_ids,
    local_states,
    benchmark,
    tokenizer,
    args,
    round_idx,
    client_dataloaders=None,
):
    domain_metrics, macro_acc, worst_acc, macro_ppl, worst_ppl = _evaluate(
        global_model,
        client_ids,
        local_states,
        benchmark,
        tokenizer,
        args,
        client_dataloaders=client_dataloaders,
    )
    return _build_round_block(
        global_model, round_idx, domain_metrics, macro_acc, worst_acc, macro_ppl, worst_ppl, args
    )


def federated_sft_single_seed(args):
    """Run one federation pass for one seed. Returns metrics dict."""
    benchmark = load_domain_sft_benchmark(args.benchmark_dir)
    split_dir = os.path.abspath(os.path.expanduser(args.benchmark_dir))
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

    maybe_apply_default_save_run_checkpoint_dir(args, split_dir)
    skipped = try_skip_if_run_fully_complete(args, split_dir, client_ids)
    if skipped is not None:
        return skipped

    def _eval_cb(global_model, client_ids, local_states, round_idx=0):
        return _run_eval_round(
            global_model,
            client_ids,
            local_states,
            benchmark,
            tokenizer,
            args,
            round_idx,
            client_dataloaders=client_dataloaders,
        )

    resumed = try_resume_eval_only_from_post_agg(args, split_dir, client_ids, _eval_cb)
    if resumed is not None:
        return resumed

    global_model = create_peft_causal_lm_model(args)
    comm = estimate_round_communication_bytes(
        global_model.state_dict(), args.agg_type,
        trainable_param_names=get_trainable_param_names(global_model),
    )
    print(f"[v4] seed={args.seed} agg_type={args.agg_type} "
          f"comm_down={comm['down_bytes_per_client']}B comm_up={comm['up_bytes_per_client']}B")

    init_fedplora_adapters(global_model)
    agg_t = (args.agg_type or "").lower()
    if agg_t.startswith("v4_svd_") and bool(int(getattr(args, "v4_svd_orth_init", 1) or 0)):
        n_orth = orthogonalize_lora_A_in_model(global_model)
        print(f"[v4-svd] orthogonalized {n_orth} LoRA A matrices at init", flush=True)

    initial_A = {
        k: v.detach().cpu().clone()
        for k, v in global_model.state_dict().items()
        if is_lora_a_param_name(k)
    }
    args._fedplora_initial_A = initial_A

    client_store = init_v4_client_store(global_model, client_ids, args)
    local_states = client_store["local_states"]

    metrics = {"args": _metrics_args_snapshot(args), "seed": int(args.seed),
               "benchmark_dir": split_dir, "rounds": [],
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
            v2_train_eval.train_client(
                global_model,
                client_dataloaders[i],
                args,
                client_idx=int(client_id),
            )
            local_state = extract_fedplora_local_state(global_model)
            persist_client_local_state(client_store, client_id, local_state, args)
            local_states[int(client_id)] = local_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fedplora_uploads.append(build_fedplora_upload_package(
                global_model, client_sizes[i],
                client_id=int(client_id),
                domain=args._fedplora_client_domains.get(int(client_id), "unknown"),
            ))
        args._tqdm_desc = None

        print(f"[v4] round {round_idx + 1}: aggregating ({args.agg_type})", flush=True)
        global_model = _dispatch_aggregate(args.agg_type, global_model, fedplora_uploads, args)
        print(f"[v4] round {round_idx + 1}: aggregation done; starting eval "
              f"({len(benchmark['test_domain'])} domains × {len(client_ids)} clients, "
              f"eval_max_batches={getattr(args, 'eval_max_batches', 0) or 'all'})", flush=True)
        args._fedplora_v4_post_align_stats = _post_align_local_b(
            global_model, client_ids, local_states, client_dataloaders, args
        )

        if str(getattr(args, "save_run_checkpoint_dir", "") or "").strip():
            save_v4_run_checkpoint(
                global_model,
                local_states,
                client_ids,
                args,
                split_dir,
                bundle_subdir=f"snapshots/round_{round_idx + 1:03d}_post_agg",
                checkpoint_phase="post_aggregation",
                round_saved_1based=round_idx + 1,
            )

        round_block = _run_eval_round(
            global_model,
            client_ids,
            local_states,
            benchmark,
            tokenizer,
            args,
            round_idx,
            client_dataloaders=client_dataloaders,
        )
        metrics["rounds"].append(round_block)

        if str(getattr(args, "save_run_checkpoint_dir", "") or "").strip():
            save_v4_run_checkpoint(
                global_model,
                local_states,
                client_ids,
                args,
                split_dir,
                checkpoint_phase="final",
                round_saved_1based=round_idx + 1,
                round_metrics=round_block,
            )

    return metrics


def _evaluate(global_model, client_ids, local_states, benchmark, tokenizer, args, client_dataloaders=None):
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
    v5_route_active = agg_type.startswith("v5_") and "_route_" in agg_type
    mix_local_dir = Path(getattr(args, "v4_mix_save_dir", "artifacts/v4_mix_a_local"))
    mix_eta_cache = {}
    mix_eta_values = []
    v5_route_cache = {}
    v5_route_records = []
    v5_aligned_local_states = {}
    v5_align_records = []
    if mix_active:
        print(
            f"[v4-mix] mode={args.v4_mix_mode} eta={args.v4_mix_eta} "
            f"val_scope={getattr(args, 'v4_mix_val_scope', 'local')}",
            flush=True,
        )
    if v5_route_active:
        print(
            f"[v5-route] scope={_v5_route_scope(args)} "
            f"post_align_steps={getattr(args, 'v4_post_align_steps', 0)} "
            f"search_batches={getattr(args, 'v5_route_search_max_batches', 4)}",
            flush=True,
        )

    metrics = {}
    domains_sorted = sorted(by_domain.items())
    n_domains = len(domains_sorted)
    n_clients = len(client_ids)
    for di, (domain, rows) in enumerate(domains_sorted, start=1):
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        stats_per_client = []
        for ci, client_id in enumerate(client_ids, start=1):
            print(
                f"[v4] eval domain {di}/{n_domains} {domain} "
                f"client {ci}/{n_clients} (id={client_id})",
                flush=True,
            )
            client_shared = dict(shared_state)
            client_shared.update(personalized.get(int(client_id), {}))
            broadcast_fedplora_shared_state(global_model, client_shared)
            load_fedplora_local_state(global_model, local_states[int(client_id)])

            if mix_active:
                # Load A_local for this client and mix it with current A (= A_down).
                a_local = _load_v4_mix_local_a(mix_local_dir, client_id)
                if a_local:
                    A_down = {
                        k: v.detach().cpu().clone()
                        for k, v in global_model.state_dict().items()
                        if is_lora_a_param_name(k)
                    }
                    if args.v4_mix_mode == "per_domain":
                        cache_key = (int(client_id), str(domain))
                        if cache_key not in mix_eta_cache:
                            mix_eta_cache[cache_key] = _search_mix_eta_for_client_domain(
                                global_model,
                                tokenizer,
                                benchmark,
                                domain,
                                int(client_id),
                                shared_state,
                                local_states[int(client_id)],
                                personalized,
                                a_local,
                                args,
                                device,
                            )
                        eta = mix_eta_cache[cache_key]
                    else:
                        eta = float(args.v4_mix_eta)
                    mix_eta_values.append(float(eta))
                    A_eff = build_mixed_A(A_down, a_local, eta)
                    # Write A_eff back into the model
                    sd = global_model.state_dict()
                    for k, v in A_eff.items():
                        if k in sd:
                            sd[k] = v.to(device=sd[k].device, dtype=sd[k].dtype)
                    global_model.load_state_dict(sd)

            if v5_route_active:
                # Validation-routed A path. local scope intentionally caches per client:
                # the client chooses one deploy-time route without seeing test-domain data.
                a_local = _load_v4_mix_local_a(mix_local_dir, client_id)
                if a_local:
                    A_down = {
                        k: v.detach().cpu().clone()
                        for k, v in global_model.state_dict().items()
                        if is_lora_a_param_name(k)
                    }
                    scope = _v5_route_scope(args)
                    cache_key = (int(client_id), "__local__") if scope == "local" else (int(client_id), str(domain))
                    if cache_key not in v5_route_cache:
                        v5_route_cache[cache_key] = _search_v5_route_for_client_domain(
                            global_model,
                            tokenizer,
                            benchmark,
                            domain,
                            int(client_id),
                            shared_state,
                            local_states[int(client_id)],
                            personalized,
                            a_local,
                            args,
                            device,
                        )
                    route_info = dict(v5_route_cache[cache_key])
                    eta = float(route_info.get("eta", 1.0))
                    route_info.update({"client_id": int(client_id), "domain": str(domain)})
                    v5_route_records.append(route_info)
                    A_eff = build_mixed_A(A_down, a_local, eta)
                    _install_lora_a_state(global_model, A_eff)
                    if cache_key in v5_aligned_local_states:
                        load_fedplora_local_state(global_model, v5_aligned_local_states[cache_key])
                    elif (
                        int(getattr(args, "v5_route_post_align_steps", 0) or 0) > 0
                        and client_dataloaders is not None
                    ):
                        new_local_state, align_info = _align_current_client_b_once(
                            global_model,
                            client_dataloaders[ci - 1],
                            args,
                        )
                        if new_local_state is not None:
                            v5_aligned_local_states[cache_key] = new_local_state
                            load_fedplora_local_state(global_model, new_local_state)
                        align_info.update({"client_id": int(client_id)})
                        align_info.update({"domain": str(domain), "cache_key": str(cache_key)})
                        v5_align_records.append(align_info)

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
    print(
        f"[v4] eval done: macro_acc={macro_acc:.4f} worst_acc={worst_acc:.4f}",
        flush=True,
    )
    if mix_active:
        vals = np.asarray(mix_eta_values, dtype=np.float64) if mix_eta_values else np.asarray([])
        args._fedplora_v4_mix_stats = {
            "mode": str(args.v4_mix_mode),
            "val_scope": str(getattr(args, "v4_mix_val_scope", "local")),
            "num_eta": int(vals.size),
            "mean_eta": float(vals.mean()) if vals.size else float("nan"),
            "min_eta": float(vals.min()) if vals.size else float("nan"),
            "max_eta": float(vals.max()) if vals.size else float("nan"),
        }
    if v5_route_active:
        vals = np.asarray([float(r.get("eta", float("nan"))) for r in v5_route_records], dtype=np.float64)
        vals = vals[~np.isnan(vals)] if vals.size else vals
        routes = {}
        for r in v5_route_records:
            route = str(r.get("route", "unknown"))
            routes[route] = routes.get(route, 0) + 1
        align_losses = [
            float(r.get("final_loss", float("nan"))) for r in v5_align_records
            if not math.isnan(float(r.get("final_loss", float("nan"))))
        ]
        args._fedplora_v5_route_stats = {
            "scope": _v5_route_scope(args),
            "num_routes": int(len(v5_route_records)),
            "num_cached_searches": int(len(v5_route_cache)),
            "route_counts": routes,
            "mean_eta": float(vals.mean()) if vals.size else float("nan"),
            "min_eta": float(vals.min()) if vals.size else float("nan"),
            "max_eta": float(vals.max()) if vals.size else float("nan"),
            "tie_breaker": str(getattr(args, "v5_route_tie_breaker", "best")),
            "tie_margin": float(getattr(args, "v5_route_tie_margin", 0.0) or 0.0),
            "search_max_batches": int(getattr(args, "v5_route_search_max_batches", 4) or 0),
            "post_align": {
                "enabled": bool(v5_align_records),
                "num_align_states": int(len(v5_aligned_local_states)),
                "num_clients_aligned": int(len({
                    int(r.get("client_id")) for r in v5_align_records if "client_id" in r
                })),
                "mean_final_loss": float(np.mean(align_losses)) if align_losses else float("nan"),
            },
        }
    return metrics, macro_acc, worst_acc, macro_ppl, worst_ppl


def main():
    args = build_parser().parse_args()
    log_file, orig_out, orig_err, log_path = setup_run_logging(
        args,
        log_dir=getattr(args, "log_dir", "log_v4"),
        filename_prefix=getattr(args, "log_filename_prefix", "v4_sft"),
    )
    try:
        if str(getattr(args, "eval_only_from_checkpoint", "") or "").strip():
            split_dir = os.path.abspath(os.path.expanduser(args.benchmark_dir))
            benchmark = load_domain_sft_benchmark(split_dir)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, use_fast=False, trust_remote_code=args.trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            client_ids, client_dataloaders = create_domain_client_dataloaders(
                benchmark["train"], tokenizer, args
            )
            args.num_clients = len(client_ids)
            maybe_apply_default_save_run_checkpoint_dir(args, split_dir)

            def _eval_cb(global_model, cids, local_states, round_idx=0):
                return _run_eval_round(
                    global_model,
                    cids,
                    local_states,
                    benchmark,
                    tokenizer,
                    args,
                    round_idx,
                    client_dataloaders=client_dataloaders,
                )

            result = try_resume_eval_only_from_post_agg(args, split_dir, client_ids, _eval_cb)
            if result is None:
                raise RuntimeError(
                    f"--eval_only_from_checkpoint failed for {args.eval_only_from_checkpoint}"
                )
            per_seed_results = [result]
            seeds = [int(args.seed)]
        else:
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
        out_path = os.path.join(args.metrics_output_dir, out_name)
        _write_v4_metrics_json(
            out_path,
            {"per_seed": per_seed_results, "agg_type": args.agg_type, "seeds": seeds},
        )
        print(f"[v4] wrote metrics to {out_path}")
    finally:
        restore_logging(log_file, orig_out, orig_err)


if __name__ == "__main__":
    main()
