import os

# Must run before any import that loads HuggingFace tokenizers (see utilities.data_utils).
# With DataLoader num_workers>0, Rust tokenizers + fork otherwise spam warnings and can stall.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import shutil
import sys
import time
import traceback
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
    shutdown_dataloader_workers,
)
from methods.fedavg_normal import aggregate_models_normal
from methods.ffa_lora import aggregate_models_ffa
from methods.fedsa_lora import aggregate_models_fedsa_lora
from methods.flora import aggregate_models_flora
from methods.flexlora import aggregate_models_flexlora
from methods.feddat import aggregate_models_feddat
from methods.fedp_lora import aggregate_models_fedplora, build_fedplora_upload_package
from methods.fedplora_oneshot import (
    aggregate_models_fedplora_oneshot,
    aggregate_models_fedplora_v3_cluster,
    aggregate_models_fedplora_v3_lite,
    aggregate_models_fedplora_v3_rpca,
)
from methods.fedalt import aggregate_models_fedalt, build_fedalt_upload_package
from methods.yoco import aggregate_models_yoco
from utilities.models import (
    create_peft_causal_lm_model,
    create_peft_causal_lm_ffa_model,
    init_fedplora_adapters,
)
from utilities.state_dict_ops import (
    broadcast_fedplora_shared_state,
    extract_fedalt_local_state,
    extract_fedplora_local_state,
    extract_round_broadcast_state,
    extract_trainable_state_dict,
    load_fedalt_local_state,
    load_fedplora_local_state,
    load_partial_state_dict,
)
from utilities.sft_checkpoint_paths import default_save_run_checkpoint_dir, run_bundle_stem
from utilities.train_eval import train_client
from utilities.utils import (
    estimate_round_communication_bytes,
    get_fedplora_shared_param_names,
    get_trainable_param_names,
    is_flexlora_agg,
    is_feddat_agg,
    is_flora_agg,
    is_fedalt_agg,
    is_fedalt_sequential_agg,
    is_fedsa_lora_agg,
    is_fedplora_oneshot_agg,
    is_fedplora_oneshot_family_agg,
    is_fedplora_v3_agg,
    is_fedplora_shared_param_name,
    is_fedplora_multiround_agg,
    is_lora_a_disk_agg,
    is_lora_a_param_name,
    is_memory_global_agg_agg,
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
parser.add_argument("--rounds", type=int, default=1)
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
parser.add_argument(
    "--save_run_checkpoint_dir",
    type=str,
    default="",
    help="Run checkpoint bundle root. If empty, auto-set to "
        "<trained_models_root>/<agg>_<model>_<benchmark_tail>_r<R>_e<E>_seed<S> (no timestamps). "
        "After each round's aggregation (before eval) writes snapshots/round_XXX_post_agg/ unless "
        "--skip_post_agg_snapshots; after all rounds + metrics JSON, writes the final bundle here.",
)
parser.add_argument(
    "--skip_post_agg_snapshots",
    action="store_true",
    help="With --save_run_checkpoint_dir, skip per-round snapshots under snapshots/ (saves disk IO).",
)
parser.add_argument(
    "--trained_models_root",
    type=str,
    default="",
    help="When --save_run_checkpoint_dir is empty: store bundles under this directory "
        "(default: <repo>/../trained_models). Env TRAINED_MODELS_ROOT overrides when this flag is empty.",
)
parser.add_argument(
    "--no_auto_save_run_checkpoint",
    action="store_true",
    help="Do not auto-set --save_run_checkpoint_dir to trained_models/<stem> (disables default on-disk bundle).",
)
parser.add_argument(
    "--force_retrain",
    action="store_true",
    help="Ignore an existing successful bundle at the resolved save path and train from scratch.",
)
parser.add_argument(
    "--eval_only_from_checkpoint",
    type=str,
    default="",
    help="Path to a directory from --save_run_checkpoint_dir (final root or snapshots/round_XXX_post_agg); "
    "skips training and runs the same eval as the end of a round (domain macro + optional "
    "--eval_personalization_metrics). Does not replace mechanism ablation (those need retraining). "
    "Pass the same --model as in meta.",
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
    help="Legacy CLI (unused by server); kept for batch script compatibility.",
)
parser.add_argument(
    "--yoco_aggregate_mode",
    type=str,
    default="conflict",
    choices=["conflict", "fedavg"],
    help="YOCO server: 'conflict' = FedMLLM aggregate_lora_weights (B-similarity); "
    "'fedavg' = legacy sample-size FedAvg (old checkpoints).",
)
parser.add_argument(
    "--yoco_conflict_method",
    type=str,
    default="avgm",
    choices=["avgm", "mean"],
    help="FedMLLM conflict branch inner method (default avgm).",
)
parser.add_argument(
    "--yoco_sign_lambda",
    type=float,
    default=0.01,
    help="YOCO local training: sign consistency penalty on LoRA B vs round-start global B.",
)
parser.add_argument(
    "--yoco_eval_use_local_clients",
    action="store_true",
    help="YOCO eval legacy alias for --memory_agg_eval_use_local_clients.",
)
parser.add_argument(
    "--memory_agg_eval_use_local_clients",
    action="store_true",
    help="Memory-agg global methods (normal/ffa/flora/flexlora/feddat/yoco): eval each "
    "client's local LoRA instead of server-aggregated global LoRA.",
)
parser.add_argument(
    "--oneshot_consensus_power",
    type=float,
    default=2.0,
    help="FedPLoRA-Oneshot: exponent for row agreement with the initial shared A0.",
)
parser.add_argument(
    "--oneshot_importance_power",
    type=float,
    default=1.0,
    help="FedPLoRA-Oneshot: exponent for private-B row-importance statistics.",
)
parser.add_argument(
    "--oneshot_importance_clip",
    type=float,
    default=5.0,
    help="FedPLoRA-Oneshot: cap row-importance weights; <=0 disables the cap.",
)
parser.add_argument(
    "--oneshot_conflict_threshold",
    type=float,
    default=0.35,
    help="FedPLoRA-Oneshot: row conflict above this threshold is blended toward A0.",
)
parser.add_argument(
    "--oneshot_conflict_blend",
    type=float,
    default=1.0,
    help="FedPLoRA-Oneshot: maximum high-conflict fallback strength toward A0.",
)
parser.add_argument(
    "--oneshot_scale_clip_ratio",
    type=float,
    default=0.0,
    help="FedPLoRA-Oneshot: if >1, clip aggregated row norm to [A0/r, A0*r].",
)
parser.add_argument(
    "--oneshot_anchor_lambda",
    type=float,
    default=1e-4,
    help="FedPLoRA-Oneshot local training: signed row-direction anchor to initial A0.",
)
parser.add_argument(
    "--oneshot_prox_lambda",
    type=float,
    default=0.0,
    help="FedPLoRA-Oneshot local training: optional MSE proximity to initial A0.",
)
parser.add_argument(
    "--oneshot_no_keep_init_on_conflict",
    action="store_true",
    help="FedPLoRA-Oneshot ablation: disable fallback to initial A0 on high-conflict rows.",
)
parser.add_argument(
    "--oneshot_orthogonalize",
    action="store_true",
    help="FedPLoRA-Oneshot ablation: QR-orthogonalize A rows after aggregation (off by default).",
)
parser.add_argument("--v3_conflict_quantile", type=float, default=0.80)
parser.add_argument("--v3_gate_temperature", type=float, default=0.05)
parser.add_argument("--v3_conflict_blend", type=float, default=1.0)
parser.add_argument("--v3_residual_norm_power", type=float, default=1.0)
parser.add_argument("--v3_residual_eps", type=float, default=1e-7)
parser.add_argument(
    "--v3_cluster_mode",
    type=str,
    default="domain_prior",
    help="v3-cluster/rpca: domain_prior | custom (use --v3_domain_cluster_map).",
)
parser.add_argument("--v3_cluster_lambda_min", type=float, default=0.2)
parser.add_argument("--v3_cluster_lambda_max", type=float, default=1.0)
parser.add_argument("--v3_rpca_rank", type=int, default=1)
parser.add_argument("--v3_sparse_quantile", type=float, default=0.80)
parser.add_argument(
    "--v3_domain_cluster_map",
    type=str,
    default="",
    help="Optional domain:cluster pairs, e.g. math:capability,code:capability",
)
parser.add_argument(
    "--feddat_teacher_lambda",
    type=float,
    default=0.01,
    help="FedDAT: MSE proximal to round-start global LoRA (teacher).",
)
parser.add_argument(
    "--eval_max_batches",
    type=int,
    default=0,
    help="If > 0, cap batches per eval forward pass (per client×domain). 0 = full eval. Does not affect training.",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=0,
    help="Eval DataLoader batch size; 0 uses --batch_size. Use a larger value when GPU memory allows to speed eval.",
)
parser.add_argument(
    "--train_max_steps_per_client",
    type=int,
    default=0,
    help="If > 0, cap optimizer steps per client per round (pilot / smoke). 0 = full local_epoch(s) over the dataloader.",
)
parser.add_argument(
    "--max_train_samples_per_client",
    type=int,
    default=0,
    help="If > 0, subsample training rows per client (seed+client_id). 0 = use full client shard.",
)
parser.add_argument(
    "--dataloader_num_workers",
    type=int,
    default=0,
    help="DataLoader workers for training (Linux: try 4–8 if CPU allows). 0 loads in the main process.",
)
parser.add_argument(
    "--eval_dataloader_num_workers",
    type=int,
    default=0,
    help="DataLoader workers for eval only (training still uses --dataloader_num_workers). "
    "Default 0 avoids extra FDs (OSError [Errno 24]) on shared clusters; try 2–4 when ulimit -n "
    "is high. Workers are always shut down after each eval pass when num_workers>0.",
)
parser.add_argument(
    "--dataloader_pin_memory",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Pin CPU tensors when CUDA is available (default: true). Use --no-dataloader-pin-memory to disable.",
)
parser.add_argument(
    "--dataloader_persistent_workers",
    action="store_true",
    help="If set with --dataloader_num_workers > 0, sets persistent_workers=True.",
)
parser.add_argument(
    "--attn_implementation",
    type=str,
    default="",
    choices=["", "sdpa", "flash_attention_2", "eager"],
    help="HF causal LM attention backend; sdpa speeds Llama on PyTorch 2+ when supported. Empty = library default.",
)
parser.add_argument(
    "--tokenizer_use_fast",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use fast tokenizer when available (default: true).",
)
parser.add_argument(
    "--eval_personalization_metrics",
    action="store_true",
    help="Also report personalization block: local / in-domain / off-domain macro "
    "token_accuracy, perplexity, loss; plus gap_token_accuracy (local-off), gap_perplexity (off-local), "
    "gap_loss (off-local).",
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


def compute_lm_eval_stats(model, dataloader, device, max_batches=0):
    """
    Causal LM eval on SFT batches: mean batch loss (same convention as HF outputs.loss),
    micro-averaged next-token accuracy on non-masked label positions (response tokens only),
    and perplexity = exp(mean batch loss) as a cheap global summary.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    steps = 0
    total_correct = 0
    total_valid = 0
    try:
        it = iter(dataloader)
        with torch.inference_mode():
            while True:
                if max_batches and steps >= max_batches:
                    break
                try:
                    batch = next(it)
                except StopIteration:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                outputs = model(**batch)
                total_loss += float(outputs.loss.detach().cpu().item())
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                preds = shift_logits.argmax(dim=-1)
                mask = shift_labels.ne(-100)
                if mask.any():
                    total_correct += int((preds[mask] == shift_labels[mask]).sum().cpu())
                    total_valid += int(mask.sum().cpu())
                steps += 1
    finally:
        # num_workers>0: tear down workers after each eval pass (full or early) to avoid FD leaks.
        if int(getattr(dataloader, "num_workers", 0) or 0) > 0:
            shutdown_dataloader_workers(dataloader)
    mean_loss = total_loss / max(steps, 1)
    tok_acc = total_correct / max(total_valid, 1)
    try:
        ppl = float(math.exp(min(mean_loss, 80.0)))
    except OverflowError:
        ppl = float("inf")
    return {
        "loss": mean_loss,
        "token_accuracy": float(tok_acc),
        "perplexity": ppl,
        "n_eval_batches": int(steps),
    }


def compute_lm_loss(model, dataloader, device, max_batches=0):
    """Backward-compatible: mean eval loss only."""
    return compute_lm_eval_stats(model, dataloader, device, max_batches=max_batches)["loss"]


def evaluate_domain_macro(client_models, domain_rows, tokenizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_domain = group_rows_by_domain(domain_rows)
    metrics = {}
    for domain, rows in sorted(by_domain.items()):
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        stats_list = [
            compute_lm_eval_stats(model, dl, device) for model in client_models
        ]
        metrics[domain] = {
            "loss": float(np.mean([s["loss"] for s in stats_list])),
            "token_accuracy": float(np.mean([s["token_accuracy"] for s in stats_list])),
            "perplexity": float(np.mean([s["perplexity"] for s in stats_list])),
        }
    losses = [v["loss"] for v in metrics.values()]
    macro = float(np.mean(losses)) if losses else float("nan")
    worst = float(max(losses)) if losses else float("nan")
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


def _client_rotw_state_path(base_dir, client_id):
    return os.path.join(base_dir, f"client_{int(client_id):03d}_rotw.pt")


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


def _ensure_sequential_fedalt_states(model, client_ids, args):
    """FedALT: per-client Individual LoRA (A+B) plus optional RoTW snapshots on disk."""
    state_dir = os.path.join(_fedplora_disk_state_dir(args), "fedalt")
    if getattr(args, "save_client_state_to_disk", False):
        os.makedirs(state_dir, exist_ok=True)
        seed_sd = extract_fedalt_local_state(model)
        for client_id in client_ids:
            path = _client_state_path(state_dir, client_id)
            if not os.path.isfile(path):
                _save_client_local_state(seed_sd, state_dir, client_id)
        return {"mode": "disk", "state_dir": state_dir, "rotw_states": {}}

    initial = extract_fedalt_local_state(model)
    local_states = {
        int(client_id): {k: v.clone() for k, v in initial.items()}
        for client_id in client_ids
    }
    return {
        "mode": "memory",
        "state_dir": state_dir,
        "local_states": local_states,
        "rotw_states": {int(cid): {} for cid in client_ids},
    }


def _get_client_rotw_state(client_store, client_id):
    if client_store.get("rotw_states") is not None:
        if client_store["mode"] == "disk":
            path = _client_rotw_state_path(client_store["state_dir"], client_id)
            if not os.path.isfile(path):
                return {}
            return torch.load(path, map_location="cpu")
        return client_store["rotw_states"].get(int(client_id), {})
    return {}


def _set_client_rotw_state(client_store, client_id, rotw_state):
    if not rotw_state:
        return
    if client_store["mode"] == "disk":
        os.makedirs(client_store["state_dir"], exist_ok=True)
        path = _client_rotw_state_path(client_store["state_dir"], client_id)
        tmp_path = path + ".tmp"
        torch.save(rotw_state, tmp_path)
        os.replace(tmp_path, path)
    else:
        client_store.setdefault("rotw_states", {})[int(client_id)] = rotw_state


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
        stats_per_client = []
        for client_id in tqdm(
            client_ids,
            desc=f"eval {domain}",
            leave=False,
            dynamic_ncols=True,
        ):
            local_state = _get_client_local_state(client_store, client_id)
            if is_fedalt_sequential_agg(args.agg_type):
                load_fedalt_local_state(global_model, local_state)
            else:
                broadcast_fedplora_shared_state(global_model, shared_state)
                load_fedplora_local_state(global_model, local_state)
            stats_per_client.append(
                compute_lm_eval_stats(global_model, dl, device, max_batches=eval_cap)
            )
        metrics[domain] = {
            "loss": float(np.mean([s["loss"] for s in stats_per_client])),
            "token_accuracy": float(
                np.mean([s["token_accuracy"] for s in stats_per_client])
            ),
            "perplexity": float(np.mean([s["perplexity"] for s in stats_per_client])),
        }
    losses = [v["loss"] for v in metrics.values()]
    accs = [v["token_accuracy"] for v in metrics.values()]
    ppls = [v["perplexity"] for v in metrics.values()]
    macro = float(np.mean(losses)) if losses else float("nan")
    worst = float(max(losses)) if losses else float("nan")
    macro_tok = float(np.mean(accs)) if accs else float("nan")
    worst_tok = float(min(accs)) if accs else float("nan")
    macro_ppl = float(np.mean(ppls)) if ppls else float("nan")
    worst_ppl = float(max(ppls)) if ppls else float("nan")
    return metrics, macro, worst, macro_tok, worst_tok, macro_ppl, worst_ppl


def _client_id_to_home_domain(clients_manifest):
    return {int(c["client_id"]): str(c["domain"]) for c in clients_manifest}


def _mean_eval_stats(stats_list):
    """Macro mean over a list of compute_lm_eval_stats dicts (loss / token_accuracy / perplexity)."""
    if not stats_list:
        return {
            "loss": float("nan"),
            "token_accuracy": float("nan"),
            "perplexity": float("nan"),
        }
    return {
        "loss": float(np.mean([s["loss"] for s in stats_list])),
        "token_accuracy": float(np.mean([s["token_accuracy"] for s in stats_list])),
        "perplexity": float(np.mean([s["perplexity"] for s in stats_list])),
    }


def _load_client_eval_state(global_model, client_store, client_id, shared_state, args):
    local_state = _get_client_local_state(client_store, client_id)
    if is_fedalt_sequential_agg(args.agg_type):
        load_fedalt_local_state(global_model, local_state)
    else:
        broadcast_fedplora_shared_state(global_model, shared_state)
        load_fedplora_local_state(global_model, local_state)


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

    local_stats = []
    for client_id in client_ids:
        rows = by_client_local.get(int(client_id), [])
        if not rows:
            continue
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        _load_client_eval_state(global_model, client_store, client_id, shared_state, args)
        local_stats.append(
            compute_lm_eval_stats(global_model, dl, device, max_batches=eval_cap)
        )
    loc = _mean_eval_stats(local_stats)

    off_stats = []
    for client_id in client_ids:
        home = id2home.get(int(client_id))
        if not home:
            continue
        for domain, rows in sorted(by_domain_test.items()):
            if domain == home or not rows:
                continue
            dl = create_domain_eval_dataloader(rows, tokenizer, args)
            _load_client_eval_state(global_model, client_store, client_id, shared_state, args)
            off_stats.append(
                compute_lm_eval_stats(global_model, dl, device, max_batches=eval_cap)
            )
    off = _mean_eval_stats(off_stats)

    in_dom_dt_stats = []
    for client_id in client_ids:
        home = id2home.get(int(client_id))
        if not home:
            continue
        rows = by_domain_test.get(home, [])
        if not rows:
            continue
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        _load_client_eval_state(global_model, client_store, client_id, shared_state, args)
        in_dom_dt_stats.append(
            compute_lm_eval_stats(global_model, dl, device, max_batches=eval_cap)
        )
    indt = _mean_eval_stats(in_dom_dt_stats)

    gap_loss = off["loss"] - loc["loss"]
    gap_tok = loc["token_accuracy"] - off["token_accuracy"]
    gap_ppl = off["perplexity"] - loc["perplexity"]

    return {
        "client_local_macro_token_accuracy": loc["token_accuracy"],
        "client_local_macro_perplexity": loc["perplexity"],
        "client_local_macro_loss": loc["loss"],
        "off_domain_macro_token_accuracy": off["token_accuracy"],
        "off_domain_macro_perplexity": off["perplexity"],
        "off_domain_macro_loss": off["loss"],
        "in_domain_domain_test_macro_token_accuracy": indt["token_accuracy"],
        "in_domain_domain_test_macro_perplexity": indt["perplexity"],
        "in_domain_domain_test_macro_loss": indt["loss"],
        "personalization_gap_token_accuracy": gap_tok,
        "personalization_gap_perplexity": gap_ppl,
        "personalization_gap_loss": gap_loss,
    }


def _evaluate_personalization_metrics_full_state(
    global_model,
    eval_store,
    eval_client_ids,
    benchmark,
    tokenizer,
    args,
):
    """Personalization metrics for memory-agg clients (normal / ffa / fedex / flora, etc.).

    Client payloads are trainable-only partial dicts; load_partial_state_dict merges with current base.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cap = int(getattr(args, "eval_max_batches", 0) or 0)
    id2home = _client_id_to_home_domain(benchmark["clients"])
    by_client_local = group_rows_by_client(benchmark["test_local"])
    by_domain_test = group_rows_by_domain(benchmark["test_domain"])

    local_stats = []
    for idx in eval_client_ids:
        rows = by_client_local.get(int(idx), [])
        if not rows:
            continue
        dl = create_domain_eval_dataloader(rows, tokenizer, args)
        state = _get_client_local_state(eval_store, idx)
        load_partial_state_dict(global_model, state)
        local_stats.append(
            compute_lm_eval_stats(global_model, dl, device, max_batches=eval_cap)
        )
    loc = _mean_eval_stats(local_stats)

    off_stats = []
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
            off_stats.append(
                compute_lm_eval_stats(global_model, dl, device, max_batches=eval_cap)
            )
    off = _mean_eval_stats(off_stats)

    in_dom_dt_stats = []
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
        in_dom_dt_stats.append(
            compute_lm_eval_stats(global_model, dl, device, max_batches=eval_cap)
        )
    indt = _mean_eval_stats(in_dom_dt_stats)

    gap_loss = off["loss"] - loc["loss"]
    gap_tok = loc["token_accuracy"] - off["token_accuracy"]
    gap_ppl = off["perplexity"] - loc["perplexity"]

    return {
        "client_local_macro_token_accuracy": loc["token_accuracy"],
        "client_local_macro_perplexity": loc["perplexity"],
        "client_local_macro_loss": loc["loss"],
        "off_domain_macro_token_accuracy": off["token_accuracy"],
        "off_domain_macro_perplexity": off["perplexity"],
        "off_domain_macro_loss": off["loss"],
        "in_domain_domain_test_macro_token_accuracy": indt["token_accuracy"],
        "in_domain_domain_test_macro_perplexity": indt["perplexity"],
        "in_domain_domain_test_macro_loss": indt["loss"],
        "personalization_gap_token_accuracy": gap_tok,
        "personalization_gap_perplexity": gap_ppl,
        "personalization_gap_loss": gap_loss,
    }


def _norm_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(p)))


def _maybe_apply_default_save_run_checkpoint_dir(args, split_dir):
    if getattr(args, "no_auto_save_run_checkpoint", False):
        return
    if str(getattr(args, "save_run_checkpoint_dir", "") or "").strip():
        return
    tmr = getattr(args, "trained_models_root", None)
    args.save_run_checkpoint_dir = default_save_run_checkpoint_dir(
        _ROOT,
        (tmr or "").strip() or None,
        agg_type=args.agg_type,
        model_path=args.model,
        benchmark_split_dir=split_dir,
        rounds=int(args.rounds),
        local_epochs=int(args.local_epochs),
        seed=int(args.seed),
    )
    print(f"[setup] auto save_run_checkpoint_dir={args.save_run_checkpoint_dir}", flush=True)


def _load_checkpoint_ok(bundle_dir: str):
    p = os.path.join(bundle_dir, "checkpoint_ok.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _resume_meta_matches(meta: dict, args, split_dir: str, client_ids) -> bool:
    def _bad(reason: str) -> bool:
        print(f"[resume] checkpoint meta mismatch ({reason}); will retrain.", flush=True)
        return False

    if meta.get("agg_type") != args.agg_type:
        return _bad("agg_type")
    if int(meta.get("train_rounds", -1)) != int(args.rounds):
        return _bad("train_rounds")
    if int(meta.get("train_local_epochs", -1)) != int(args.local_epochs):
        return _bad("train_local_epochs")
    if _norm_path(str(meta.get("benchmark_dir", ""))) != _norm_path(split_dir):
        return _bad("benchmark_dir")
    if _norm_path(str(meta.get("model", ""))) != _norm_path(args.model):
        return _bad("model")
    if int(meta.get("seed", -1)) != int(args.seed):
        return _bad("seed")
    if int(meta.get("num_clients", -1)) != int(args.num_clients):
        return _bad("num_clients")
    meta_cids = meta.get("client_ids")
    want = [int(x) for x in client_ids]
    if meta_cids != want:
        return _bad("client_ids")
    if int(meta.get("lora_r", -1)) != int(args.lora_r):
        return _bad("lora_r")
    if int(meta.get("lora_alpha", -1)) != int(args.lora_alpha):
        return _bad("lora_alpha")
    if float(meta.get("lora_dropout", -1.0)) != float(args.lora_dropout):
        return _bad("lora_dropout")
    if str(meta.get("target_modules", "")) != str(args.target_modules):
        return _bad("target_modules")
    if str(meta.get("torch_dtype", "")) != str(args.torch_dtype):
        return _bad("torch_dtype")
    if bool(meta.get("use_ffa_peft", False)) != (args.agg_type == "ffa"):
        return _bad("use_ffa_peft")
    if bool(meta.get("disk_sequential_protocol", False)) != bool(
        is_lora_a_disk_agg(args.agg_type) or is_fedalt_sequential_agg(args.agg_type)
    ):
        return _bad("disk_sequential_protocol")
    return True


def _try_skip_if_run_fully_complete(args, split_dir, client_ids) -> bool:
    """eval-after bundle at root: skip training and evaluation (method already done)."""
    if getattr(args, "force_retrain", False):
        return False
    bundle = str(getattr(args, "save_run_checkpoint_dir", "") or "").strip()
    if not bundle:
        return False
    bundle = os.path.abspath(os.path.expanduser(bundle))
    ok = _load_checkpoint_ok(bundle)
    meta_path = os.path.join(bundle, "run_checkpoint_meta.json")
    if not ok or not ok.get("ok") or not os.path.isfile(meta_path):
        return False
    phase = str(ok.get("checkpoint_phase", "final") or "final")
    if phase != "final":
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not _resume_meta_matches(meta, args, split_dir, client_ids):
        return False
    print(
        f"[resume] Run fully complete (root checkpoint_ok phase=final) at {bundle}; "
        "skipping training and evaluation.",
        flush=True,
    )
    return True


def _list_post_agg_snapshot_dirs(bundle_dir: str):
    snap = os.path.join(bundle_dir, "snapshots")
    if not os.path.isdir(snap):
        return []
    out = []
    for name in os.listdir(snap):
        path = os.path.join(snap, name)
        if not os.path.isdir(path):
            continue
        if not (name.startswith("round_") and name.endswith("_post_agg")):
            continue
        if name.endswith("_failed"):
            continue
        mid = name[len("round_") : -len("_post_agg")]
        try:
            rid = int(mid)
        except ValueError:
            continue
        out.append((rid, path))
    out.sort(key=lambda x: -x[0])
    return [p for _, p in out]


def _try_resume_eval_only_from_latest_post_agg_snapshot(args, split_dir, client_ids) -> bool:
    """
    Latest eval-before (post-aggregation) snapshot: same tensors eval would use next; skip training, run eval.
    """
    if getattr(args, "force_retrain", False):
        return False
    bundle = str(getattr(args, "save_run_checkpoint_dir", "") or "").strip()
    if not bundle:
        return False
    bundle = os.path.abspath(os.path.expanduser(bundle))
    for snap_dir in _list_post_agg_snapshot_dirs(bundle):
        ok = _load_checkpoint_ok(snap_dir)
        meta_path = os.path.join(snap_dir, "run_checkpoint_meta.json")
        if not ok or not ok.get("ok") or not os.path.isfile(meta_path):
            continue
        if str(ok.get("checkpoint_phase", "")) != "post_aggregation":
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if str(meta.get("checkpoint_phase", "")) != "post_aggregation":
            continue
        if not _resume_meta_matches(meta, args, split_dir, client_ids):
            continue
        print(
            f"[resume] Found latest post-aggregation snapshot (eval-before save) at {snap_dir}; "
            "skipping training and running evaluation only.",
            flush=True,
        )
        args.eval_only_from_checkpoint = snap_dir
        eval_only_from_checkpoint(args)
        return True
    return False


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


def _metrics_recommended_kpis(include_personalization: bool) -> dict:
    """Unified index for papers / dashboards: acc, ppl, communication, oneshot conflict."""
    kpi = {
        "token_accuracy": [
            "domain_macro_token_accuracy",
            "worst_domain_token_accuracy",
        ],
        "perplexity": [
            "domain_macro_perplexity",
            "worst_domain_perplexity",
        ],
        "communication": {
            "path": "communication",
            "fields": ["down_bytes_per_client", "up_bytes_per_client", "agg_type"],
        },
        "fedplora_oneshot_conflict": {
            "path": "rounds[i].fedplora_oneshot_conflict",
            "note": "Present after FedPLoRA-Oneshot server aggregation; "
            "typically mean_conflict, max_conflict, high_conflict_row_frac, mean_init_gate. "
            "Other agg_type: omit or {}. Eval-only reload uses checkpoint meta when available.",
        },
    }
    if include_personalization:
        kpi["token_accuracy"].extend(
            [
                "client_local_macro_token_accuracy",
                "in_domain_domain_test_macro_token_accuracy",
                "off_domain_macro_token_accuracy",
            ]
        )
        kpi["perplexity"].extend(
            [
                "client_local_macro_perplexity",
                "in_domain_domain_test_macro_perplexity",
                "off_domain_macro_perplexity",
            ]
        )
        kpi["personalization_gaps"] = [
            "personalization_gap_token_accuracy",
            "personalization_gap_perplexity",
        ]
    return kpi


RUN_CHECKPOINT_VERSION = 1


def _memory_agg_eval_use_local_clients(args):
    return bool(
        getattr(args, "memory_agg_eval_use_local_clients", False)
        or (
            is_yoco_agg(getattr(args, "agg_type", None))
            and getattr(args, "yoco_eval_use_local_clients", False)
        )
    )


def _reaggregate_memory_global_model(global_model, client_states_for_agg, args):
    """Rebuild server global trainable state from client uploads (eval-only / sanity)."""
    agg = args.agg_type
    if agg == "normal":
        return aggregate_models_normal(global_model, client_states_for_agg)
    if agg == "ffa":
        return aggregate_models_ffa(global_model, client_states_for_agg)
    if is_yoco_agg(agg):
        return aggregate_models_yoco(global_model, client_states_for_agg, args)
    if is_flora_agg(agg):
        return aggregate_models_flora(global_model, client_states_for_agg, args)
    if is_flexlora_agg(agg):
        return aggregate_models_flexlora(global_model, client_states_for_agg, args)
    if is_feddat_agg(agg):
        return aggregate_models_feddat(global_model, client_states_for_agg, args)
    raise ValueError(f"No memory-global re-aggregate handler for agg_type={agg!r}")


def _sft_eval_phase(
    global_model,
    client_ids,
    client_store,
    client_states_for_agg,
    benchmark,
    tokenizer,
    args,
    round_idx,
    bests,
):
    """
    Domain-macro eval (+ optional personalization). Mutates bests dict in place.
    Returns round_payload for metrics_history['rounds'].
    """
    pfl_block = {}
    if is_lora_a_disk_agg(args.agg_type) or is_fedalt_sequential_agg(args.agg_type):
        (
            domain_metrics,
            domain_macro,
            worst_domain,
            domain_macro_token_accuracy,
            worst_domain_token_accuracy,
            domain_macro_perplexity,
            worst_domain_perplexity,
        ) = _evaluate_domain_macro_sequential(
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
        eval_client_ids = list(range(len(client_states_for_agg)))
        if is_memory_global_agg_agg(args.agg_type) and not _memory_agg_eval_use_local_clients(
            args
        ):
            global_eval = extract_trainable_state_dict(global_model)
            eval_store = {
                "mode": "memory",
                "local_states": {idx: global_eval for idx in eval_client_ids},
            }
            print(
                f"[eval] {args.agg_type}: server-aggregated global LoRA for all clients",
                flush=True,
            )
        else:
            eval_store = {
                "mode": "memory",
                "local_states": {
                    idx: client_states_for_agg[idx] for idx in eval_client_ids
                },
            }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        by_domain = group_rows_by_domain(benchmark["test_domain"])
        eval_cap = int(getattr(args, "eval_max_batches", 0) or 0)
        domains_sorted = sorted(by_domain.items())
        eval_mode = (
            "aggregated global LoRA"
            if is_memory_global_agg_agg(args.agg_type)
            and not _memory_agg_eval_use_local_clients(args)
            else "per-client local trainable snapshots"
        )
        print(
            f"[eval] memory-agg ({eval_mode}): {len(domains_sorted)} domains × "
            f"{len(eval_client_ids)} clients; eval_max_batches={eval_cap or 'all'}",
            flush=True,
        )
        metrics = {}
        for domain, rows in domains_sorted:
            dl = create_domain_eval_dataloader(rows, tokenizer, args)
            stats_per_client = []
            for idx in tqdm(
                eval_client_ids,
                desc=f"eval {domain}",
                leave=False,
                dynamic_ncols=True,
            ):
                state = _get_client_local_state(eval_store, idx)
                load_partial_state_dict(global_model, state)
                stats_per_client.append(
                    compute_lm_eval_stats(
                        global_model, dl, device, max_batches=eval_cap
                    )
                )
            metrics[domain] = {
                "loss": float(np.mean([s["loss"] for s in stats_per_client])),
                "token_accuracy": float(
                    np.mean([s["token_accuracy"] for s in stats_per_client])
                ),
                "perplexity": float(
                    np.mean([s["perplexity"] for s in stats_per_client])
                ),
            }
        domain_metrics = metrics
        losses = [v["loss"] for v in metrics.values()]
        accs = [v["token_accuracy"] for v in metrics.values()]
        domain_macro = float(np.mean(losses)) if losses else float("nan")
        worst_domain = float(max(losses)) if losses else float("nan")
        domain_macro_token_accuracy = (
            float(np.mean(accs)) if accs else float("nan")
        )
        worst_domain_token_accuracy = float(min(accs)) if accs else float("nan")
        ppls = [v["perplexity"] for v in metrics.values()]
        domain_macro_perplexity = (
            float(np.mean(ppls)) if ppls else float("nan")
        )
        worst_domain_perplexity = float(max(ppls)) if ppls else float("nan")
        if getattr(args, "eval_personalization_metrics", False):
            pfl_block = _evaluate_personalization_metrics_full_state(
                global_model,
                eval_store,
                eval_client_ids,
                benchmark,
                tokenizer,
                args,
            )

    bests["best_domain_macro"] = min(bests["best_domain_macro"], domain_macro)
    bests["best_worst_domain"] = min(bests["best_worst_domain"], worst_domain)
    if not math.isnan(domain_macro_token_accuracy):
        bests["best_domain_macro_token_accuracy"] = max(
            bests["best_domain_macro_token_accuracy"], domain_macro_token_accuracy
        )
    if not math.isnan(worst_domain_token_accuracy):
        bests["best_worst_domain_token_accuracy"] = max(
            bests["best_worst_domain_token_accuracy"], worst_domain_token_accuracy
        )
    if not math.isnan(domain_macro_perplexity):
        bests["best_domain_macro_perplexity"] = min(
            bests["best_domain_macro_perplexity"], domain_macro_perplexity
        )
    if not math.isnan(worst_domain_perplexity):
        bests["best_worst_domain_perplexity"] = min(
            bests["best_worst_domain_perplexity"], worst_domain_perplexity
        )

    loss_parts = []
    acc_parts = []
    ppl_parts = []
    for d, v in domain_metrics.items():
        if isinstance(v, dict):
            loss_parts.append(f"{d}_loss={v['loss']:.4f}")
            acc_parts.append(f"{d}_tok_acc={v['token_accuracy']:.4f}")
            ppl_parts.append(f"{d}_ppl={v['perplexity']:.2f}")
        else:
            loss_parts.append(f"{d}_loss={float(v):.4f}")
    metrics_str = " | ".join(loss_parts)
    metrics_acc_str = " | ".join(acc_parts) if acc_parts else ""
    metrics_ppl_str = " | ".join(ppl_parts) if ppl_parts else ""
    print(
        f"[eval] round={round_idx + 1} "
        f"primary_macro_tok_acc={domain_macro_token_accuracy:.4f} "
        f"best_macro_tok_acc={bests['best_domain_macro_token_accuracy']:.4f} "
        f"primary_macro_ppl={domain_macro_perplexity:.2f} "
        f"best_macro_ppl={bests['best_domain_macro_perplexity']:.2f} | "
        f"worst_domain_tok_acc={worst_domain_token_accuracy:.4f} "
        f"best_worst_domain_tok_acc={bests['best_worst_domain_token_accuracy']:.4f} "
        f"worst_domain_ppl={worst_domain_perplexity:.2f} "
        f"best_worst_domain_ppl={bests['best_worst_domain_perplexity']:.2f} | "
        f"aux_macro_loss={domain_macro:.4f} "
        f"best_aux_macro_loss={bests['best_domain_macro']:.4f} "
        f"aux_worst_domain_loss={worst_domain:.4f} "
        f"best_aux_worst_domain_loss={bests['best_worst_domain']:.4f} | "
        f"{metrics_str}"
    )
    if metrics_acc_str:
        print(f"[eval] per-domain token_accuracy: {metrics_acc_str}", flush=True)
    if metrics_ppl_str:
        print(f"[eval] per-domain perplexity: {metrics_ppl_str}", flush=True)
    if pfl_block:
        print(
            f"[eval] personalization round={round_idx + 1} "
            f"primary local tok_acc={pfl_block['client_local_macro_token_accuracy']:.4f} "
            f"local_ppl={pfl_block['client_local_macro_perplexity']:.2f} | "
            f"off tok_acc={pfl_block['off_domain_macro_token_accuracy']:.4f} "
            f"off_ppl={pfl_block['off_domain_macro_perplexity']:.2f} | "
            f"in_dom_test tok_acc={pfl_block['in_domain_domain_test_macro_token_accuracy']:.4f} "
            f"in_dom_test ppl={pfl_block['in_domain_domain_test_macro_perplexity']:.2f} | "
            f"gap_tok_acc(local-off)={pfl_block['personalization_gap_token_accuracy']:.4f} "
            f"gap_ppl(off-local)={pfl_block['personalization_gap_perplexity']:.2f} | "
            f"aux local_loss={pfl_block['client_local_macro_loss']:.4f} "
            f"aux off_loss={pfl_block['off_domain_macro_loss']:.4f} "
            f"gap_loss(off-local)={pfl_block['personalization_gap_loss']:.4f}",
            flush=True,
        )

    round_payload = {
        "round": round_idx + 1,
        "domain_macro_token_accuracy": domain_macro_token_accuracy,
        "best_domain_macro_token_accuracy": bests["best_domain_macro_token_accuracy"],
        "worst_domain_token_accuracy": worst_domain_token_accuracy,
        "best_worst_domain_token_accuracy": bests["best_worst_domain_token_accuracy"],
        "domain_macro_perplexity": domain_macro_perplexity,
        "best_domain_macro_perplexity": bests["best_domain_macro_perplexity"],
        "worst_domain_perplexity": worst_domain_perplexity,
        "best_worst_domain_perplexity": bests["best_worst_domain_perplexity"],
        "domain_macro_loss": domain_macro,
        "best_domain_macro_loss": bests["best_domain_macro"],
        "worst_domain_loss": worst_domain,
        "best_worst_domain_loss": bests["best_worst_domain"],
        "domain_metrics": domain_metrics,
    }
    if is_fedplora_oneshot_agg(args.agg_type):
        round_payload["fedplora_oneshot_conflict"] = getattr(
            args, "_fedplora_oneshot_conflict_stats", {}
        ).get("_summary", {})
    if is_fedplora_v3_agg(args.agg_type):
        round_payload["fedplora_v3_stats"] = getattr(
            args, "_fedplora_v3_stats", {}
        ).get("_summary", {})
        round_payload["fedplora_v3_client_clusters"] = getattr(
            args, "_fedplora_v3_client_clusters", {}
        )
    round_payload.update(pfl_block)
    return round_payload


def _metrics_path_from_checkpoint_eval(args, split_dir, ckpt_dir):
    split_tag = os.path.basename(os.path.normpath(split_dir))
    model_tag = os.path.basename(os.path.normpath(args.model.rstrip("/")))
    ckpt_tag = os.path.basename(os.path.normpath(ckpt_dir)).replace(" ", "_")
    fname = (
        f"{args.agg_type}_{model_tag}_{split_tag}_eval_ckpt_{ckpt_tag}_"
        f"r{args.rounds}_e{args.local_epochs}_seed{args.seed}.json"
    )
    os.makedirs(args.metrics_output_dir, exist_ok=True)
    return os.path.join(args.metrics_output_dir, fname)


def _write_checkpoint_ok_file(root, checkpoint_phase, round_saved_1based=None):
    payload = {
        "ok": True,
        "checkpoint_phase": str(checkpoint_phase),
        "saved_at_unix": int(time.time()),
    }
    if round_saved_1based is not None:
        payload["round_saved_1based"] = int(round_saved_1based)
    with open(os.path.join(root, "checkpoint_ok.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _checkpoint_failure_cleanup(root: str, exc: BaseException, checkpoint_phase: str) -> str:
    failed_root = root + "_failed"
    if os.path.isdir(root):
        if os.path.isdir(failed_root):
            shutil.rmtree(failed_root, ignore_errors=True)
        try:
            shutil.move(root, failed_root)
        except OSError:
            os.makedirs(failed_root, exist_ok=True)
    else:
        os.makedirs(failed_root, exist_ok=True)
    with open(
        os.path.join(failed_root, "checkpoint_failed.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "checkpoint_phase": checkpoint_phase,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return failed_root


def _save_run_checkpoint(
    global_model,
    client_store,
    client_ids,
    client_states_for_agg,
    args,
    split_dir,
    metrics_path,
    *,
    bundle_subdir="",
    checkpoint_phase="final",
    round_saved_1based=None,
):
    """
    bundle_subdir: e.g. "snapshots/round_001_post_agg" under save_run_checkpoint_dir for mid-run recovery.
    checkpoint_phase: "post_aggregation" | "final" (recorded in run_checkpoint_meta.json).

    Writes checkpoint_ok.json on success. On failure, moves partial tree to a sibling directory whose
    name ends with "_failed" and writes checkpoint_failed.json (post_aggregation errors are swallowed).
    """
    base = os.path.abspath(os.path.expanduser(args.save_run_checkpoint_dir))
    root = os.path.join(base, bundle_subdir) if bundle_subdir else base
    if checkpoint_phase == "post_aggregation" and os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    failed_adjacent = root + "_failed"
    if checkpoint_phase == "post_aggregation" and os.path.isdir(failed_adjacent):
        shutil.rmtree(failed_adjacent, ignore_errors=True)

    bundle_stem = run_bundle_stem(
        args.agg_type,
        args.model,
        split_dir,
        int(args.rounds),
        int(args.local_epochs),
        int(args.seed),
    )

    try:
        os.makedirs(root, exist_ok=True)
        clients_dir = os.path.join(root, "clients")
        os.makedirs(clients_dir, exist_ok=True)
        meta = {
            "run_checkpoint_version": RUN_CHECKPOINT_VERSION,
            "checkpoint_phase": str(checkpoint_phase),
            "saved_after_aggregation_before_eval": checkpoint_phase == "post_aggregation",
            "bundle_stem": bundle_stem,
            "agg_type": args.agg_type,
            "model": os.path.abspath(os.path.expanduser(args.model)),
            "benchmark_dir": os.path.abspath(os.path.expanduser(split_dir)),
            "seed": int(args.seed),
            "num_clients": int(args.num_clients),
            "client_ids": [int(x) for x in client_ids],
            "disk_sequential_protocol": bool(
                is_lora_a_disk_agg(args.agg_type) or is_fedalt_sequential_agg(args.agg_type)
            ),
            "use_ffa_peft": args.agg_type == "ffa",
            "train_rounds": int(args.rounds),
            "train_local_epochs": int(args.local_epochs),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "rslora": bool(getattr(args, "rslora", False)),
            "target_modules": str(args.target_modules),
            "torch_dtype": str(args.torch_dtype),
            "trust_remote_code": bool(args.trust_remote_code),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "metrics_path": os.path.abspath(metrics_path) if metrics_path else "",
        }
        if round_saved_1based is not None:
            meta["round_saved_1based"] = int(round_saved_1based)
        if not (
            is_lora_a_disk_agg(args.agg_type) or is_fedalt_sequential_agg(args.agg_type)
        ):
            meta["memory_agg_client_payload"] = "trainable_only"
        if is_fedplora_oneshot_agg(args.agg_type):
            summ = getattr(args, "_fedplora_oneshot_conflict_stats", {}).get("_summary", {})
            if summ:
                meta["fedplora_oneshot_conflict_summary"] = summ
        if is_lora_a_disk_agg(args.agg_type) or is_fedalt_sequential_agg(args.agg_type):
            shared_names = get_fedplora_shared_param_names(global_model)
            sd = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
                if k in shared_names
            }
            torch.save(sd, os.path.join(root, "global_shared.pt"))
            if client_store["mode"] == "disk":
                src_dir = client_store["state_dir"]
                for cid in client_ids:
                    src = _client_state_path(src_dir, cid)
                    if not os.path.isfile(src):
                        raise FileNotFoundError(
                            f"[checkpoint] missing client state {src}; ensure --save_client_state_to_disk "
                            "for disk-protocol methods."
                        )
                    dst = os.path.join(clients_dir, f"client_{int(cid):03d}.pt")
                    shutil.copy2(src, dst)
            else:
                for cid in client_ids:
                    st = client_store["local_states"][int(cid)]
                    torch.save(st, os.path.join(clients_dir, f"client_{int(cid):03d}.pt"))
        else:
            if not client_states_for_agg:
                raise RuntimeError(
                    "[checkpoint] memory-agg requires non-empty client_states_for_agg at end of training."
                )
            torch.save(client_states_for_agg, os.path.join(root, "full_clients.pt"))
        meta_path = os.path.join(root, "run_checkpoint_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        _write_checkpoint_ok_file(root, checkpoint_phase, round_saved_1based)
        tag = "post-aggregation snapshot" if checkpoint_phase == "post_aggregation" else "final bundle"
        print(f"[checkpoint] {tag} -> {root}", flush=True)
    except Exception as e:
        fr = _checkpoint_failure_cleanup(root, e, checkpoint_phase)
        print(f"[checkpoint][error] save failed -> {fr}", flush=True)
        if checkpoint_phase == "post_aggregation":
            print(
                f"[checkpoint][warn] post-aggregation snapshot failed ({e!r}); continuing to eval.",
                flush=True,
            )
            return
        raise


def eval_only_from_checkpoint(args):
    ckpt = os.path.abspath(os.path.expanduser(args.eval_only_from_checkpoint))
    meta_path = os.path.join(ckpt, "run_checkpoint_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"missing {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if int(meta.get("run_checkpoint_version", 0)) != RUN_CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported run_checkpoint_version={meta.get('run_checkpoint_version')!r} "
            f"(expected {RUN_CHECKPOINT_VERSION})"
        )
    split_dir = args.benchmark_dir or meta.get("benchmark_dir") or ""
    if not split_dir:
        raise ValueError("provide --benchmark_dir or ensure meta['benchmark_dir'] exists")
    split_dir = os.path.abspath(os.path.expanduser(split_dir))
    benchmark = load_domain_sft_benchmark(split_dir)
    print(f"[benchmark] loaded from {split_dir} (eval-only)", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=getattr(args, "tokenizer_use_fast", True),
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    by_c = group_rows_by_client(benchmark["train"])
    client_ids = sorted(by_c.keys())
    args.num_clients = len(client_ids)
    _relocate_legacy_artifact_dirs(args, args.num_clients)
    print(
        f"[setup] eval-only client_state_dir={args.client_state_dir} "
        f"metrics_output_dir={args.metrics_output_dir}",
        flush=True,
    )

    if meta.get("model") and os.path.normpath(meta["model"]) != os.path.normpath(
        os.path.abspath(os.path.expanduser(args.model))
    ):
        print(
            f"[warn] --model ({args.model}) differs from checkpoint meta model ({meta.get('model')}); "
            "continuing with CLI --model.",
            flush=True,
        )

    if meta["agg_type"] != args.agg_type:
        print(
            f"[warn] --agg_type ({args.agg_type}) differs from checkpoint ({meta['agg_type']}); "
            "using checkpoint agg_type for loading.",
            flush=True,
        )
        args.agg_type = meta["agg_type"]

    args.rounds = int(meta.get("train_rounds", args.rounds))
    args.local_epochs = int(meta.get("train_local_epochs", args.local_epochs))
    args.lora_r = int(meta.get("lora_r", args.lora_r))
    args.lora_alpha = int(meta.get("lora_alpha", args.lora_alpha))
    args.lora_dropout = float(meta.get("lora_dropout", args.lora_dropout))
    if "rslora" in meta:
        args.rslora = bool(meta["rslora"])
    args.target_modules = meta.get("target_modules", args.target_modules)
    args.torch_dtype = meta.get("torch_dtype", args.torch_dtype)

    if meta.get("use_ffa_peft"):
        global_model = create_peft_causal_lm_ffa_model(args)
    else:
        global_model = create_peft_causal_lm_model(args)

    if is_lora_a_disk_agg(args.agg_type) or is_fedalt_sequential_agg(args.agg_type):
        init_fedplora_adapters(global_model)
        shared_path = os.path.join(ckpt, "global_shared.pt")
        if not os.path.isfile(shared_path):
            raise FileNotFoundError(f"missing {shared_path}")
        shared_sd = torch.load(shared_path, map_location="cpu")
        broadcast_fedplora_shared_state(global_model, shared_sd)
        clients_dir = os.path.join(ckpt, "clients")
        client_store = {"mode": "disk", "state_dir": clients_dir}
        _disk_assert_all_client_states(client_store, client_ids, context="eval-only load")
        client_states_for_agg = None
    else:
        fc_path = os.path.join(ckpt, "full_clients.pt")
        if not os.path.isfile(fc_path):
            raise FileNotFoundError(f"missing {fc_path}")
        client_states_for_agg = torch.load(fc_path, map_location="cpu")
        if len(client_states_for_agg) != len(client_ids):
            print(
                f"[warn] checkpoint has {len(client_states_for_agg)} client tensors "
                f"but benchmark has {len(client_ids)} clients.",
                flush=True,
            )
        scratch = os.path.join(ckpt, "_eval_materialized")
        os.makedirs(scratch, exist_ok=True)
        for idx, st in enumerate(client_states_for_agg):
            torch.save(st, _client_state_path(scratch, idx))
        client_store = {"mode": "disk", "state_dir": scratch}
        if is_memory_global_agg_agg(args.agg_type):
            args._aggregate_client_sizes = [
                len(by_c.get(int(cid), [])) for cid in client_ids
            ]
            extra = ""
            if is_yoco_agg(args.agg_type):
                extra = (
                    f" yoco_aggregate_mode="
                    f"{getattr(args, 'yoco_aggregate_mode', 'conflict')}"
                )
            print(
                f"[eval-only] {args.agg_type}: re-aggregating "
                f"{len(client_states_for_agg)} client uploads{extra}",
                flush=True,
            )
            global_model = _reaggregate_memory_global_model(
                global_model, client_states_for_agg, args
            )

    comm_info = estimate_round_communication_bytes(
        global_model.state_dict(),
        args.agg_type,
        trainable_param_names=get_trainable_param_names(global_model),
    )
    metrics_history = {
        "args": vars(args).copy(),
        "benchmark_dir": split_dir,
        "eval_only_from_checkpoint": ckpt,
        "checkpoint_meta": meta,
        "recommended_primary_metrics": [
            "domain_macro_token_accuracy",
            "domain_macro_perplexity",
            "worst_domain_token_accuracy",
            "worst_domain_perplexity",
        ],
        "communication": {
            "agg_type": args.agg_type,
            "down_bytes_per_client": int(comm_info["down_bytes_per_client"]),
            "up_bytes_per_client": int(comm_info["up_bytes_per_client"]),
        },
        "rounds": [],
    }
    metrics_history["recommended_kpis"] = _metrics_recommended_kpis(
        getattr(args, "eval_personalization_metrics", False)
    )
    if getattr(args, "eval_personalization_metrics", False):
        metrics_history["recommended_primary_personalization_metrics"] = [
            "client_local_macro_token_accuracy",
            "client_local_macro_perplexity",
            "off_domain_macro_token_accuracy",
            "off_domain_macro_perplexity",
            "personalization_gap_token_accuracy",
            "personalization_gap_perplexity",
        ]

    bests = {
        "best_domain_macro": float("inf"),
        "best_worst_domain": float("inf"),
        "best_domain_macro_token_accuracy": float("-inf"),
        "best_worst_domain_token_accuracy": float("-inf"),
        "best_domain_macro_perplexity": float("inf"),
        "best_worst_domain_perplexity": float("inf"),
    }
    round_payload = _sft_eval_phase(
        global_model,
        client_ids,
        client_store,
        client_states_for_agg,
        benchmark,
        tokenizer,
        args,
        0,
        bests,
    )
    round_payload["eval_note"] = "eval_only_from_checkpoint (no training)"
    summ = meta.get("fedplora_oneshot_conflict_summary")
    if summ:
        round_payload["fedplora_oneshot_conflict"] = summ
    elif is_fedplora_oneshot_agg(args.agg_type):
        round_payload["fedplora_oneshot_conflict"] = {}
    metrics_history["rounds"].append(round_payload)

    out_path = _metrics_path_from_checkpoint_eval(args, split_dir, ckpt)
    _write_metrics_file(out_path, metrics_history)
    print(f"[metrics] eval-only saved to {out_path}", flush=True)


def federated_sft(args):
    if int(args.rounds) != 1:
        print(
            f"[setup] one-round comparison: forcing --rounds 1 (CLI had {args.rounds})",
            flush=True,
        )
        args.rounds = 1

    benchmark, split_dir = build_or_load_benchmark(args)
    print(f"[benchmark] loaded from {split_dir}")
    print(f"[benchmark] domains={sorted(benchmark['domain_stats'].keys())}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=getattr(args, "tokenizer_use_fast", True),
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
    args._fedplora_client_domains = _client_id_to_home_domain(benchmark["clients"])
    args._fedplora_round_client_ids = list(client_ids)
    _relocate_legacy_artifact_dirs(args, args.num_clients)
    _maybe_apply_default_save_run_checkpoint_dir(args, split_dir)
    if _try_skip_if_run_fully_complete(args, split_dir, client_ids):
        return
    if _try_resume_eval_only_from_latest_post_agg_snapshot(args, split_dir, client_ids):
        return
    nw = int(getattr(args, "dataloader_num_workers", 0) or 0)
    enw = int(getattr(args, "eval_dataloader_num_workers", 0) or 0)
    tcap = int(getattr(args, "train_max_steps_per_client", 0) or 0)
    scap = int(getattr(args, "max_train_samples_per_client", 0) or 0)
    print(
        f"[setup] client_state_dir={args.client_state_dir} "
        f"metrics_output_dir={args.metrics_output_dir}",
        flush=True,
    )
    if nw > 0 or enw > 0 or tcap > 0 or scap > 0:
        print(
            f"[setup] speed: dataloader_num_workers={nw} "
            f"eval_dataloader_num_workers={enw} "
            f"train_max_steps_per_client={tcap or 'off'} "
            f"max_train_samples_per_client={scap or 'off'}",
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
        initial_A_for_oneshot = {
            k: v.detach().cpu().clone()
            for k, v in global_model.state_dict().items()
            if is_lora_a_param_name(k)
        }
    elif is_fedalt_sequential_agg(args.agg_type):
        init_fedplora_adapters(global_model)
        client_store = _ensure_sequential_fedalt_states(
            global_model, client_ids, args
        )
        initial_A_for_oneshot = {}
    else:
        client_store = None
        initial_A_for_oneshot = {}

    bests = {
        "best_domain_macro": float("inf"),
        "best_worst_domain": float("inf"),
        "best_domain_macro_token_accuracy": float("-inf"),
        "best_worst_domain_token_accuracy": float("-inf"),
        "best_domain_macro_perplexity": float("inf"),
        "best_worst_domain_perplexity": float("inf"),
    }
    metrics_history = {
        "args": vars(args).copy(),
        "benchmark_dir": split_dir,
        "recommended_primary_metrics": [
            "domain_macro_token_accuracy",
            "domain_macro_perplexity",
            "worst_domain_token_accuracy",
            "worst_domain_perplexity",
        ],
        "communication": {
            "agg_type": args.agg_type,
            "down_bytes_per_client": int(comm_info["down_bytes_per_client"]),
            "up_bytes_per_client": int(comm_info["up_bytes_per_client"]),
        },
        "rounds": [],
    }
    metrics_history["recommended_kpis"] = _metrics_recommended_kpis(
        getattr(args, "eval_personalization_metrics", False)
    )

    if getattr(args, "eval_personalization_metrics", False):
        metrics_history["recommended_primary_personalization_metrics"] = [
            "client_local_macro_token_accuracy",
            "client_local_macro_perplexity",
            "off_domain_macro_token_accuracy",
            "off_domain_macro_perplexity",
            "personalization_gap_token_accuracy",
            "personalization_gap_perplexity",
        ]

    if is_fedplora_oneshot_family_agg(args.agg_type):
        args._fedplora_initial_A = initial_A_for_oneshot

    client_states_for_agg = []
    for round_idx in range(args.rounds):
        print(f"Round {round_idx + 1}/{args.rounds}")

        if (
            (is_lora_a_disk_agg(args.agg_type) or is_fedalt_sequential_agg(args.agg_type))
            and getattr(args, "save_client_state_to_disk", False)
            and round_idx > 0
        ):
            _disk_assert_all_client_states(
                client_store,
                client_ids,
                context=f"start of round {round_idx + 1}",
            )

        gp_global_state = None
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
            # Trainable LoRA + heads only (not full backbone clone).
            round_global_state = extract_round_broadcast_state(global_model, args.agg_type)
            if is_yoco_agg(args.agg_type):
                args._yoco_round_start_trainable = {
                    k: v.detach().cpu().clone() for k, v in round_global_state.items()
                }

        client_states_for_agg = []
        fedplora_uploads = []
        fedalt_uploads = []
        for i, client_id in enumerate(client_ids):
            args._tqdm_desc = f"R{round_idx + 1}/{args.rounds} client{i + 1}/{args.num_clients}"
            if is_lora_a_disk_agg(args.agg_type):
                broadcast_fedplora_shared_state(global_model, gp_global_state)
                local_state = _get_client_local_state(client_store, client_id)
                load_fedplora_local_state(global_model, local_state)
            elif is_fedalt_sequential_agg(args.agg_type):
                local_state = _get_client_local_state(client_store, client_id)
                load_fedalt_local_state(global_model, local_state)
            else:
                load_partial_state_dict(global_model, round_global_state)
                if is_feddat_agg(args.agg_type):
                    args._feddat_teacher_state = round_global_state
            train_client(global_model, client_dataloaders[i], args, client_idx=i)
            if nw > 0:
                shutdown_dataloader_workers(client_dataloaders[i])
            if is_lora_a_disk_agg(args.agg_type):
                cid = int(client_ids[i])
                dom = (getattr(args, "_fedplora_client_domains", {}) or {}).get(cid, "unknown")
                fedplora_uploads.append(
                    build_fedplora_upload_package(
                        global_model,
                        client_sizes[i],
                        client_id=cid,
                        domain=dom,
                    )
                )
            elif is_fedalt_sequential_agg(args.agg_type):
                fedalt_uploads.append(
                    build_fedalt_upload_package(global_model, client_sizes[i])
                )
            else:
                # Trainable-only (LoRA + heads): aggregators and eval only touch these keys;
                # cloning full state_dict() per client duplicates frozen base N× and commonly OOMs Linux RAM.
                client_states_for_agg.append(extract_trainable_state_dict(global_model))
            if is_lora_a_disk_agg(args.agg_type):
                updated_local_state = extract_fedplora_local_state(global_model)
                _set_client_local_state(client_store, client_id, updated_local_state)
            elif is_fedalt_sequential_agg(args.agg_type):
                updated_local_state = extract_fedalt_local_state(global_model)
                _set_client_local_state(client_store, client_id, updated_local_state)

        args._tqdm_desc = None

        if (
            (is_lora_a_disk_agg(args.agg_type) or is_fedalt_sequential_agg(args.agg_type))
            and getattr(args, "save_client_state_to_disk", False)
        ):
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
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_states_for_agg)
        elif is_fedplora_multiround_agg(args.agg_type):
            global_model = aggregate_models_fedplora(global_model, fedplora_uploads, args)
        elif is_fedplora_oneshot_agg(args.agg_type):
            global_model = aggregate_models_fedplora_oneshot(global_model, fedplora_uploads, args)
            stats = getattr(args, "_fedplora_oneshot_conflict_stats", {}).get("_summary", {})
            if stats:
                print(
                    f"[fedplora-oneshot] conflict mean={stats.get('mean_conflict', float('nan')):.4f} "
                    f"max={stats.get('max_conflict', float('nan')):.4f} "
                    f"high_row_frac={stats.get('high_conflict_row_frac', float('nan')):.4f} "
                    f"init_gate={stats.get('mean_init_gate', float('nan')):.4f}",
                    flush=True,
                )
        elif is_fedplora_v3_agg(args.agg_type):
            norm = (args.agg_type or "").strip().lower().replace("-", "_")
            if norm in {"fedplora_v3_lite", "v3_lite"}:
                global_model = aggregate_models_fedplora_v3_lite(
                    global_model, fedplora_uploads, args
                )
            elif norm in {"fedplora_v3_cluster", "v3_cluster"}:
                global_model = aggregate_models_fedplora_v3_cluster(
                    global_model, fedplora_uploads, args
                )
            elif norm in {"fedplora_v3_rpca", "v3_rpca"}:
                global_model = aggregate_models_fedplora_v3_rpca(
                    global_model, fedplora_uploads, args
                )
            else:
                raise ValueError(f"Unknown v3 agg_type: {args.agg_type}")
            v3_summ = getattr(args, "_fedplora_v3_stats", {}).get("_summary", {})
            if v3_summ:
                print(
                    f"[fedplora-v3] variant={v3_summ.get('variant', norm)} "
                    f"mean_conflict={v3_summ.get('mean_conflict', float('nan')):.4f} "
                    f"high_row_frac={v3_summ.get('high_conflict_row_frac', float('nan')):.4f} "
                    f"num_clusters={v3_summ.get('num_clusters', 0)}",
                    flush=True,
                )
        elif is_yoco_agg(args.agg_type):
            args._aggregate_client_sizes = client_sizes
            global_model = aggregate_models_yoco(global_model, client_states_for_agg, args)
        elif is_fedalt_agg(args.agg_type):
            global_model = aggregate_models_fedalt(global_model, fedalt_uploads, args)
            rotw_list = getattr(args, "_fedalt_rotw_by_client", [])
            for i, client_id in enumerate(client_ids):
                if i < len(rotw_list):
                    _set_client_rotw_state(client_store, client_id, rotw_list[i])
        elif is_fedsa_lora_agg(args.agg_type):
            global_model = aggregate_models_fedsa_lora(global_model, fedplora_uploads, args)
        elif is_flora_agg(args.agg_type):
            global_model = aggregate_models_flora(global_model, client_states_for_agg, args)
        elif is_flexlora_agg(args.agg_type):
            args._aggregate_client_sizes = client_sizes
            global_model = aggregate_models_flexlora(
                global_model, client_states_for_agg, args
            )
        elif is_feddat_agg(args.agg_type):
            args._aggregate_client_sizes = client_sizes
            global_model = aggregate_models_feddat(
                global_model, client_states_for_agg, args
            )
        else:
            raise ValueError(f"Unknown agg_type: {args.agg_type}")

        print(f"[round {round_idx + 1}] aggregation done; running evaluation ...", flush=True)
        save_ckpt = getattr(args, "save_run_checkpoint_dir", None) or ""
        if (
            str(save_ckpt).strip()
            and not getattr(args, "skip_post_agg_snapshots", False)
        ):
            snap_rel = os.path.join("snapshots", f"round_{round_idx + 1:03d}_post_agg")
            _save_run_checkpoint(
                global_model,
                client_store,
                client_ids,
                client_states_for_agg,
                args,
                split_dir,
                "",
                bundle_subdir=snap_rel,
                checkpoint_phase="post_aggregation",
                round_saved_1based=round_idx + 1,
            )
        round_payload = _sft_eval_phase(
            global_model,
            client_ids,
            client_store,
            client_states_for_agg,
            benchmark,
            tokenizer,
            args,
            round_idx,
            bests,
        )
        metrics_history["rounds"].append(round_payload)

    metrics_path = _metrics_path(args, split_dir)
    _write_metrics_file(metrics_path, metrics_history)
    print(f"[metrics] saved to {metrics_path}")

    save_ckpt = getattr(args, "save_run_checkpoint_dir", None) or ""
    if str(save_ckpt).strip():
        _save_run_checkpoint(
            global_model,
            client_store,
            client_ids,
            client_states_for_agg,
            args,
            split_dir,
            metrics_path,
            checkpoint_phase="final",
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    set_seed(args.seed)
    log_file, orig_out, orig_err, _ = setup_run_logging(args, filename_prefix="sft")
    try:
        if getattr(args, "eval_only_from_checkpoint", None):
            if args.build_benchmark:
                raise ValueError("--eval_only_from_checkpoint cannot be combined with --build_benchmark")
            eval_only_from_checkpoint(args)
        else:
            federated_sft(args)
    finally:
        restore_logging(log_file, orig_out, orig_err)
