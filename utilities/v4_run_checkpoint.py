"""Run checkpoint save/resume for tasks/fed_train_sft_v4.py (防白训).

Mirrors v2 fed_train_sft.py:
  - post-aggregation snapshot under snapshots/round_XXX_post_agg/  → eval-only resume
  - final bundle at save_run_checkpoint_dir root with checkpoint_phase=final → skip all
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from pathlib import Path

import torch

from utilities.sft_checkpoint_paths import default_save_run_checkpoint_dir, run_bundle_stem
from utilities.state_dict_ops import extract_fedplora_local_state
from utilities.utils import (
    get_fedplora_shared_param_names,
    is_fedplora_oneshot_agg,
    tensor_to_list,
    uses_fedplora_oneshot_server_agg,
)

RUN_CHECKPOINT_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[1]


def client_state_path(base_dir: str, client_id) -> str:
    return os.path.join(base_dir, f"client_{int(client_id):03d}.pt")


def disk_state_dir(args) -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(args.client_state_dir)), f"seed_{args.seed}")


def save_client_local_state(local_state, base_dir: str, client_id) -> str:
    os.makedirs(base_dir, exist_ok=True)
    path = client_state_path(base_dir, client_id)
    tmp_path = path + ".tmp"
    torch.save(local_state, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_client_local_state(base_dir: str, client_id):
    path = client_state_path(base_dir, client_id)
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu")


def init_v4_client_store(global_model, client_ids, args) -> dict:
    """Initialize per-client local B; optionally persist under client_state_dir/seed_{seed}/."""
    seed_sd = extract_fedplora_local_state(global_model)
    if getattr(args, "save_client_state_to_disk", False):
        state_dir = disk_state_dir(args)
        os.makedirs(state_dir, exist_ok=True)
        local_states = {}
        for client_id in client_ids:
            if not os.path.isfile(client_state_path(state_dir, client_id)):
                save_client_local_state(seed_sd, state_dir, client_id)
            local_states[int(client_id)] = load_client_local_state(state_dir, client_id)
        return {"mode": "disk", "state_dir": state_dir, "local_states": local_states}
    local_states = {
        int(client_id): {k: v.clone() for k, v in seed_sd.items()} for client_id in client_ids
    }
    return {"mode": "memory", "state_dir": disk_state_dir(args), "local_states": local_states}


def persist_client_local_state(client_store: dict, client_id, local_state, args) -> None:
    client_store["local_states"][int(client_id)] = local_state
    if getattr(args, "save_client_state_to_disk", False):
        save_client_local_state(local_state, client_store["state_dir"], client_id)


def _norm_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(p)))


def _load_checkpoint_ok(bundle_dir: str):
    p = os.path.join(bundle_dir, "checkpoint_ok.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


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
    with open(os.path.join(failed_root, "checkpoint_failed.json"), "w", encoding="utf-8") as f:
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


def maybe_apply_default_save_run_checkpoint_dir(args, split_dir: str) -> None:
    if getattr(args, "no_auto_save_run_checkpoint", False):
        return
    if str(getattr(args, "save_run_checkpoint_dir", "") or "").strip():
        return
    tmr = getattr(args, "trained_models_root", None)
    args.save_run_checkpoint_dir = default_save_run_checkpoint_dir(
        _REPO_ROOT,
        (tmr or "").strip() or None,
        agg_type=args.agg_type,
        model_path=args.model,
        benchmark_split_dir=split_dir,
        rounds=int(args.rounds),
        local_epochs=int(args.local_epochs),
        seed=int(args.seed),
    )
    print(f"[v4-checkpoint] auto save_run_checkpoint_dir={args.save_run_checkpoint_dir}", flush=True)


def resume_meta_matches(meta: dict, args, split_dir: str, client_ids) -> bool:
    def _bad(reason: str) -> bool:
        print(f"[v4-resume] checkpoint meta mismatch ({reason}); will retrain.", flush=True)
        return False

    if meta.get("entry_script") != "fed_train_sft_v4":
        return _bad("entry_script")
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
    if meta.get("client_ids") != [int(x) for x in client_ids]:
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


def load_v4_bundle(ckpt_dir: str, global_model, client_ids):
    meta_path = os.path.join(ckpt_dir, "run_checkpoint_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    shared_path = os.path.join(ckpt_dir, "global_shared.pt")
    if not os.path.isfile(shared_path):
        raise FileNotFoundError(f"missing {shared_path}")
    shared_sd = torch.load(shared_path, map_location="cpu")
    from utilities.state_dict_ops import broadcast_fedplora_shared_state

    broadcast_fedplora_shared_state(global_model, shared_sd)
    clients_dir = os.path.join(ckpt_dir, "clients")
    local_states = {}
    for cid in client_ids:
        src = os.path.join(clients_dir, f"client_{int(cid):03d}.pt")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"missing client state {src}")
        local_states[int(cid)] = torch.load(src, map_location="cpu")
    personalized_path = os.path.join(ckpt_dir, "personalized_shared.pt")
    personalized = {}
    if os.path.isfile(personalized_path):
        personalized = torch.load(personalized_path, map_location="cpu")
    return meta, local_states, personalized


def save_v4_run_checkpoint(
    global_model,
    local_states,
    client_ids,
    args,
    split_dir: str,
    *,
    bundle_subdir: str = "",
    checkpoint_phase: str = "final",
    round_saved_1based=None,
    round_metrics=None,
):
    if getattr(args, "skip_post_agg_snapshots", False) and checkpoint_phase == "post_aggregation":
        return
    base = os.path.abspath(os.path.expanduser(args.save_run_checkpoint_dir))
    root = os.path.join(base, bundle_subdir) if bundle_subdir else base
    if checkpoint_phase == "post_aggregation" and os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)

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
        shared_names = get_fedplora_shared_param_names(global_model)
        shared_sd = {
            k: v.detach().cpu().clone()
            for k, v in global_model.state_dict().items()
            if k in shared_names
        }
        torch.save(shared_sd, os.path.join(root, "global_shared.pt"))
        for cid in client_ids:
            st = local_states[int(cid)]
            save_client_local_state(st, clients_dir, cid)
        personalized = getattr(args, "_fedplora_personalized_shared_states", None) or {}
        if personalized:
            torch.save(personalized, os.path.join(root, "personalized_shared.pt"))
        meta = {
            "run_checkpoint_version": RUN_CHECKPOINT_VERSION,
            "entry_script": "fed_train_sft_v4",
            "checkpoint_phase": str(checkpoint_phase),
            "saved_after_aggregation_before_eval": checkpoint_phase == "post_aggregation",
            "bundle_stem": bundle_stem,
            "agg_type": args.agg_type,
            "model": os.path.abspath(os.path.expanduser(args.model)),
            "benchmark_dir": os.path.abspath(os.path.expanduser(split_dir)),
            "seed": int(args.seed),
            "num_clients": int(args.num_clients),
            "client_ids": [int(x) for x in client_ids],
            "disk_sequential_protocol": True,
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
        }
        if round_saved_1based is not None:
            meta["round_saved_1based"] = int(round_saved_1based)
        if uses_fedplora_oneshot_server_agg(args.agg_type) or is_fedplora_oneshot_agg(args.agg_type):
            summ = getattr(args, "_fedplora_oneshot_conflict_stats", {}).get("_summary", {})
            if summ:
                meta["fedplora_oneshot_conflict_summary"] = summ
        v4_summ = getattr(args, "_fedplora_v4_stats", {}).get("_summary", None)
        if v4_summ:
            meta["v4_stats_summary"] = v4_summ
        with open(os.path.join(root, "run_checkpoint_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        if round_metrics is not None and checkpoint_phase == "final":
            with open(os.path.join(root, "metrics_round.json"), "w", encoding="utf-8") as f:
                json.dump(tensor_to_list(round_metrics), f, indent=2, ensure_ascii=False)
        _write_checkpoint_ok_file(root, checkpoint_phase, round_saved_1based)
        tag = "post-aggregation snapshot" if checkpoint_phase == "post_aggregation" else "final bundle"
        print(f"[v4-checkpoint] {tag} -> {root}", flush=True)
    except Exception as e:
        fr = _checkpoint_failure_cleanup(root, e, checkpoint_phase)
        print(f"[v4-checkpoint][error] save failed -> {fr}", flush=True)
        if checkpoint_phase == "post_aggregation":
            print(
                f"[v4-checkpoint][warn] post-aggregation snapshot failed ({e!r}); continuing to eval.",
                flush=True,
            )
            return
        raise


def try_skip_if_run_fully_complete(args, split_dir: str, client_ids):
    if getattr(args, "force_retrain", False):
        return None
    bundle = str(getattr(args, "save_run_checkpoint_dir", "") or "").strip()
    if not bundle:
        return None
    bundle = os.path.abspath(os.path.expanduser(bundle))
    ok = _load_checkpoint_ok(bundle)
    meta_path = os.path.join(bundle, "run_checkpoint_meta.json")
    if not ok or not ok.get("ok") or not os.path.isfile(meta_path):
        return None
    if str(ok.get("checkpoint_phase", "final") or "final") != "final":
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not resume_meta_matches(meta, args, split_dir, client_ids):
        return None
    metrics_path = os.path.join(bundle, "metrics_round.json")
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        round_block = json.load(f)
    print(
        f"[v4-resume] Run fully complete at {bundle}; skipping training and evaluation.",
        flush=True,
    )
    return {
        "args": {k: v for k, v in vars(args).items() if not str(k).startswith("_")},
        "seed": int(args.seed),
        "benchmark_dir": split_dir,
        "rounds": [round_block],
        "communication": {"agg_type": args.agg_type, "resume": "final_checkpoint"},
        "resume_note": "skipped_train_and_eval",
    }


def try_resume_eval_only_from_post_agg(args, split_dir: str, client_ids, evaluate_fn):
    """Load snapshot/final bundle, run evaluate_fn(...), return metrics dict."""
    if getattr(args, "force_retrain", False):
        return None

    explicit = str(getattr(args, "eval_only_from_checkpoint", "") or "").strip()
    if explicit:
        snap_dirs = [os.path.abspath(os.path.expanduser(explicit))]
        allow_phases = {"post_aggregation", "final"}
    else:
        bundle = str(getattr(args, "save_run_checkpoint_dir", "") or "").strip()
        if not bundle:
            return None
        snap_dirs = _list_post_agg_snapshot_dirs(os.path.abspath(os.path.expanduser(bundle)))
        allow_phases = {"post_aggregation"}

    for snap_dir in snap_dirs:
        ok = _load_checkpoint_ok(snap_dir)
        meta_path = os.path.join(snap_dir, "run_checkpoint_meta.json")
        if not ok or not ok.get("ok") or not os.path.isfile(meta_path):
            continue
        phase = str(ok.get("checkpoint_phase", "") or "")
        if phase not in allow_phases:
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not resume_meta_matches(meta, args, split_dir, client_ids):
            continue
        print(
            f"[v4-resume] Loading checkpoint at {snap_dir} (phase={phase}); "
            "skipping training and running evaluation only.",
            flush=True,
        )
        from utilities.models import create_peft_causal_lm_model, init_fedplora_adapters

        global_model = create_peft_causal_lm_model(args)
        init_fedplora_adapters(global_model)
        _, local_states, personalized = load_v4_bundle(snap_dir, global_model, client_ids)
        if personalized:
            args._fedplora_personalized_shared_states = personalized
        round_idx = int(meta.get("round_saved_1based", 1) or 1) - 1
        round_block = evaluate_fn(global_model, client_ids, local_states, round_idx=round_idx)
        round_block["eval_note"] = "eval_only_from_checkpoint"
        if phase == "post_aggregation":
            save_v4_run_checkpoint(
                global_model,
                local_states,
                client_ids,
                args,
                split_dir,
                checkpoint_phase="final",
                round_saved_1based=meta.get("round_saved_1based"),
                round_metrics=round_block,
            )
        return {
            "args": {k: v for k, v in vars(args).items() if not str(k).startswith("_")},
            "seed": int(args.seed),
            "benchmark_dir": split_dir,
            "rounds": [round_block],
            "communication": {"agg_type": args.agg_type, "resume": f"eval_only_{phase}"},
            "resume_note": "eval_only_from_checkpoint",
        }
    return None
