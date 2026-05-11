import torch
import os
from datetime import datetime
import json
import sys


def _norm_agg_type(agg_type):
    return (agg_type or "").strip().lower().replace("-", "_")


def is_fedplora_agg(agg_type):
    """Multi-round FedP-LoRA: sign-aligned A agg + local FedP regularizers."""
    return _norm_agg_type(agg_type) == "fedplora"


def is_fedplora_oneshot_agg(agg_type):
    """
    FedPLoRA-Oneshot: exactly one federated round; clients upload LoRA A,
    trainable heads, and row-importance stats while B stays local. Server uses
    conflict-gated A aggregation against the initial shared A0.
    """
    return _norm_agg_type(agg_type) == "fedplora_oneshot"


def is_lora_a_disk_agg(agg_type):
    """Domain SFT: server holds A (+ heads), clients keep B on disk/memory."""
    return _norm_agg_type(agg_type) in {
        "fedplora",
        "fedplora_oneshot",
        "fedsa_lora",
        "fedsa",
        "yoco",
        "fedalt",
    }


def is_fedplora_multiround_agg(agg_type):
    """Multi-round FedP-LoRA server aggregate + local FedP regularizers (train_eval)."""
    return _norm_agg_type(agg_type) == "fedplora"


def is_fedsa_lora_agg(agg_type):
    return _norm_agg_type(agg_type) in {"fedsa_lora", "fedsa"}


def is_yoco_agg(agg_type):
    return _norm_agg_type(agg_type) == "yoco"


def is_lora_a2_agg(agg_type):
    return _norm_agg_type(agg_type) in {"lora_a2", "loraa2"}


def is_hetlora_agg(agg_type):
    return _norm_agg_type(agg_type) in {"hetlora", "het_lora"}


def is_flora_agg(agg_type):
    return _norm_agg_type(agg_type) == "flora"


def is_fdlora_agg(agg_type):
    return _norm_agg_type(agg_type) in {"fdlora", "fd_lora"}


def is_fedalt_agg(agg_type):
    return _norm_agg_type(agg_type) == "fedalt"


def is_lora_param_name(key):
    return "lora" in key


def is_lora_a_param_name(key):
    return "lora_A" in key and key.endswith("default.weight")


def is_lora_b_param_name(key):
    return "lora_B" in key and key.endswith("default.weight")


def is_task_head_param_name(key):
    return (
        "classifier" in key
        or "lm_head" in key
        or key.endswith(".score.weight")
        or key.endswith(".score.bias")
    )


def is_fedplora_shared_param_name(key, trainable_param_names=None):
    if is_lora_a_param_name(key):
        return True
    if is_task_head_param_name(key):
        return trainable_param_names is None or key in trainable_param_names
    return False


def get_trainable_param_names(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def get_fedplora_shared_param_names(model):
    trainable_param_names = get_trainable_param_names(model)
    return {
        key
        for key in model.state_dict().keys()
        if is_fedplora_shared_param_name(key, trainable_param_names)
    }


def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj


def save_dict_to_json(data_dict, args, base_path):
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"compare_dict_rounds_{timestamp}.json"
    file_path = os.path.join(base_path, filename)

    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)

    # Combine data_dict and args
    combined_dict = {"args": vars(args), "data": data_dict}

    # Convert tensors to lists
    json_serializable_dict = tensor_to_list(combined_dict)

    # Write JSON data to the file
    with open(file_path, "w") as json_file:
        json.dump(json_serializable_dict, json_file, indent=2)

    print(f"Data and args saved to {file_path}")


def _tensor_bytes_comm(t):
    return int(t.numel() * t.element_size())


def _sum_bytes_state_dict(state_dict, key_pred):
    return sum(
        _tensor_bytes_comm(v) for k, v in state_dict.items() if key_pred(k)
    )


def estimate_round_communication_bytes(
    state_dict, agg_type, trainable_param_names=None
):
    """
    Per client, per round:
    - down_bytes: server -> client (parameters overwritten at round start)
    - up_bytes: client -> server (parameters used in aggregation)
    Total link volume per round ~= num_clients * (down_bytes + up_bytes).
    """
    sd = state_dict

    def is_trainable_task_head(k):
        return is_task_head_param_name(k) and (
            trainable_param_names is None or k in trainable_param_names
        )

    def gp_row_importance_bytes():
        total = 0
        for k, v in sd.items():
            if is_lora_a_param_name(k):
                total += int(v.shape[0] * 4)
        return total

    full_model = sum(_tensor_bytes_comm(v) for v in sd.values())

    lora_all = _sum_bytes_state_dict(sd, is_lora_param_name)
    lora_a = _sum_bytes_state_dict(sd, is_lora_a_param_name)
    lora_b = _sum_bytes_state_dict(sd, is_lora_b_param_name)
    task_head = _sum_bytes_state_dict(sd, is_trainable_task_head)
    gp_stats = gp_row_importance_bytes()

    agg_type = _norm_agg_type(agg_type) or "normal"

    if is_fedplora_multiround_agg(agg_type) or is_fedplora_oneshot_agg(agg_type):
        down = lora_a + task_head
        up = lora_a + task_head + gp_stats
    elif is_lora_a_disk_agg(agg_type):
        down = lora_a + task_head
        up = lora_a + task_head
    elif agg_type == "ffa":
        down = full_model
        up = lora_b + task_head
    elif agg_type == "fedex":
        down = full_model
        up = lora_all + task_head
    else:
        down = full_model
        up = lora_all + task_head

    return {"down_bytes_per_client": down, "up_bytes_per_client": up}


class Tee:
    """Write all output to multiple streams (e.g., console + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_run_logging(args, log_dir="log", filename_prefix=None):
    """
    Tee stdout/stderr to a timestamped log file under log_dir.
    Returns (log_file, orig_stdout, orig_stderr, log_path).
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "run"
    agg = getattr(args, "agg_type", "agg")
    # GLUE / E2E: filename encodes label-partition scheme (iid vs dirichlet + alpha).
    # Domain SFT (fed_train_sft): clients are domain-disjoint → task-shift non-IID, not iid_iid.
    if filename_prefix == "sft":
        skew_tag = "task_shift_non-iid"
    else:
        part = getattr(args, "partition", "iid")
        alpha = getattr(args, "dirichlet_alpha", None)
        alpha_s = f"a{alpha}" if (part == "dirichlet" and alpha is not None) else "iid"
        skew_tag = f"{part}_{alpha_s}"
    num_clients = getattr(args, "num_clients", "C")
    rounds = getattr(args, "rounds", "R")
    local_epochs = getattr(args, "local_epochs", "E")
    lr = getattr(args, "lr", "LR")
    seed = getattr(args, "seed", "S")

    prefix = filename_prefix + "_" if filename_prefix else ""
    fname = (
        f"{prefix}{task}_{agg}_{skew_tag}_"
        f"c{num_clients}_r{rounds}_e{local_epochs}_lr{lr}_seed{seed}_{ts}.log"
    )
    path = os.path.join(log_dir, fname)
    f = open(path, "w", encoding="utf-8")

    orig_out, orig_err = sys.__stdout__, sys.__stderr__
    sys.stdout = Tee(orig_out, f)
    sys.stderr = Tee(orig_err, f)
    print(f"[log] writing console output to {path}")
    try:
        print(f"[log] args: {vars(args)}")
    except Exception:
        pass
    return f, orig_out, orig_err, path


def restore_logging(log_file, orig_out, orig_err):
    """Undo setup_run_logging() and close the log file."""
    try:
        sys.stdout = orig_out
        sys.stderr = orig_err
    finally:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass


def print_round_metrics_client_mean(round_idx, rounds, pfl, n1, n2, max_m1, max_m2, tag):
    """Client-mean metrics line (macro mean over clients, plus per-client values)."""
    m1 = pfl["pfl_metric1_macro"]
    m2 = pfl["pfl_metric2_macro"]
    parts = [
        f"Round {round_idx + 1}/{rounds}",
        f"{tag}_mean_{n1}={m1:.8f}",
        f"max_{tag}_mean_{n1}={max_m1:.8f}",
    ]
    if m2 is not None and n2 is not None:
        parts.append(f"{tag}_mean_{n2}={m2:.8f}")
        parts.append(f"max_{tag}_mean_{n2}={max_m2:.8f}")
    pc1 = ", ".join(
        f"c{i}={float(x):.6f}" for i, x in enumerate(pfl["pfl_per_client_metric1"])
    )
    parts.append(f"{tag}_per_client_{n1}: {pc1}")
    if (
        m2 is not None
        and n2 is not None
        and pfl["pfl_per_client_metric2"]
        and all(x is not None for x in pfl["pfl_per_client_metric2"])
    ):
        pc2 = ", ".join(
            f"c{i}={float(x):.6f}" for i, x in enumerate(pfl["pfl_per_client_metric2"])
        )
        parts.append(f"{tag}_per_client_{n2}: {pc2}")
    print(" | ".join(parts))
