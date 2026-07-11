import torch
import os
from datetime import datetime
import json
import sys

import numpy as np


def _norm_agg_type(agg_type):
    return (agg_type or "").strip().lower().replace("-", "_")


FEDPLORA_V8_B_ONLY_AGGS = {
    "fedplora_v8",
    "v8_bsim",
}

FEDPLORA_V8_AB_AGGS = {
    "fedplora_v8_ab",
    "v8_bsim_ab",
}

FEDPLORA_V8_WARMA_AGGS = {
    "fedplora_v8_warma",
    "fedplora_v8_warm_a",
    "fedplora_v8_warma_then_freeze",
    "v8_warma",
    "v8_warm_a",
    "v8_warma_then_freeze",
}

FEDPLORA_V8_PERIODIC_A_AGGS = {
    "fedplora_v8_periodic",
    "fedplora_v8_periodica",
    "fedplora_v8_periodica_t",
    "v8_periodic",
    "v8_periodica",
    "v8_periodica_t",
}

FEDPLORA_V8_FAMILY_AGGS = (
    FEDPLORA_V8_B_ONLY_AGGS
    | FEDPLORA_V8_AB_AGGS
    | FEDPLORA_V8_WARMA_AGGS
    | FEDPLORA_V8_PERIODIC_A_AGGS
)

FEDPLORA_V9_B_ONLY_AGGS = {
    "fedplora_v9_mix",
    "v9_mix",
}

FEDPLORA_V9_AB_AGGS = {
    "fedplora_v9_mix_ab",
    "v9_mix_ab",
}

FEDPLORA_V9_FAMILY_AGGS = FEDPLORA_V9_B_ONLY_AGGS | FEDPLORA_V9_AB_AGGS

FEDPLORA_V10_GEOM_A_AGGS = {
    "fedplora_v10_geom_a",
    "v10_geom_a",
}

FEDPLORA_V10_SKETCH_A_AGGS = {
    "fedplora_v10_sketch_a",
    "v10_sketch_a",
}

FEDPLORA_V10_FAMILY_AGGS = FEDPLORA_V10_GEOM_A_AGGS | FEDPLORA_V10_SKETCH_A_AGGS

FEDPLORA_V11A_RELAXED_A_AGGS = {
    "fedplora_v11a_relaxed_a",
    "fedplora_v11a",
    "v11a_relaxed_a",
    "v11a",
}

FEDPLORA_V11C_GMIX_AGGS = {
    "fedplora_v11c_gmix",
    "fedplora_v11_gmix",
    "v11c_gmix",
    "v11_gmix",
}

FEDPLORA_V11_FAMILY_AGGS = FEDPLORA_V11A_RELAXED_A_AGGS | FEDPLORA_V11C_GMIX_AGGS

FEDPLORA_V12A_SCHED_GMIX_AGGS = {
    "fedplora_v12a_sched_gmix",
    "fedplora_v12_sched_gmix",
    "v12a_sched_gmix",
    "v12_sched_gmix",
}

FEDPLORA_V12B_NMI_GUARD_GMIX_AGGS = {
    "fedplora_v12b_nmi_guard_gmix",
    "fedplora_v12_adaptive_gmix",
    "v12b_nmi_guard_gmix",
    "v12_adaptive_gmix",
}

FEDPLORA_V12_FAMILY_AGGS = (
    FEDPLORA_V12A_SCHED_GMIX_AGGS | FEDPLORA_V12B_NMI_GUARD_GMIX_AGGS
)

FEDPLORA_V13A_OS_AGGS = {
    "fedplora_v13a_os",
    "fedplora_v13a_oneshot",
    "v13a_os",
    "v13a_oneshot",
    "fedplora_os",
    "os_alpha100",
}

FEDPLORA_V13B_OS_BONLY_AGGS = {
    "fedplora_v13b_os_bonly",
    "fedplora_v13b_bonly",
    "v13b_os_bonly",
    "v13b_bonly",
    "fedplora_os_bonly",
    "os_bonly",
}

FEDPLORA_V13_FAMILY_AGGS = FEDPLORA_V13A_OS_AGGS | FEDPLORA_V13B_OS_BONLY_AGGS


def is_fedplora_v8_family_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V8_FAMILY_AGGS


def is_fedplora_v9_family_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V9_FAMILY_AGGS


def is_fedplora_v10_family_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V10_FAMILY_AGGS


def is_fedplora_v10_sketch_a_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V10_SKETCH_A_AGGS


def is_fedplora_v11_family_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V11_FAMILY_AGGS


def is_fedplora_v11_gmix_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V11C_GMIX_AGGS


def is_fedplora_v12_family_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V12_FAMILY_AGGS


def is_fedplora_v12_gmix_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V12_FAMILY_AGGS


def is_fedplora_v13_family_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V13_FAMILY_AGGS


def is_fedplora_v13_a_sketch_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V13A_OS_AGGS


def is_fedplora_v13_os_bonly_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V13B_OS_BONLY_AGGS


def is_fedplora_v8_warma_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V8_WARMA_AGGS


def is_fedplora_v8_periodic_a_agg(agg_type):
    return _norm_agg_type(agg_type) in FEDPLORA_V8_PERIODIC_A_AGGS


def is_fedplora_v8_scheduled_a_agg(agg_type):
    t = _norm_agg_type(agg_type)
    return t in FEDPLORA_V8_WARMA_AGGS or t in FEDPLORA_V8_PERIODIC_A_AGGS


def v8_train_a_this_round(
    agg_type,
    round_idx,
    *,
    warmup_rounds=1,
    refresh_interval=5,
):
    """Whether a v8 variant should train/upload LoRA A in zero-based round_idx."""
    t = _norm_agg_type(agg_type)
    if t in FEDPLORA_V8_AB_AGGS:
        return True
    if t in FEDPLORA_V8_B_ONLY_AGGS:
        return False
    if t in FEDPLORA_V8_WARMA_AGGS:
        return int(round_idx) < max(0, int(warmup_rounds or 0))
    if t in FEDPLORA_V8_PERIODIC_A_AGGS:
        interval = max(1, int(refresh_interval or 1))
        return int(round_idx) % interval == 0
    return False


def v8_a_refresh_round_count(
    agg_type,
    rounds,
    *,
    warmup_rounds=1,
    refresh_interval=5,
):
    """Number of rounds whose v8 uplink/downlink includes A under the schedule."""
    rounds = max(1, int(rounds or 1))
    if not is_fedplora_v8_family_agg(agg_type):
        return 0
    return sum(
        1
        for r in range(rounds)
        if v8_train_a_this_round(
            agg_type,
            r,
            warmup_rounds=warmup_rounds,
            refresh_interval=refresh_interval,
        )
    )


def is_fedplora_agg(agg_type):
    """Multi-round FedP-LoRA: sign-aligned A agg + local FedP regularizers."""
    return _norm_agg_type(agg_type) == "fedplora"


def is_v4_agg(agg_type):
    return _norm_agg_type(agg_type).startswith("v4_")


def is_v5_agg(agg_type):
    return _norm_agg_type(agg_type).startswith("v5_")


def is_v5_merge_agg(agg_type):
    """v5-merge family: clients upload trainable LoRA A+B; server merges ΔW = B·A
    with an interference-aware operator (mean / ties / dare_ties / knots_ties)."""
    return _norm_agg_type(agg_type).startswith("v5m_")


def is_fedplora_v6_dcr_agg(agg_type):
    """FedPLoRA v6 / DCR: A-only one-shot Grassmann subspace consensus."""
    return _norm_agg_type(agg_type) in {
        "v6_dcr",
        "v6_dcr_global",
        "v6_dcr_domain",
        "fedplora_dcr",
        "fedplora_dcr_global",
        "fedplora_dcr_domain",
    }


def is_lora_expert_agg(agg_type):
    """Shared-A / B-expert baselines: v7/v8/v9/v10/v11/v12/v13, FedLEASE, HiLoRA, EcoLoRA, HydraLoRA."""
    return _norm_agg_type(agg_type) in {
        "fedplora_v7",
        "fedplora_v7_bonly",
        "v7_bsim",
        "v7_bonly",
        "fedlease",
        "hilora",
        "ecolora",
        "hydralora",
    } or is_fedplora_v8_family_agg(agg_type) or is_fedplora_v9_family_agg(agg_type) or is_fedplora_v10_family_agg(agg_type) or is_fedplora_v11_family_agg(agg_type) or is_fedplora_v12_family_agg(agg_type) or is_fedplora_v13_family_agg(agg_type)


def is_lora_expert_b_only_agg(agg_type):
    """FedPLoRA variants that freeze shared A and upload B-only each round."""
    return _norm_agg_type(agg_type) in {
        "fedplora_v7_bonly",
        "v7_bonly",
    } or _norm_agg_type(agg_type) in (
        FEDPLORA_V8_B_ONLY_AGGS | FEDPLORA_V9_B_ONLY_AGGS | FEDPLORA_V13B_OS_BONLY_AGGS
    )


def uses_fedplora_oneshot_server_agg(agg_type):
    """Server-side conflict-gated oneshot (v2 + v4/v5 local-personalized branches)."""
    t = _norm_agg_type(agg_type)
    if t == "fedplora_oneshot":
        return True
    return t in {
        "v4_sign_v2agg",
        "v4_sign_full",
        "v4_mix_fixed05",
        "v4_mix_per_domain",
        "v4_mix_moe",
        "v5_route_mix_align",
        "v5_route_mix_align_domain",
        "v5_route_mix_align_local",
        "v5_rpca_route_mix_align",
    }


def is_fedplora_oneshot_agg(agg_type):
    """
    FedPLoRA-Oneshot: exactly one federated round; clients upload LoRA A,
    trainable heads, and row-importance stats while B stays local. Server uses
    conflict-gated A aggregation against the initial shared A0.
    """
    return _norm_agg_type(agg_type) == "fedplora_oneshot"


def is_fedplora_v3_agg(agg_type):
    """FedPLoRA-Oneshot v3 variants (residual conflict; A-only upload)."""
    return _norm_agg_type(agg_type) in {
        "fedplora_v3_lite",
        "v3_lite",
        "fedplora_v3_cluster",
        "v3_cluster",
        "fedplora_v3_rpca",
        "v3_rpca",
    }


def is_fedplora_oneshot_family_agg(agg_type):
    """One-round A-only FedPLoRA family: v2/v3/v4/v5 plus v6 DCR."""
    return (
        is_fedplora_oneshot_agg(agg_type)
        or is_fedplora_v3_agg(agg_type)
        or is_fedplora_v6_dcr_agg(agg_type)
        or uses_fedplora_oneshot_server_agg(agg_type)
    )


def is_lora_a_disk_agg(agg_type):
    """Domain SFT: server holds A (+ heads), clients keep B on disk/memory."""
    return _norm_agg_type(agg_type) in {
        "fedplora",
        "fedplora_oneshot",
        "fedsa_lora",
        "fedsa",
    } or is_fedplora_v3_agg(agg_type) or is_fedplora_v6_dcr_agg(agg_type) or is_lora_expert_agg(agg_type) or is_v4_agg(agg_type) or is_v5_agg(agg_type)


def is_fedalt_sequential_agg(agg_type):
    """FedALT: per-client Individual LoRA (A+B) on disk; server sends personalized RoTW (A+B)."""
    return _norm_agg_type(agg_type) == "fedalt"


def is_fedplora_multiround_agg(agg_type):
    """Multi-round FedP-LoRA server aggregate + local FedP regularizers (train_eval)."""
    return _norm_agg_type(agg_type) == "fedplora"


def is_fedsa_lora_agg(agg_type):
    return _norm_agg_type(agg_type) in {"fedsa_lora", "fedsa"}


def is_yoco_agg(agg_type):
    return _norm_agg_type(agg_type) == "yoco"


def is_flora_agg(agg_type):
    return _norm_agg_type(agg_type) == "flora"


def is_flexlora_agg(agg_type):
    return _norm_agg_type(agg_type) == "flexlora"


def is_feddat_agg(agg_type):
    return _norm_agg_type(agg_type) == "feddat"


def is_fedalt_agg(agg_type):
    return _norm_agg_type(agg_type) == "fedalt"


def is_ffa_agg(agg_type):
    return _norm_agg_type(agg_type) == "ffa"


def is_memory_global_agg_agg(agg_type):
    """
    In-memory FL: server fuses client LoRA into one global trainable state each round.
    Domain-macro eval should use that aggregated LoRA (not per-client local snapshots).
    """
    return _norm_agg_type(agg_type) in {
        "normal",
        "ffa",
        "flora",
        "flexlora",
        "feddat",
        "yoco",
    } or is_v5_merge_agg(agg_type)


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


def is_peft_base_layer_weight_key(key):
    """Frozen PEFT base weights (legacy FedEx path removed from domain SFT)."""
    return ".base_layer.weight" in key


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
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tensor_to_list(v) for v in obj]
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
    state_dict,
    agg_type,
    trainable_param_names=None,
    *,
    ecolora_keep_ratio=0.25,
    rounds=1,
    v8_a_warmup_rounds=1,
    v8_a_refresh_interval=5,
    v8_cache_shared_a_downlink=True,
    v10_a_sketch_rank=2,
    v11_global_b_mix_mu=0.4,
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

    lora_all = _sum_bytes_state_dict(sd, is_lora_param_name)
    lora_a = _sum_bytes_state_dict(sd, is_lora_a_param_name)
    lora_b = _sum_bytes_state_dict(sd, is_lora_b_param_name)
    task_head = _sum_bytes_state_dict(sd, is_trainable_task_head)
    gp_stats = gp_row_importance_bytes()
    has_trainable_lora_a = True
    if trainable_param_names is not None:
        has_trainable_lora_a = any(
            is_lora_a_param_name(k) for k in trainable_param_names
        )

    agg_type = _norm_agg_type(agg_type) or "normal"

    v8_refresh_rounds = v8_a_refresh_round_count(
        agg_type,
        rounds,
        warmup_rounds=v8_a_warmup_rounds,
        refresh_interval=v8_a_refresh_interval,
    )
    if is_fedplora_v9_family_agg(agg_type) and agg_type in FEDPLORA_V9_AB_AGGS:
        v8_refresh_rounds = max(1, int(rounds or 1))
    if is_fedplora_v10_family_agg(agg_type):
        v8_refresh_rounds = max(1, int(rounds or 1))
    if (
        is_fedplora_v11_family_agg(agg_type)
        or is_fedplora_v12_family_agg(agg_type)
        or is_fedplora_v13_a_sketch_agg(agg_type)
    ):
        v8_refresh_rounds = max(1, int(rounds or 1))

    if is_fedplora_v8_scheduled_a_agg(agg_type):
        # Raw per-refresh-round protocol: A+B are available when A is refreshed;
        # effective_* below amortizes A across the full run and optionally caches
        # shared A on clients for B-only rounds.
        down = lora_all + task_head
        up = lora_all + task_head
    elif is_fedplora_v13_os_bonly_agg(agg_type):
        # v13b is the explicit one-shot/cached-A routed-B attribution branch:
        # only routed B (+ task head) is paid in the measured round, both raw
        # and effective.  Historical v8 accounting is intentionally unchanged.
        down = lora_b + task_head
        up = lora_b + task_head
    elif is_lora_expert_b_only_agg(agg_type) or (
        is_lora_expert_agg(agg_type) and not has_trainable_lora_a
    ):
        # A is broadcast as the shared coordinate system, but only B is uploaded.
        down = lora_all + task_head
        up = lora_b + task_head
    elif is_fedplora_v10_family_agg(agg_type):
        down = lora_all + task_head
        up = lora_all + task_head
    elif (
        is_fedplora_v11_family_agg(agg_type)
        or is_fedplora_v12_family_agg(agg_type)
        or is_fedplora_v13_a_sketch_agg(agg_type)
    ):
        # v11/v12/v13a use the real sketch payload: B + task head + rank-k A-delta
        # sketch.  Unlike v10, raw and effective uplink are intentionally the
        # same accounting object for the A-correction branch.
        rank = max(1, int(v10_a_sketch_rank or 1))
        a_sketch_bytes = 0
        for k, v in sd.items():
            if not is_lora_a_param_name(k) or v.ndim != 2:
                continue
            r_dim = int(v.shape[0])
            in_dim = int(v.shape[1])
            elem = int(v.element_size())
            a_sketch_bytes += int((r_dim * rank + rank + rank * in_dim) * elem)
        down = lora_b + task_head + a_sketch_bytes
        up = lora_b + task_head + a_sketch_bytes
    elif is_lora_expert_agg(agg_type):
        down = lora_all + task_head
        if _norm_agg_type(agg_type) == "ecolora":
            # EcoLoRA-style sparse/segmented upload; use the actual CLI keep ratio
            # when the caller provides it.
            ratio = min(1.0, max(0.0, float(ecolora_keep_ratio)))
            up = int(round(ratio * lora_all)) + task_head
        else:
            up = lora_all + task_head
    elif is_fedplora_multiround_agg(agg_type) or is_fedplora_oneshot_family_agg(agg_type):
        down = lora_a + task_head
        up = lora_a + task_head + gp_stats
    elif is_lora_a_disk_agg(agg_type):
        down = lora_a + task_head
        up = lora_a + task_head
    elif is_fedalt_sequential_agg(agg_type):
        # FedALT: uplink Individual LoRA A+B; downlink personalized RoTW LoRA A+B.
        down = lora_all + task_head
        up = lora_all + task_head
    elif is_ffa_agg(agg_type):
        # FFA: frozen shared A; broadcast / aggregate B (+ heads) only.
        down = lora_b + task_head
        up = lora_b + task_head
    elif is_memory_global_agg_agg(agg_type):
        # normal / flora / flexlora / feddat / yoco: trainable LoRA A+B + heads (no frozen backbone).
        down = lora_all + task_head
        up = lora_all + task_head
    else:
        # Fallback: never count frozen backbone in link budget.
        down = lora_all + task_head
        up = lora_all + task_head

    effective_down = down
    effective_up = up
    v10_a_correction_bytes = 0
    v11_a_correction_bytes = 0
    if is_fedplora_v10_sketch_a_agg(agg_type):
        rank = max(1, int(v10_a_sketch_rank or 1))
        for k, v in sd.items():
            if not is_lora_a_param_name(k) or v.ndim != 2:
                continue
            r_dim = int(v.shape[0])
            in_dim = int(v.shape[1])
            elem = int(v.element_size())
            # U(r x k) + S(k) + V(k x in_dim), matching the rank-k delta sketch.
            v10_a_correction_bytes += int((r_dim * rank + rank + rank * in_dim) * elem)
    elif is_fedplora_v10_family_agg(agg_type):
        v10_a_correction_bytes = lora_a
    elif (
        is_fedplora_v11_family_agg(agg_type)
        or is_fedplora_v12_family_agg(agg_type)
        or is_fedplora_v13_a_sketch_agg(agg_type)
    ):
        rank = max(1, int(v10_a_sketch_rank or 1))
        for k, v in sd.items():
            if not is_lora_a_param_name(k) or v.ndim != 2:
                continue
            r_dim = int(v.shape[0])
            in_dim = int(v.shape[1])
            elem = int(v.element_size())
            v11_a_correction_bytes += int((r_dim * rank + rank + rank * in_dim) * elem)

    if is_fedplora_v8_family_agg(agg_type) or is_fedplora_v9_family_agg(agg_type):
        run_rounds = max(1, int(rounds or 1))
        a_share = float(v8_refresh_rounds) / float(run_rounds)
        if is_fedplora_v9_family_agg(agg_type) and agg_type in FEDPLORA_V9_AB_AGGS:
            effective_up = int(round(lora_all + task_head))
            a_share = 1.0
        else:
            effective_up = int(round(lora_b + task_head + a_share * lora_a))
        if bool(v8_cache_shared_a_downlink):
            effective_down = int(round(lora_b + task_head + a_share * lora_a))
            downlink_policy = "cache_shared_a_routed_b"
        else:
            effective_down = int(round(lora_all + task_head))
            downlink_policy = "raw_full_lora"
    elif is_fedplora_v10_family_agg(agg_type):
        effective_up = int(round(lora_b + task_head + v10_a_correction_bytes))
        if bool(v8_cache_shared_a_downlink):
            effective_down = int(round(lora_b + task_head + v10_a_correction_bytes))
            downlink_policy = "cache_shared_a_plus_v10_a_correction"
        else:
            effective_down = int(round(lora_all + task_head))
            downlink_policy = "raw_full_lora"
    elif (
        is_fedplora_v11_family_agg(agg_type)
        or is_fedplora_v12_family_agg(agg_type)
        or is_fedplora_v13_a_sketch_agg(agg_type)
    ):
        effective_up = int(round(lora_b + task_head + v11_a_correction_bytes))
        if bool(v8_cache_shared_a_downlink):
            effective_down = int(round(lora_b + task_head + v11_a_correction_bytes))
            downlink_policy = (
                "cache_shared_a_plus_true_v12_a_sketch"
                if is_fedplora_v12_family_agg(agg_type)
                else (
                    "cache_shared_a_plus_true_v13_a_sketch"
                    if is_fedplora_v13_a_sketch_agg(agg_type)
                    else "cache_shared_a_plus_true_v11_a_sketch"
                )
            )
        else:
            effective_down = int(round(lora_all + task_head))
            downlink_policy = "raw_full_lora"
    elif is_fedplora_v13_os_bonly_agg(agg_type):
        effective_up = int(round(lora_b + task_head))
        effective_down = int(round(lora_b + task_head))
        downlink_policy = "one_shot_cached_shared_a_routed_b"
    else:
        downlink_policy = "raw"

    return {
        "down_bytes_per_client": down,
        "up_bytes_per_client": up,
        "effective_down_bytes_per_client": effective_down,
        "effective_up_bytes_per_client": effective_up,
        "downlink_policy": downlink_policy,
        "v8_a_refresh_rounds": int(v8_refresh_rounds),
        "v8_a_refresh_fraction": (
            float(v8_refresh_rounds) / float(max(1, int(rounds or 1)))
            if is_fedplora_v8_family_agg(agg_type) or is_fedplora_v9_family_agg(agg_type) or is_fedplora_v10_family_agg(agg_type) or is_fedplora_v11_family_agg(agg_type) or is_fedplora_v12_family_agg(agg_type) or is_fedplora_v13_family_agg(agg_type)
            else 0.0
        ),
        "v10_a_correction_bytes_per_client": int(v10_a_correction_bytes),
        "v11_a_correction_bytes_per_client": int(v11_a_correction_bytes),
        "v10_a_correction_mode": (
            "lowrank_delta"
            if is_fedplora_v10_sketch_a_agg(agg_type)
            else ("anchored_full_delta" if is_fedplora_v10_family_agg(agg_type) else "")
        ),
        "v11_a_correction_mode": (
            "true_lowrank_delta" if (is_fedplora_v11_family_agg(agg_type) or is_fedplora_v12_family_agg(agg_type) or is_fedplora_v13_a_sketch_agg(agg_type)) else ""
        ),
        "v11_global_b_mix_mu": (
            float(v11_global_b_mix_mu) if (is_fedplora_v11_gmix_agg(agg_type) or is_fedplora_v12_gmix_agg(agg_type)) else 0.0
        ),
    }


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
