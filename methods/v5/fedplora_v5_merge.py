"""FedPLoRA v5-merge server aggregation.

One-shot federated LoRA reframed as interference-aware model merging:
clients upload trainable LoRA (A_i, B_i) + heads (same uplink as
normal/flora/flexlora baselines); the server merges the implied updates
ΔW_i = B_i A_i with a merging operator that resolves cross-domain
interference, then refactors ΔW* back into LoRA factors for the single
downlink.

Operators (selected by agg_type):
  v5m_mean        weighted mean (sanity; ≈ FlexLoRA protocol)
  v5m_ties        entry-wise TIES: trim → elect sign → disjoint weighted mean
  v5m_dare_ties   DARE drop-and-rescale before TIES
  v5m_knots_ties  shared-subspace alignment (KnOTS-style) + TIES on aligned
                  coefficients; fully factored, exact reconstruction

Rank policy (args.v5m_rank_policy):
  fixed   downlink rank = lora_r (identical communication to flora/flexlora)
  energy  downlink rank = smallest r with ≥ args.v5m_energy_tau spectral
          energy, capped at args.v5m_rank_cap (PEFT modules are rebuilt
          in-place to the new rank; scaling convention unchanged)
"""

from __future__ import annotations

import torch

from methods import common as M
from methods.v5.merge_ops import (
    delta_frobenius_norm,
    factorize_coefficient_delta,
    factorize_dense_delta,
    knots_align_factors,
    merge_delta_entrywise,
    normalize_weights,
    pick_rank_by_energy,
    ties_on_coefficients,
)
from utilities.utils import (
    is_lora_a_param_name,
    is_task_head_param_name,
)


_VALID_MODES = {"mean", "ties", "dare_ties", "knots_ties"}


def _merge_mode_from_agg_type(agg_type: str) -> str:
    t = (agg_type or "").strip().lower().replace("-", "_")
    if not t.startswith("v5m_"):
        raise ValueError(f"Not a v5-merge agg_type: {agg_type!r}")
    mode = t[len("v5m_"):]
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown v5-merge mode {mode!r}; valid: {sorted(_VALID_MODES)}")
    return mode


def _resolve_device(global_model, args):
    pref = str(getattr(args, "v5m_device", "auto") or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            return next(global_model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def _short_layer_name(key: str) -> str:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            tail = ".".join(parts[i: i + 4])
            return tail if len(tail) < 80 else tail[:77] + "..."
    return parts[-4] if len(parts) >= 4 else key[-60:]


def _module_path_from_lora_key(key: str) -> str:
    """'<path>.lora_A.default.weight' → '<path>' (the PEFT-wrapped Linear)."""
    marker = ".lora_A."
    idx = key.find(marker)
    if idx < 0:
        raise ValueError(f"Not a lora_A key: {key!r}")
    return key[:idx]


def _surgery_set_rank(global_model, key_a: str, A_new: torch.Tensor, B_new: torch.Tensor):
    """Rebuild lora_A/lora_B 'default' Linears in-place to the new rank.

    The PEFT ``scaling`` dict entry is intentionally left unchanged: all client
    updates and the merged factors share the same alpha/r_init convention, so
    the absorbed magnitudes stay consistent (same protocol as flora's refactor).
    """
    path = _module_path_from_lora_key(key_a)
    module = global_model.get_submodule(path)
    old_a = module.lora_A["default"].weight
    device, dtype = old_a.device, old_a.dtype
    requires_grad = bool(old_a.requires_grad)

    r_new, n = A_new.shape
    m = B_new.shape[0]

    new_a = torch.nn.Linear(n, r_new, bias=False, device=device, dtype=dtype)
    new_b = torch.nn.Linear(r_new, m, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        new_a.weight.copy_(A_new.to(device=device, dtype=dtype))
        new_b.weight.copy_(B_new.to(device=device, dtype=dtype))
    new_a.weight.requires_grad_(requires_grad)
    new_b.weight.requires_grad_(requires_grad)

    module.lora_A["default"] = new_a
    module.lora_B["default"] = new_b
    if hasattr(module, "r") and isinstance(module.r, dict):
        module.r["default"] = int(r_new)


def aggregate_models_v5_merge(global_model, client_states_for_agg, args):
    """Server step: merge client ΔW with the selected operator, refactor, load."""
    client_states = [
        src["state_dict"] if isinstance(src, dict) and "state_dict" in src else M.obj_sd(src)
        for src in client_states_for_agg
    ]
    n_clients = len(client_states)
    if n_clients == 0:
        return global_model

    mode = _merge_mode_from_agg_type(getattr(args, "agg_type", ""))
    weights = normalize_weights(getattr(args, "_aggregate_client_sizes", None), n_clients)

    keep_ratio = float(getattr(args, "v5m_keep_ratio", 0.2))
    dare_p = float(getattr(args, "v5m_dare_p", 0.3)) if mode == "dare_ties" else float(
        getattr(args, "v5m_dare_p_always", 0.0)
    )
    rank_policy = str(getattr(args, "v5m_rank_policy", "fixed") or "fixed").lower()
    rank_cap = int(getattr(args, "v5m_rank_cap", 64))
    energy_tau = float(getattr(args, "v5m_energy_tau", 0.95))
    knots_normalize = bool(int(getattr(args, "v5m_knots_normalize", 1)))
    basis_tau = float(getattr(args, "v5m_basis_energy_tau", 0.9999))
    chunk_rows = int(getattr(args, "v5m_chunk_rows", 2048))
    seed = int(getattr(args, "seed", 0))

    r_fixed = int(getattr(args, "lora_r", 8))
    device = _resolve_device(global_model, args)
    global_dict = global_model.state_dict()

    lora_keys = [
        k
        for k in global_dict.keys()
        if is_lora_a_param_name(k)
        and k.replace("lora_A", "lora_B") in global_dict
        and all(
            k in st and k.replace("lora_A", "lora_B") in st for st in client_states
        )
    ]
    print(
        f"[v5m] mode={mode} clients={n_clients} layers={len(lora_keys)} "
        f"keep_ratio={keep_ratio} dare_p={dare_p} rank_policy={rank_policy} "
        f"(cap={rank_cap}, tau={energy_tau}) device={device}",
        flush=True,
    )

    layer_logs = []
    surgery_layers = []
    down_bytes = 0

    for idx, kA in enumerate(lora_keys):
        kB = kA.replace("lora_A", "lora_B")
        B_list = [client_states[i][kB] for i in range(n_clients)]
        A_list = [client_states[i][kA] for i in range(n_clients)]

        if mode == "knots_ties":
            C_list, V, align_stats = knots_align_factors(
                B_list, A_list, weights,
                device=device, basis_energy_tau=basis_tau,
            )
            C_star, ties_stats = ties_on_coefficients(
                C_list, weights,
                keep_ratio=keep_ratio, dare_p=dare_p,
                normalize=knots_normalize, seed=seed + idx,
            )
            # First factorize at max useful rank to read the spectrum, then cut.
            A_full, B_full, S = factorize_coefficient_delta(
                C_star, V, r_down=min(rank_cap, C_star.shape[1])
            )
            stats = {**align_stats, **ties_stats}
        else:
            ew_mode = "mean" if mode == "mean" else "ties"
            ew_dare = dare_p if mode == "dare_ties" else 0.0
            delta, stats = merge_delta_entrywise(
                B_list, A_list, weights,
                mode=ew_mode, keep_ratio=keep_ratio, dare_p=ew_dare,
                chunk_rows=chunk_rows, device=device, seed=seed + idx,
            )
            A_full, B_full, S = factorize_dense_delta(
                delta, r_down=min(rank_cap, min(delta.shape)), device=device
            )
            del delta

        if rank_policy == "energy":
            r_down = pick_rank_by_energy(S, tau=energy_tau, r_min=1, r_cap=rank_cap)
        else:
            r_down = min(r_fixed, S.numel())
        A_new = A_full[:r_down].contiguous()
        B_new = B_full[:, :r_down].contiguous()

        energy_kept = float(
            (S[:r_down].pow(2).sum() / S.pow(2).sum().clamp_min(1e-12)).item()
        )
        down_bytes += int(A_new.numel() + B_new.numel()) * 2  # bf16 link estimate

        if r_down == global_dict[kA].shape[0]:
            global_dict[kA] = A_new.to(device=global_dict[kA].device, dtype=global_dict[kA].dtype)
            global_dict[kB] = B_new.to(device=global_dict[kB].device, dtype=global_dict[kB].dtype)
        else:
            surgery_layers.append((kA, A_new.cpu(), B_new.cpu()))

        layer_logs.append(
            {
                "layer": _short_layer_name(kA),
                "r_down": int(r_down),
                "energy_kept": energy_kept,
                **{k: float(v) if isinstance(v, (int, float)) else v for k, v in stats.items()},
            }
        )
        if (idx == 0) or (idx + 1) % 16 == 0 or (idx + 1) == len(lora_keys):
            print(
                f"[v5m] layer {idx + 1}/{len(lora_keys)} {_short_layer_name(kA)} "
                f"r_down={r_down} energy={energy_kept:.4f}",
                flush=True,
            )

    # Heads: sample-size weighted mean (same as flexlora).
    for k in global_dict.keys():
        if is_task_head_param_name(k) and all(k in st for st in client_states):
            stacked = torch.stack([client_states[i][k].float() for i in range(n_clients)], dim=0)
            wf = weights.to(device=stacked.device, dtype=stacked.dtype)
            agg = (wf.view(n_clients, *([1] * (stacked.ndim - 1))) * stacked).sum(dim=0)
            global_dict[k] = agg.to(device=global_dict[k].device, dtype=global_dict[k].dtype)

    # Apply same-rank tensors first, then any rank-changing surgery.
    global_model.load_state_dict(global_dict)
    for kA, A_new, B_new in surgery_layers:
        _surgery_set_rank(global_model, kA, A_new, B_new)

    if layer_logs:
        mean_r = sum(l["r_down"] for l in layer_logs) / len(layer_logs)
        mean_energy = sum(l["energy_kept"] for l in layer_logs) / len(layer_logs)
        actives = [l["active_frac"] for l in layer_logs if "active_frac" in l]
        summary = {
            "mode": mode,
            "mean_r_down": mean_r,
            "mean_energy_kept": mean_energy,
            "mean_active_frac": (sum(actives) / len(actives)) if actives else None,
            "num_surgery_layers": len(surgery_layers),
            "down_bytes_lora_estimate": int(down_bytes),
        }
        setattr(args, "_v5m_merge_stats", {"layers": layer_logs, "_summary": summary})
        print(
            f"[v5m] done: mean_r_down={mean_r:.1f} mean_energy={mean_energy:.4f} "
            f"surgery_layers={len(surgery_layers)} "
            f"down_lora_bytes≈{down_bytes / (1024**2):.1f}MB",
            flush=True,
        )

    return global_model
