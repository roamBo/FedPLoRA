"""
Framework-level reproductions of recent LoRA expert / sparse-communication
baselines for the domain SFT pipeline.

These implementations keep the repository's single-adapter PEFT model intact.
They reproduce the server-side mechanisms that can be expressed as one
downlinked LoRA state per client:
  - fedplora_v7 / v7_bsim: shared A plus B-similarity pools.
  - fedplora_v8 / v8_bsim: shared/frozen A, B-only upload, and automatic
    LoRA-B principal-angle pools.
  - fedplora_v8_ab / v8_bsim_ab: v8 ablation with A+B upload.
  - fedplora_v8_warma / fedplora_v8_periodic: v8 A-refresh schedules that
    train/upload A only in warmup or periodic refresh rounds.
  - fedplora_v9_mix: global/routed dual-branch B mixing (implemented in
    methods.v9; listed here only for shared helper compatibility).
  - fedplora_v10_geom_a / sketch_a: geometry-preserving A correction plus
    routed B (implemented in methods.v10; listed here for shared helpers).
  - fedplora_v11a_relaxed_a / v11c_gmix: true sketch A-correction payloads
    plus routed/global-mixed B (implemented in methods.v11; listed here for
    shared helpers).
  - fedplora_v12a_sched_gmix / v12b_nmi_guard_gmix: v11-style true sketch
    payloads plus scheduled/adaptive global-routed B mixing (implemented in
    methods.v12; listed here for shared helpers).
  - fedplora_v13a_os / v13b_os_bonly: 2026-07-11 one-shot protocol branches
    (implemented in methods.v13; listed here for shared helper compatibility).
  - fedlease: adaptive B-subspace expert allocation plus top-M expert blending.
  - hilora: root shared A, cluster B, and a local leaf residual folded into B.
  - hydralora: shared A plus multiple B experts with hard/soft routing.
  - ecolora: sparse/segmented LoRA upload simulation with unbiased aggregation.

Full paper-level reproductions of FedLEASE/HiLoRA/HydraLoRA require model
architecture changes (MoE routers or multi-tier adapters). This module is the
comparable federated aggregation baseline within the current framework.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import math
import numpy as np
import torch

from methods import common as M
from utilities.utils import (
    FEDPLORA_V8_FAMILY_AGGS,
    FEDPLORA_V11_FAMILY_AGGS,
    FEDPLORA_V12_FAMILY_AGGS,
    FEDPLORA_V13_FAMILY_AGGS,
    get_trainable_param_names,
    is_lora_a_param_name,
    is_lora_b_param_name,
    is_task_head_param_name,
)


SUPPORTED_LORA_EXPERT_AGGS = {
    "fedplora_v7",
    "fedplora_v7_bonly",
    "v7_bsim",
    "v7_bonly",
    "fedplora_v8",
    "fedplora_v8_ab",
    "fedplora_v8_warma",
    "fedplora_v8_warm_a",
    "fedplora_v8_warma_then_freeze",
    "fedplora_v8_periodic",
    "fedplora_v8_periodica",
    "fedplora_v8_periodica_t",
    "v8_bsim",
    "v8_bsim_ab",
    "v8_warma",
    "v8_warm_a",
    "v8_warma_then_freeze",
    "v8_periodic",
    "v8_periodica",
    "v8_periodica_t",
    "fedplora_v9_mix",
    "fedplora_v9_mix_ab",
    "v9_mix",
    "v9_mix_ab",
    "fedplora_v10_geom_a",
    "fedplora_v10_sketch_a",
    "v10_geom_a",
    "v10_sketch_a",
    "fedplora_v11a_relaxed_a",
    "fedplora_v11a",
    "v11a_relaxed_a",
    "v11a",
    "fedplora_v11c_gmix",
    "fedplora_v11_gmix",
    "v11c_gmix",
    "v11_gmix",
    "fedplora_v12a_sched_gmix",
    "fedplora_v12_sched_gmix",
    "v12a_sched_gmix",
    "v12_sched_gmix",
    "fedplora_v12b_nmi_guard_gmix",
    "fedplora_v12_adaptive_gmix",
    "v12b_nmi_guard_gmix",
    "v12_adaptive_gmix",
    "fedplora_v13a_os",
    "fedplora_v13a_oneshot",
    "v13a_os",
    "v13a_oneshot",
    "fedplora_os",
    "os_alpha100",
    "fedplora_v13b_os_bonly",
    "fedplora_v13b_bonly",
    "v13b_os_bonly",
    "v13b_bonly",
    "fedplora_os_bonly",
    "os_bonly",
    "fedlease",
    "hilora",
    "ecolora",
    "hydralora",
}


def _norm_agg(agg_type) -> str:
    return (agg_type or "").strip().lower().replace("-", "_")


def _canonical_agg(agg_type) -> str:
    t = _norm_agg(agg_type)
    if t in {"fedplora_v7_bonly", "v7_bonly"}:
        return "fedplora_v7"
    if t in FEDPLORA_V8_FAMILY_AGGS:
        return "fedplora_v8"
    if t in FEDPLORA_V11_FAMILY_AGGS:
        return t
    if t in FEDPLORA_V12_FAMILY_AGGS:
        return t
    if t in FEDPLORA_V13_FAMILY_AGGS:
        return t
    return t


def is_supported_lora_expert_agg(agg_type) -> bool:
    return _norm_agg(agg_type) in SUPPORTED_LORA_EXPERT_AGGS


def _client_weight_vector(client_uploads, args) -> np.ndarray:
    sizes = [M.upload_package_client_size(m) for m in client_uploads]
    if all(x is None for x in sizes):
        sizes = getattr(args, "_runtime_client_sizes", None)
    n = len(client_uploads)
    if sizes is None or len(sizes) < n:
        return np.ones(n, dtype=np.float64) / max(n, 1)
    arr = np.asarray([float(sizes[i]) for i in range(n)], dtype=np.float64)
    if not np.isfinite(arr).all() or float(arr.sum()) <= 0:
        return np.ones(n, dtype=np.float64) / max(n, 1)
    return arr / arr.sum()


def _metadata_from_uploads(client_uploads, args) -> Tuple[List[int], List[str]]:
    runtime_ids = list(getattr(args, "_fedplora_round_client_ids", []) or [])
    domain_map = getattr(args, "_fedplora_client_domains", {}) or {}
    client_ids: List[int] = []
    domains: List[str] = []
    for idx, upload in enumerate(client_uploads):
        client_id = None
        domain = None
        if isinstance(upload, dict):
            client_id = upload.get("client_id", None)
            domain = upload.get("domain", None)
        if client_id is None and idx < len(runtime_ids):
            client_id = runtime_ids[idx]
        if client_id is None:
            client_id = idx
        try:
            client_id = int(client_id)
        except Exception:
            client_id = idx
        if domain is None:
            domain = domain_map.get(client_id, domain_map.get(str(client_id), "unknown"))
        client_ids.append(client_id)
        domains.append(str(domain or "unknown").strip().lower())
    return client_ids, domains


def _is_ecolora(args) -> bool:
    return _norm_agg(getattr(args, "agg_type", "")) == "ecolora"


def _is_b_only_upload(args) -> bool:
    if args is None:
        return False
    override = getattr(args, "_v8_current_b_only_upload", None)
    if override is not None:
        return bool(override)
    t = _norm_agg(getattr(args, "agg_type", ""))
    return t in {
        "fedplora_v7_bonly",
        "v7_bonly",
        "fedplora_v8",
        "v8_bsim",
    } or bool(getattr(args, "expert_freeze_a", False))


def _ecolora_keep_ratio(args) -> float:
    ratio = float(getattr(args, "ecolora_keep_ratio", 0.25) or 0.25)
    return min(1.0, max(0.0, ratio))


def _ecolora_mask_for_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    client_index: int,
    round_index: int,
    args,
) -> torch.Tensor:
    keep_ratio = _ecolora_keep_ratio(args)
    flat_n = int(tensor.numel())
    if keep_ratio >= 1.0 or flat_n == 0:
        return torch.ones_like(tensor, dtype=torch.bool, device="cpu")
    keep_n = max(1, int(math.ceil(flat_n * keep_ratio)))
    mode = str(getattr(args, "ecolora_mask_mode", "round_robin") or "round_robin").lower()
    flat_mask = torch.zeros(flat_n, dtype=torch.bool)
    if mode == "topk":
        vals = tensor.detach().float().abs().reshape(-1).cpu()
        if keep_n >= flat_n:
            flat_mask[:] = True
        else:
            idx = torch.topk(vals, keep_n, largest=True, sorted=False).indices
            flat_mask[idx] = True
    else:
        # Complementary contiguous segments across clients and rounds.
        start = ((int(client_index) + int(round_index)) * keep_n) % flat_n
        end = start + keep_n
        if end <= flat_n:
            flat_mask[start:end] = True
        else:
            flat_mask[start:] = True
            flat_mask[: end - flat_n] = True
    return flat_mask.reshape(tensor.shape)


def build_lora_expert_upload_package(
    model,
    client_size=None,
    client_id=None,
    domain=None,
    *,
    args=None,
    client_index: int = 0,
    round_index: int = 0,
):
    """Client payload for expert baselines: trainable LoRA A+B plus heads."""
    sd = model.state_dict() if hasattr(model, "state_dict") else model
    trainable_names = (
        get_trainable_param_names(model) if hasattr(model, "named_parameters") else None
    )
    upload_sd = {}
    sparse_masks = {}
    use_sparse = args is not None and _is_ecolora(args)
    b_only_upload = _is_b_only_upload(args)

    for k, v in sd.items():
        include = False
        if is_lora_a_param_name(k):
            include = not b_only_upload
        elif is_lora_b_param_name(k):
            include = True
        elif is_task_head_param_name(k) and (
            trainable_names is None or k in trainable_names
        ):
            include = True
        if not include:
            continue

        value = v.detach().cpu().clone()
        if use_sparse and (is_lora_a_param_name(k) or is_lora_b_param_name(k)):
            mask = _ecolora_mask_for_tensor(
                k,
                value,
                client_index=client_index,
                round_index=round_index,
                args=args,
            )
            sparse_masks[k] = mask.cpu()
            value = value * mask.to(dtype=value.dtype)
        upload_sd[k] = value

    return {
        "state_dict": upload_sd,
        "client_size": client_size,
        "client_id": client_id,
        "domain": domain,
        "sparse_masks": sparse_masks,
    }


def _common_lora_keys(client_states: Sequence[Dict[str, torch.Tensor]], pred) -> List[str]:
    if not client_states:
        return []
    keys = [k for k in client_states[0].keys() if pred(k)]
    return [k for k in keys if all(k in st for st in client_states)]


def _aggregate_shared_A_and_heads(global_dict, client_states, weights: np.ndarray):
    n = len(client_states)
    for key in list(global_dict.keys()):
        if is_lora_a_param_name(key) and all(key in st for st in client_states):
            stacked = torch.stack([client_states[i][key].float() for i in range(n)], 0)
            w = torch.tensor(weights, dtype=stacked.dtype).view(n, *([1] * (stacked.ndim - 1)))
            agg = (w * stacked).sum(0)
            global_dict[key] = agg.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
        elif is_task_head_param_name(key) and all(key in st for st in client_states):
            stacked = torch.stack([client_states[i][key].float() for i in range(n)], 0)
            w = torch.tensor(weights, dtype=stacked.dtype).view(n, *([1] * (stacked.ndim - 1)))
            agg = (w * stacked).sum(0)
            global_dict[key] = agg.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )


def _orth_basis(mat: torch.Tensor, rank: int) -> torch.Tensor:
    x = mat.detach().float().cpu()
    if x.ndim != 2 or x.numel() == 0 or float(torch.linalg.vector_norm(x).item()) <= 1e-12:
        return torch.zeros((x.shape[0], min(rank, x.shape[1] if x.ndim == 2 else rank)))
    q, _ = torch.linalg.qr(x, mode="reduced")
    return q[:, : min(rank, q.shape[1])].contiguous()


def _b_subspace_distance_matrix(client_states, b_keys: Sequence[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    n = len(client_states)
    dist = np.zeros((n, n), dtype=np.float64)
    if n <= 1 or not b_keys:
        return dist, {"mean_pair_angle_deg": float("nan"), "mean_pair_cos": float("nan")}

    bases: Dict[Tuple[int, str], torch.Tensor] = {}
    for i, st in enumerate(client_states):
        for key in b_keys:
            b = st[key]
            rank = int(b.shape[1]) if b.ndim == 2 else min(b.shape)
            bases[(i, key)] = _orth_basis(b, rank)

    pair_cos = []
    pair_angle = []
    for i in range(n):
        for j in range(i + 1, n):
            sims = []
            for key in b_keys:
                qi = bases[(i, key)]
                qj = bases[(j, key)]
                if qi.numel() == 0 or qj.numel() == 0:
                    continue
                try:
                    sv = torch.linalg.svdvals(qi.transpose(0, 1) @ qj)
                    if sv.numel():
                        sims.append(float(sv.clamp(0, 1).mean().item()))
                except Exception:
                    continue
            sim = float(np.mean(sims)) if sims else 0.0
            sim = min(1.0, max(0.0, sim))
            d = 1.0 - sim
            dist[i, j] = dist[j, i] = d
            pair_cos.append(sim)
            pair_angle.append(math.degrees(math.acos(sim)))

    return dist, {
        "mean_pair_angle_deg": float(np.mean(pair_angle)) if pair_angle else float("nan"),
        "mean_pair_cos": float(np.mean(pair_cos)) if pair_cos else float("nan"),
    }


def _avg_linkage_distance(dist: np.ndarray, c1: Sequence[int], c2: Sequence[int]) -> float:
    vals = [dist[i, j] for i in c1 for j in c2 if i != j]
    return float(np.mean(vals)) if vals else 0.0


def _agglomerative_assign(dist: np.ndarray, k: int) -> List[int]:
    n = int(dist.shape[0])
    if n == 0:
        return []
    k = max(1, min(int(k), n))
    clusters: List[List[int]] = [[i] for i in range(n)]
    while len(clusters) > k:
        best = None
        best_d = float("inf")
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = _avg_linkage_distance(dist, clusters[a], clusters[b])
                if d < best_d:
                    best_d = d
                    best = (a, b)
        if best is None:
            break
        a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    assign = [0] * n
    for cid, members in enumerate(clusters):
        for idx in members:
            assign[idx] = cid
    return assign


def _silhouette_score(dist: np.ndarray, assign: Sequence[int]) -> float:
    n = int(dist.shape[0])
    if n <= 2:
        return 0.0
    labels = sorted(set(assign))
    if len(labels) <= 1 or len(labels) >= n:
        return 0.0
    scores = []
    for i in range(n):
        own = [j for j in range(n) if assign[j] == assign[i] and j != i]
        a = float(np.mean([dist[i, j] for j in own])) if own else 0.0
        b_vals = []
        for lab in labels:
            if lab == assign[i]:
                continue
            members = [j for j in range(n) if assign[j] == lab]
            if members:
                b_vals.append(float(np.mean([dist[i, j] for j in members])))
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        scores.append(0.0 if denom <= 1e-12 else (b - a) / denom)
    return float(np.mean(scores)) if scores else 0.0


def _auto_cluster_assign(dist: np.ndarray, args) -> Tuple[List[int], Dict[str, object]]:
    n = int(dist.shape[0])
    if n <= 2:
        return [0] * n, {"selected_k": 1, "silhouette": 0.0, "candidates": {}}
    explicit = int(getattr(args, "expert_cluster_k", 0) or 0)
    if explicit > 0:
        assign = _agglomerative_assign(dist, explicit)
        return assign, {
            "selected_k": int(max(assign) + 1) if assign else 0,
            "silhouette": _silhouette_score(dist, assign),
            "candidates": {},
        }
    max_k = int(getattr(args, "expert_max_clusters", 0) or 0)
    if max_k <= 0:
        algo = _norm_agg(getattr(args, "agg_type", ""))
        default_cap = 4 if algo == "fedlease" else 8
        max_k = min(default_cap, n - 1)
    max_k = max(2, min(max_k, n - 1))
    best_assign = [0] * n
    best_score = -float("inf")
    scores = {}
    for k in range(2, max_k + 1):
        assign = _agglomerative_assign(dist, k)
        score = _silhouette_score(dist, assign)
        scores[str(k)] = float(score)
        if score > best_score:
            best_score = score
            best_assign = assign
    if best_score < 0:
        best_assign = [0] * n
        best_score = 0.0
    return best_assign, {
        "selected_k": int(max(best_assign) + 1) if best_assign else 0,
        "silhouette": float(best_score),
        "candidates": scores,
    }


def _domain_assign(domains: Sequence[str]) -> Tuple[List[int], Dict[str, str]]:
    domain_to_id: Dict[str, int] = {}
    assign = []
    for d in domains:
        key = str(d or "unknown").lower()
        if key not in domain_to_id:
            domain_to_id[key] = len(domain_to_id)
        assign.append(domain_to_id[key])
    return assign, {str(v): k for k, v in domain_to_id.items()}


def _comb2(n: int) -> float:
    n = int(n)
    return float(n * (n - 1) / 2.0) if n >= 2 else 0.0


def _cluster_quality_against_domains(assign: Sequence[int], domains: Sequence[str]) -> Dict[str, float]:
    """NMI/ARI for predicted B pools against optional domain labels."""
    n = len(assign)
    if n == 0 or n != len(domains):
        return {"domain_nmi": float("nan"), "domain_ari": float("nan")}
    true_labels = [str(d or "unknown").lower() for d in domains]
    pred_labels = [str(int(a)) for a in assign]
    if len(set(true_labels)) <= 1 and "unknown" in set(true_labels):
        return {"domain_nmi": float("nan"), "domain_ari": float("nan")}

    true_counts: Dict[str, int] = defaultdict(int)
    pred_counts: Dict[str, int] = defaultdict(int)
    joint_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for y, c in zip(true_labels, pred_labels):
        true_counts[y] += 1
        pred_counts[c] += 1
        joint_counts[(y, c)] += 1

    def entropy(counts: Dict[str, int]) -> float:
        out = 0.0
        for cnt in counts.values():
            p = float(cnt) / float(n)
            if p > 0:
                out -= p * math.log(p)
        return out

    mi = 0.0
    for (y, c), cnt in joint_counts.items():
        if cnt <= 0:
            continue
        pxy = float(cnt) / float(n)
        py = float(true_counts[y]) / float(n)
        pc = float(pred_counts[c]) / float(n)
        if py > 0 and pc > 0:
            mi += pxy * math.log(pxy / (py * pc))
    h_true = entropy(true_counts)
    h_pred = entropy(pred_counts)
    denom = h_true + h_pred
    nmi = 1.0 if denom <= 1e-12 else float(2.0 * mi / denom)

    sum_joint = sum(_comb2(v) for v in joint_counts.values())
    sum_true = sum(_comb2(v) for v in true_counts.values())
    sum_pred = sum(_comb2(v) for v in pred_counts.values())
    total = _comb2(n)
    if total <= 0:
        ari = 1.0
    else:
        expected = sum_true * sum_pred / total
        max_index = 0.5 * (sum_true + sum_pred)
        ari = 1.0 if abs(max_index - expected) <= 1e-12 else (sum_joint - expected) / (max_index - expected)
    return {"domain_nmi": float(nmi), "domain_ari": float(ari)}


def _cluster_members(assign: Sequence[int]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = defaultdict(list)
    for idx, cid in enumerate(assign):
        out[int(cid)].append(idx)
    return dict(out)


def _weighted_mean_tensors(tensors: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    stacked = torch.stack([t.float() for t in tensors], 0)
    w = torch.tensor(weights, dtype=stacked.dtype).view(
        len(tensors), *([1] * (stacked.ndim - 1))
    )
    return (w * stacked).sum(0)


def _consolidate_B(
    tensors: Sequence[torch.Tensor],
    weights: Sequence[float],
    mode: str,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("empty B tensor list")
    mode = str(mode or "mean").lower()
    if mode == "rep":
        return tensors[0].detach().float().clone()
    if mode == "svd":
        r = int(tensors[0].shape[1])
        stacked = torch.cat([b.detach().float().cpu() for b in tensors], dim=1)
        try:
            u, s, _ = torch.linalg.svd(stacked, full_matrices=False)
            r_eff = min(r, int(s.numel()))
            out = u[:, :r_eff] * s[:r_eff].unsqueeze(0)
            target = torch.stack([b.detach().float().cpu() for b in tensors], 0).norm(
                dim=(1, 2)
            ).mean()
            out = out * (target / out.norm().clamp_min(1e-12))
            if r_eff < r:
                out = torch.cat([out, out.new_zeros(out.shape[0], r - r_eff)], dim=1)
            return out
        except Exception:
            pass
    return _weighted_mean_tensors(tensors, weights)


def _expert_B_states(client_states, b_keys, assign, weights, mode) -> Dict[int, Dict[str, torch.Tensor]]:
    members = _cluster_members(assign)
    experts: Dict[int, Dict[str, torch.Tensor]] = {}
    for expert_id, idxs in sorted(members.items()):
        total_w = float(np.sum([weights[i] for i in idxs]))
        local_w = [float(weights[i] / total_w) if total_w > 0 else 1.0 / len(idxs) for i in idxs]
        experts[expert_id] = {}
        for key in b_keys:
            experts[expert_id][key] = _consolidate_B(
                [client_states[i][key] for i in idxs],
                local_w,
                mode,
            )
    return experts


def _cluster_distances_for_client(dist: np.ndarray, assign: Sequence[int], idx: int) -> Dict[int, float]:
    members = _cluster_members(assign)
    out = {}
    for expert_id, ids in members.items():
        vals = [float(dist[idx, j]) for j in ids if j != idx]
        if not vals and idx in ids:
            out[expert_id] = 0.0
        elif vals:
            out[expert_id] = float(np.mean(vals))
        else:
            out[expert_id] = float(np.mean([dist[idx, j] for j in ids])) if ids else 1.0
    return out


def _blend_experts(
    experts: Dict[int, Dict[str, torch.Tensor]],
    expert_weights: Sequence[Tuple[int, float]],
    b_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in b_keys:
        parts = []
        ws = []
        for expert_id, weight in expert_weights:
            if expert_id in experts and key in experts[expert_id]:
                parts.append(experts[expert_id][key])
                ws.append(float(weight))
        if not parts:
            continue
        s = sum(ws)
        ws = [w / s if s > 0 else 1.0 / len(ws) for w in ws]
        out[key] = _weighted_mean_tensors(parts, ws).detach().cpu()
    return out


def _topm_expert_weights(
    dist: np.ndarray,
    assign: Sequence[int],
    idx: int,
    top_m: int,
    temperature: float,
) -> List[Tuple[int, float]]:
    cdist = _cluster_distances_for_client(dist, assign, idx)
    if not cdist:
        return []
    ordered = sorted(cdist.items(), key=lambda x: (x[1], x[0]))
    top_m = max(1, min(int(top_m), len(ordered)))
    chosen = ordered[:top_m]
    if top_m == 1:
        return [(int(chosen[0][0]), 1.0)]
    temp = max(float(temperature), 1e-6)
    logits = np.asarray([-d / temp for _, d in chosen], dtype=np.float64)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return [(int(chosen[i][0]), float(probs[i])) for i in range(len(chosen))]


def _resolve_assignments(client_states, b_keys, domains, args) -> Tuple[List[int], np.ndarray, Dict[str, object]]:
    mode = str(getattr(args, "expert_cluster_mode", "auto") or "auto").lower()
    n = len(client_states)
    dist, subspace_stats = _b_subspace_distance_matrix(client_states, b_keys)
    if mode in {"domain", "domain_label", "oracle"}:
        assign, label_map = _domain_assign(domains)
        info = {
            "cluster_mode": "domain",
            "selected_k": int(max(assign) + 1) if assign else 0,
            "domain_label_map": label_map,
        }
    elif mode in {"global", "one", "single"}:
        assign = [0] * n
        info = {"cluster_mode": "global", "selected_k": 1}
    elif mode in {"singleton", "local"}:
        assign = list(range(n))
        info = {"cluster_mode": "singleton", "selected_k": n}
    else:
        assign, info = _auto_cluster_assign(dist, args)
        info["cluster_mode"] = "b_subspace_auto"
    info.update(subspace_stats)
    info["cluster_sizes"] = {
        str(k): int(len(v)) for k, v in sorted(_cluster_members(assign).items())
    }
    return assign, dist, info


def _stats_for_assignments(
    algo: str,
    client_ids: Sequence[int],
    domains: Sequence[str],
    assign: Sequence[int],
    info: Dict[str, object],
) -> Dict[str, object]:
    client_clusters = {int(client_ids[i]): int(assign[i]) for i in range(len(client_ids))}
    domain_by_cluster: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i, cid in enumerate(assign):
        domain_by_cluster[int(cid)][str(domains[i])] += 1
    summary = dict(info)
    summary.update(_cluster_quality_against_domains(assign, domains))
    return {
        "algorithm": algo,
        "client_clusters": client_clusters,
        "cluster_domain_hist": {
            str(k): dict(v) for k, v in sorted(domain_by_cluster.items())
        },
        "_summary": summary,
    }


def _aggregate_ecolora(global_model, client_uploads, args):
    global_dict = global_model.state_dict()
    client_states = [M.upload_package_state(m) for m in client_uploads]
    client_ids, domains = _metadata_from_uploads(client_uploads, args)
    weights = _client_weight_vector(client_uploads, args)
    n = len(client_states)
    if n == 0:
        return global_model

    masks = [
        (m.get("sparse_masks", {}) if isinstance(m, dict) else {})
        for m in client_uploads
    ]
    b_keys = _common_lora_keys(client_states, is_lora_b_param_name)
    sent_frac = []
    for key in list(global_dict.keys()):
        if is_lora_a_param_name(key) or is_lora_b_param_name(key):
            if not all(key in st for st in client_states):
                continue
            accum = torch.zeros_like(client_states[0][key].float())
            denom = torch.zeros_like(accum)
            for i in range(n):
                mask = masks[i].get(key, None)
                if mask is None:
                    mask_f = torch.ones_like(accum)
                else:
                    mask_f = mask.to(dtype=accum.dtype)
                    sent_frac.append(float(mask_f.mean().item()))
                w = float(weights[i])
                accum = accum + w * client_states[i][key].float()
                denom = denom + w * mask_f
            prev = global_dict[key].detach().float().cpu()
            merged = torch.where(denom > 0, accum / denom.clamp_min(1e-12), prev)
            global_dict[key] = merged.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
        elif is_task_head_param_name(key) and all(key in st for st in client_states):
            agg = _weighted_mean_tensors([st[key] for st in client_states], weights)
            global_dict[key] = agg.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
    global_model.load_state_dict(global_dict)
    personalized = {
        int(cid): {
            key: global_dict[key].detach().cpu().clone()
            for key in b_keys
            if key in global_dict
        }
        for cid in client_ids
    }
    args._lora_expert_personalized_local_states = personalized
    keep_ratio = _ecolora_keep_ratio(args)
    args._lora_expert_stats = {
        "algorithm": "ecolora",
        "client_clusters": {int(cid): 0 for cid in client_ids},
        "cluster_domain_hist": {"0": {d: domains.count(d) for d in sorted(set(domains))}},
        "_summary": {
            "cluster_mode": "global_sparse",
            "selected_k": 1,
            "ecolora_keep_ratio": keep_ratio,
            "ecolora_mask_mode": str(getattr(args, "ecolora_mask_mode", "round_robin")),
            "mean_sent_fraction": float(np.mean(sent_frac)) if sent_frac else keep_ratio,
        },
    }
    return global_model


def aggregate_models_lora_expert_baseline(global_model, client_uploads, args):
    """Aggregate expert/sparse baselines and materialize per-client B states."""
    raw_algo = _norm_agg(getattr(args, "agg_type", ""))
    algo = _canonical_agg(raw_algo)
    if algo == "ecolora":
        return _aggregate_ecolora(global_model, client_uploads, args)

    global_dict = global_model.state_dict()
    client_states = [M.upload_package_state(m) for m in client_uploads]
    client_ids, domains = _metadata_from_uploads(client_uploads, args)
    weights = _client_weight_vector(client_uploads, args)
    n = len(client_states)
    if n == 0:
        return global_model

    b_keys = _common_lora_keys(client_states, is_lora_b_param_name)
    _aggregate_shared_A_and_heads(global_dict, client_states, weights)

    assign, dist, info = _resolve_assignments(client_states, b_keys, domains, args)
    b_mode = str(getattr(args, "expert_b_mode", "") or "").lower()
    if not b_mode:
        b_mode = str(getattr(args, "v7_b_mode", "mean") or "mean").lower()
    experts = _expert_B_states(client_states, b_keys, assign, weights, b_mode)

    default_top_m = 1
    if algo == "fedlease":
        default_top_m = 2
    top_m = int(getattr(args, "expert_top_m", 0) or default_top_m)
    if algo in {"fedplora_v7", "fedplora_v8", "v7_bsim", "hilora"}:
        top_m = 1
    temperature = float(getattr(args, "expert_router_temperature", 0.2) or 0.2)
    leaf_blend = float(getattr(args, "hilora_leaf_blend", 0.25) or 0.25)

    personalized: Dict[int, Dict[str, torch.Tensor]] = {}
    client_routes: Dict[int, List[Tuple[int, float]]] = {}
    for i, client_id in enumerate(client_ids):
        if algo in {
            "fedplora_v7",
            "fedplora_v8",
            "v7_bsim",
            "hilora",
            "fedplora_v9_mix",
            "fedplora_v9_mix_ab",
            "v9_mix",
            "v9_mix_ab",
            "fedplora_v10_geom_a",
            "fedplora_v10_sketch_a",
            "v10_geom_a",
            "v10_sketch_a",
            "fedplora_v11a_relaxed_a",
            "fedplora_v11a",
            "v11a_relaxed_a",
            "v11a",
            "fedplora_v11c_gmix",
            "fedplora_v11_gmix",
            "v11c_gmix",
            "v11_gmix",
            "fedplora_v12a_sched_gmix",
            "fedplora_v12_sched_gmix",
            "v12a_sched_gmix",
            "v12_sched_gmix",
            "fedplora_v12b_nmi_guard_gmix",
            "fedplora_v12_adaptive_gmix",
            "v12b_nmi_guard_gmix",
            "v12_adaptive_gmix",
            "fedplora_v13a_os",
            "fedplora_v13a_oneshot",
            "v13a_os",
            "v13a_oneshot",
            "fedplora_os",
            "os_alpha100",
            "fedplora_v13b_os_bonly",
            "fedplora_v13b_bonly",
            "v13b_os_bonly",
            "v13b_bonly",
            "fedplora_os_bonly",
            "os_bonly",
        }:
            route = [(int(assign[i]), 1.0)]
        else:
            route = _topm_expert_weights(dist, assign, i, top_m, temperature)
        client_routes[int(client_id)] = [(int(eid), float(w)) for eid, w in route]
        state = _blend_experts(experts, route, b_keys)
        if algo == "hilora":
            lam = min(1.0, max(0.0, leaf_blend))
            for key in b_keys:
                if key in state and key in client_states[i]:
                    state[key] = (
                        (1.0 - lam) * state[key].float()
                        + lam * client_states[i][key].float().cpu()
                    ).detach().cpu()
        personalized[int(client_id)] = state

    for key in b_keys:
        # Keep global_model internally consistent for checkpoint sanity; eval uses personalized B.
        all_b = [personalized[int(cid)][key] for cid in client_ids if key in personalized[int(cid)]]
        if all_b:
            merged = torch.stack([b.float() for b in all_b], 0).mean(0)
            global_dict[key] = merged.to(
                device=global_dict[key].device, dtype=global_dict[key].dtype
            )
    global_model.load_state_dict(global_dict)

    stats = _stats_for_assignments(raw_algo or algo, client_ids, domains, assign, info)
    stats["_summary"].update(
        {
            "b_mode": b_mode,
            "expert_top_m": int(top_m),
            "expert_router_temperature": float(temperature),
            "hilora_leaf_blend": float(leaf_blend) if algo == "hilora" else 0.0,
            "num_b_layers": int(len(b_keys)),
            "upload_scope": "b_only" if _is_b_only_upload(args) else "a_b",
            "freeze_a": bool(_is_b_only_upload(args)),
        }
    )
    stats["client_routes"] = {
        str(cid): [[int(eid), float(w)] for eid, w in route]
        for cid, route in client_routes.items()
    }
    args._lora_expert_personalized_local_states = personalized
    args._lora_expert_stats = stats
    return global_model
