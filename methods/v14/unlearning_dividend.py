"""Server-side LoRA-B unlearning-dividend protocol.

This module intentionally lives in a fresh ``methods/v14`` namespace.  It does
not change any training-time FedPLoRA algorithm.  Instead, it materializes the
Phase-0 protocol proposed on 2026-07-30 from already saved client LoRA states:

* ``pool_all``: sample-weighted global LoRA-B pool.
* ``pool_loo``: leave-one-domain re-pooling without retraining.
* ``proj``: projection surgery, ``B <- (I - U_f U_f^T) B``.
* ``task_arith``: negative task-vector baseline, ``B <- B_all - lambda B_f``.
* ``random_proj``: rank-matched random projection control.

The resulting states are exported as FedALT-compatible synthetic checkpoint
bundles.  FedALT is used only as an eval-only container because its client files
can carry a full LoRA A+B state; no FedALT training logic is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence


try:  # Keep --help / static import usable on machines without torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only laptops.
    torch = None  # type: ignore[assignment]


TensorDict = Dict[str, "torch.Tensor"]


def _require_torch():
    if torch is None:  # pragma: no cover - only happens on machines without torch.
        raise RuntimeError(
            "methods.v14.unlearning_dividend requires torch to load/write .pt states. "
            "Run this script in the FedPLoRA training/eval environment."
        )
    return torch


def _clone_state(state: Mapping[str, "torch.Tensor"]) -> TensorDict:
    return {
        str(key): value.detach().cpu().clone()
        for key, value in state.items()
        if hasattr(value, "detach")
    }


def is_lora_a_key(key: str) -> bool:
    return "lora_A" in key and key.endswith("default.weight")


def is_lora_b_key(key: str) -> bool:
    return "lora_B" in key and key.endswith("default.weight")


def b_key_for_a(a_key: str) -> str:
    return a_key.replace("lora_A", "lora_B")


def _safe_tag(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return out.strip("_") or "none"


def _float_tag(value: float) -> str:
    return (f"{float(value):g}").replace("-", "m").replace(".", "p")


def _stable_int(*parts: object) -> int:
    digest = hashlib.sha1("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def client_id_from_path(path: Path) -> int:
    match = re.search(r"client_(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot infer client id from {path}")
    return int(match.group(1))


@dataclass
class SourceStates:
    """Loaded client states plus optional source checkpoint metadata."""

    client_ids: list[int]
    states_by_client: dict[int, TensorDict]
    shared_state: TensorDict
    source_meta: dict = field(default_factory=dict)
    source_path: str = ""


@dataclass
class ArmSpec:
    """A materialized Phase-0 arm before it is saved to disk."""

    tag: str
    arm: str
    forget_domain: str | None
    states_by_client: dict[int, TensorDict]
    shared_state: TensorDict
    metadata: dict = field(default_factory=dict)


@dataclass
class Phase0BuildResult:
    """All generated arms and diagnostics for one source state set."""

    arms: list[ArmSpec]
    summary: dict


def _load_torch_state(path: Path):
    th = _require_torch()
    return th.load(str(path), map_location="cpu")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _state_has_a(state: Mapping[str, object]) -> bool:
    return any(is_lora_a_key(str(key)) for key in state)


def _combine_shared_and_local(shared: Mapping[str, "torch.Tensor"], local: Mapping[str, "torch.Tensor"]) -> TensorDict:
    combined = _clone_state(shared)
    combined.update(_clone_state(local))
    return combined


def _coerce_full_clients(obj) -> list[TensorDict]:
    if isinstance(obj, list):
        return [_clone_state(item) for item in obj]
    if isinstance(obj, tuple):
        return [_clone_state(item) for item in obj]
    if isinstance(obj, dict):
        # Some experimental utilities store {client_id: state}.
        try:
            return [_clone_state(obj[key]) for key in sorted(obj, key=lambda x: int(x))]
        except Exception as exc:
            raise ValueError("unsupported full_clients.pt dict layout") from exc
    raise ValueError(f"unsupported full_clients.pt payload type: {type(obj)!r}")


def load_client_rows(clients_json: str | Path) -> list[dict]:
    path = Path(clients_json).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing clients.json: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"clients.json must contain a list: {path}")
    for row in rows:
        if "client_id" not in row or "domain" not in row:
            raise ValueError(f"clients.json rows require client_id/domain: {path}")
    return rows


def _client_ids_from_manifest(client_rows: Sequence[Mapping[str, object]] | None, n: int) -> list[int]:
    if client_rows and len(client_rows) == n:
        return [int(row["client_id"]) for row in client_rows]
    return list(range(n))


def _resolve_state_ids(raw_ids: Sequence[int], client_rows: Sequence[Mapping[str, object]] | None) -> list[int]:
    ids = [int(x) for x in raw_ids]
    if not client_rows:
        return ids
    manifest_ids = {int(row["client_id"]) for row in client_rows}
    if set(ids).issubset(manifest_ids):
        return ids
    shifted = [x - 1 for x in ids]
    if set(shifted).issubset(manifest_ids):
        return shifted
    if len(ids) == len(client_rows):
        return [int(row["client_id"]) for row in client_rows]
    return ids


def load_source_states(
    *,
    checkpoint_dir: str | Path | None = None,
    state_dir: str | Path | None = None,
    shared_state_path: str | Path | None = None,
    client_rows: Sequence[Mapping[str, object]] | None = None,
) -> SourceStates:
    """Load source client states from a final checkpoint or a raw state dir."""

    if not checkpoint_dir and not state_dir:
        raise ValueError("provide --checkpoint_dir or --state_dir")
    if checkpoint_dir and state_dir:
        raise ValueError("provide only one of --checkpoint_dir / --state_dir")

    shared_state: TensorDict = {}
    source_meta: dict = {}
    source_path = ""

    if checkpoint_dir:
        root = Path(checkpoint_dir).expanduser().resolve()
        source_path = str(root)
        if not root.is_dir():
            raise FileNotFoundError(f"missing checkpoint_dir: {root}")
        source_meta = _load_json(root / "run_checkpoint_meta.json")
        shared_path = root / "global_shared.pt"
        if shared_path.is_file():
            shared_state = _clone_state(_load_torch_state(shared_path))

        full_path = root / "full_clients.pt"
        if full_path.is_file():
            loaded = _coerce_full_clients(_load_torch_state(full_path))
            raw_ids = _client_ids_from_manifest(client_rows, len(loaded))
            client_ids = _resolve_state_ids(raw_ids, client_rows)
            states_by_client = {
                cid: (
                    _combine_shared_and_local(shared_state, state)
                    if shared_state and not _state_has_a(state)
                    else _clone_state(state)
                )
                for cid, state in zip(client_ids, loaded)
            }
            return SourceStates(client_ids, states_by_client, shared_state, source_meta, source_path)

        clients_root = root / "clients"
        if not clients_root.is_dir():
            raise FileNotFoundError(
                f"{root} has neither full_clients.pt nor clients/client_*.pt"
            )
    else:
        clients_root = Path(state_dir).expanduser().resolve()  # type: ignore[arg-type]
        source_path = str(clients_root)
        if not clients_root.is_dir():
            raise FileNotFoundError(f"missing state_dir: {clients_root}")

    if shared_state_path:
        shared_state = _clone_state(_load_torch_state(Path(shared_state_path).expanduser().resolve()))

    files = sorted(clients_root.glob("client_*.pt"), key=client_id_from_path)
    if not files:
        raise FileNotFoundError(f"no client_*.pt found under {clients_root}")

    raw_ids = [client_id_from_path(path) for path in files]
    client_ids = _resolve_state_ids(raw_ids, client_rows)
    local_states = [_clone_state(_load_torch_state(path)) for path in files]
    if not any(_state_has_a(state) for state in local_states) and not _state_has_a(shared_state):
        raise ValueError(
            "source states contain LoRA-B only and no shared LoRA-A was found. "
            "Use --checkpoint_dir that contains global_shared.pt, or pass --shared_state."
        )
    states_by_client = {
        int(cid): (
            _combine_shared_and_local(shared_state, state)
            if shared_state and not _state_has_a(state)
            else _clone_state(state)
        )
        for cid, state in zip(client_ids, local_states)
    }
    return SourceStates(client_ids, states_by_client, shared_state, source_meta, source_path)


def _weights_for_clients(
    client_ids: Sequence[int],
    client_rows: Sequence[Mapping[str, object]],
    mode: str,
) -> dict[int, float]:
    mode = (mode or "sample").strip().lower()
    if mode not in {"sample", "uniform"}:
        raise ValueError("weights mode must be 'sample' or 'uniform'")
    if mode == "uniform":
        return {int(cid): 1.0 / max(len(client_ids), 1) for cid in client_ids}
    by_id = {int(row["client_id"]): row for row in client_rows}
    raw = {}
    for cid in client_ids:
        row = by_id.get(int(cid), {})
        value = row.get("n_train", row.get("num_train", row.get("train_size", 0)))
        raw[int(cid)] = max(float(value or 0.0), 0.0)
    total = sum(raw.values())
    if total <= 0.0:
        return {int(cid): 1.0 / max(len(client_ids), 1) for cid in client_ids}
    return {cid: value / total for cid, value in raw.items()}


def _domain_by_client(client_rows: Sequence[Mapping[str, object]]) -> dict[int, str]:
    return {int(row["client_id"]): str(row["domain"]) for row in client_rows}


def _renormalize(weights: Mapping[int, float], keep: Iterable[int]) -> dict[int, float]:
    keep_set = {int(x) for x in keep}
    total = sum(float(w) for cid, w in weights.items() if int(cid) in keep_set)
    if total <= 0:
        return {cid: 1.0 / max(len(keep_set), 1) for cid in sorted(keep_set)}
    return {cid: float(weights[cid]) / total for cid in sorted(keep_set)}


def _common_keys(states: Sequence[Mapping[str, "torch.Tensor"]], predicate) -> list[str]:
    if not states:
        return []
    keys = [str(k) for k in states[0] if predicate(str(k))]
    return sorted(k for k in keys if all(k in state for state in states))


def _weighted_mean(
    states_by_client: Mapping[int, Mapping[str, "torch.Tensor"]],
    client_weights: Mapping[int, float],
    keys: Sequence[str],
) -> TensorDict:
    th = _require_torch()
    out: TensorDict = {}
    for key in keys:
        acc = None
        ref = None
        for cid, weight in client_weights.items():
            state = states_by_client.get(int(cid))
            if state is None or key not in state:
                continue
            value = state[key].detach().float().cpu()
            ref = state[key]
            part = float(weight) * value
            acc = part if acc is None else acc + part
        if acc is not None and ref is not None:
            out[key] = acc.to(dtype=ref.dtype if hasattr(ref, "dtype") else th.float32)
    return out


def _core_state(
    source: SourceStates,
    client_weights: Mapping[int, float],
) -> tuple[TensorDict, list[str], list[str]]:
    states = [source.states_by_client[cid] for cid in source.client_ids]
    a_keys = _common_keys(states, is_lora_a_key)
    b_keys = _common_keys(states, is_lora_b_key)
    if not b_keys:
        raise ValueError("no common LoRA-B keys found in source states")

    core: TensorDict = {}
    # Prefer final server shared A when available; otherwise average client A.
    for key in sorted(k for k in source.shared_state if is_lora_a_key(k)):
        core[key] = source.shared_state[key].detach().cpu().clone()
    missing_a = [key for key in a_keys if key not in core]
    core.update(_weighted_mean(source.states_by_client, client_weights, missing_a))
    if not any(is_lora_a_key(key) for key in core):
        raise ValueError("no LoRA-A keys available for synthetic deployment state")

    # Preserve any trainable non-B shared tensors if present, e.g. task heads.
    for key, value in source.shared_state.items():
        if not is_lora_b_key(key) and key not in core:
            core[str(key)] = value.detach().cpu().clone()
    return core, sorted(k for k in core if is_lora_a_key(k)), b_keys


def _zero_b_like(b_state: Mapping[str, "torch.Tensor"]) -> TensorDict:
    th = _require_torch()
    return {key: th.zeros_like(value.detach().cpu()) for key, value in b_state.items()}


def _state_from_b(core: Mapping[str, "torch.Tensor"], b_state: Mapping[str, "torch.Tensor"]) -> TensorDict:
    state = _clone_state(core)
    state.update(_clone_state(b_state))
    return state


def _state_nbytes(state: Mapping[str, "torch.Tensor"]) -> int:
    return int(
        sum(int(value.numel()) * int(value.element_size()) for value in state.values())
    )


def _state_l2(state: Mapping[str, "torch.Tensor"], keys: Sequence[str]) -> float:
    total = 0.0
    for key in keys:
        if key in state:
            value = state[key].detach().float()
            total += float((value * value).sum().item())
    return math.sqrt(max(total, 0.0))


def _relative_l2(
    left: Mapping[str, "torch.Tensor"],
    right: Mapping[str, "torch.Tensor"],
    keys: Sequence[str],
) -> float:
    th = _require_torch()
    num = 0.0
    den = 0.0
    for key in keys:
        if key not in left or key not in right:
            continue
        diff = left[key].detach().float() - right[key].detach().float()
        ref = right[key].detach().float()
        num += float((diff * diff).sum().item())
        den += float((ref * ref).sum().item())
    if den <= 0.0:
        return float("nan") if num > 0.0 else 0.0
    return float(th.tensor(num / den).sqrt().item())


def _energy_rank(singular_values: "torch.Tensor", energy_tau: float) -> int:
    if singular_values.numel() == 0:
        return 0
    energy = singular_values.detach().float().square()
    total = float(energy.sum().item())
    if total <= 0.0:
        return 0
    tau = min(1.0, max(0.0, float(energy_tau)))
    cumulative = energy.cumsum(0) / total
    idx = int((cumulative >= tau).nonzero(as_tuple=False)[0].item())
    return max(1, idx + 1)


def _domain_bases(
    states_by_client: Mapping[int, Mapping[str, "torch.Tensor"]],
    forget_clients: Sequence[int],
    b_keys: Sequence[str],
    *,
    energy_tau: float,
    fixed_rank: int | None = None,
) -> tuple[dict[str, "torch.Tensor"], dict[str, int], dict[str, float]]:
    th = _require_torch()
    bases: dict[str, "torch.Tensor"] = {}
    ranks: dict[str, int] = {}
    captured_energy: dict[str, float] = {}
    for key in b_keys:
        mats = [
            states_by_client[int(cid)][key].detach().float().cpu()
            for cid in forget_clients
            if key in states_by_client[int(cid)]
        ]
        if not mats:
            continue
        merged = th.cat(mats, dim=1)
        if merged.numel() == 0:
            continue
        try:
            u, singular, _ = th.linalg.svd(merged, full_matrices=False)
        except RuntimeError:
            u, singular, _ = th.linalg.svd(merged.cpu(), full_matrices=False)
        rank = min(int(fixed_rank), singular.numel()) if fixed_rank is not None else _energy_rank(singular, energy_tau)
        rank = max(0, int(rank))
        if rank <= 0:
            bases[key] = merged.new_zeros((merged.shape[0], 0))
            ranks[key] = 0
            captured_energy[key] = 0.0
            continue
        bases[key] = u[:, :rank].detach().cpu()
        ranks[key] = int(rank)
        total = float(singular.float().square().sum().item())
        got = float(singular[:rank].float().square().sum().item())
        captured_energy[key] = got / total if total > 0.0 else 0.0
    return bases, ranks, captured_energy


def _project_b(
    b_state: Mapping[str, "torch.Tensor"],
    bases: Mapping[str, "torch.Tensor"],
) -> tuple[TensorDict, dict[str, float]]:
    projected: TensorDict = {}
    removed_frac: dict[str, float] = {}
    for key, value in b_state.items():
        b = value.detach().float().cpu()
        u = bases.get(key)
        if u is None or u.numel() == 0:
            projected[key] = value.detach().cpu().clone()
            removed_frac[key] = 0.0
            continue
        component = u @ (u.T @ b)
        out = b - component
        denom = float((b * b).sum().item())
        numer = float((component * component).sum().item())
        removed_frac[key] = numer / denom if denom > 0.0 else 0.0
        projected[key] = out.to(dtype=value.dtype)
    return projected, removed_frac


def _random_bases_like(
    b_state: Mapping[str, "torch.Tensor"],
    ranks: Mapping[str, int],
    *,
    seed: int,
    forget_domain: str,
    trial: int,
) -> dict[str, "torch.Tensor"]:
    th = _require_torch()
    out = {}
    for key, value in b_state.items():
        rank = int(ranks.get(key, 0) or 0)
        if rank <= 0:
            continue
        gen = th.Generator(device="cpu")
        gen.manual_seed((_stable_int(seed, forget_domain, trial, key) % (2**31 - 1)) or 1)
        rand = th.randn((value.shape[0], rank), generator=gen, dtype=th.float32)
        q, _ = th.linalg.qr(rand, mode="reduced")
        out[key] = q[:, :rank].detach().cpu()
    return out


def _mean_of_values(values: Iterable[float]) -> float | None:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else None


def _arm_diag(
    arm_b: Mapping[str, "torch.Tensor"],
    pool_b: Mapping[str, "torch.Tensor"],
    loo_b: Mapping[str, "torch.Tensor"] | None,
    b_keys: Sequence[str],
    extra: Mapping[str, object] | None = None,
) -> dict:
    diag = {
        "b_l2_norm": _state_l2(arm_b, b_keys),
        "rel_l2_to_pool_all": _relative_l2(arm_b, pool_b, b_keys),
    }
    if loo_b is not None:
        diag["rel_l2_to_pool_loo"] = _relative_l2(arm_b, loo_b, b_keys)
    if extra:
        diag.update(dict(extra))
    return diag


def build_phase0_arms(
    source: SourceStates,
    client_rows: Sequence[Mapping[str, object]],
    *,
    forget_domains: Sequence[str] | None = None,
    weight_mode: str = "sample",
    energy_tau: float = 0.90,
    projection_ranks: Sequence[str] = ("auto",),
    task_arith_lambdas: Sequence[float] = (0.5, 1.0),
    random_trials: int = 1,
    include_routed: bool = True,
    seed: int = 42,
) -> Phase0BuildResult:
    """Build all Phase-0 synthetic arms in memory."""

    client_rows = list(client_rows)
    domain_by_id = _domain_by_client(client_rows)
    missing = [cid for cid in source.client_ids if int(cid) not in domain_by_id]
    if missing:
        raise ValueError(f"source clients not present in clients.json: {missing[:10]}")

    weights = _weights_for_clients(source.client_ids, client_rows, weight_mode)
    core, a_keys, b_keys = _core_state(source, weights)
    b_pool_all = _weighted_mean(source.states_by_client, weights, b_keys)
    if not b_pool_all:
        raise ValueError("failed to build pool_all B state")
    all_domains = sorted({domain_by_id[int(cid)] for cid in source.client_ids})
    selected_domains = list(forget_domains or all_domains)
    unknown_domains = sorted(set(selected_domains) - set(all_domains))
    if unknown_domains:
        raise ValueError(f"unknown forget_domains={unknown_domains}; available={all_domains}")

    arms: list[ArmSpec] = []
    arm_summaries: list[dict] = []

    def add_global_arm(tag: str, arm: str, state: TensorDict, meta: dict):
        states = {int(cid): _clone_state(state) for cid in source.client_ids}
        arms.append(ArmSpec(tag=tag, arm=arm, forget_domain=None, states_by_client=states, shared_state=_clone_state(core), metadata=meta))
        arm_summaries.append({"tag": tag, "arm": arm, "forget_domain": None, **meta})

    base_b = _zero_b_like(b_pool_all)
    add_global_arm(
        "global__base",
        "base",
        _state_from_b(core, base_b),
        {"diagnostics": _arm_diag(base_b, b_pool_all, None, b_keys)},
    )
    add_global_arm(
        "global__pool_all",
        "pool_all",
        _state_from_b(core, b_pool_all),
        {"diagnostics": _arm_diag(b_pool_all, b_pool_all, None, b_keys)},
    )

    if include_routed:
        by_domain_clients: dict[str, list[int]] = {dom: [] for dom in all_domains}
        for cid in source.client_ids:
            by_domain_clients[domain_by_id[int(cid)]].append(int(cid))
        b_per_domain = {}
        for dom, cids in by_domain_clients.items():
            b_per_domain[dom] = _weighted_mean(
                source.states_by_client,
                _renormalize(weights, cids),
                b_keys,
            )
        routed_states = {
            int(cid): _state_from_b(core, b_per_domain[domain_by_id[int(cid)]])
            for cid in source.client_ids
        }
        arms.append(
            ArmSpec(
                tag="global__routed_domain",
                arm="routed_domain",
                forget_domain=None,
                states_by_client=routed_states,
                shared_state=_clone_state(core),
                metadata={
                    "diagnostics": {
                        "num_domains": len(b_per_domain),
                        "domain_client_counts": {
                            dom: len(cids) for dom, cids in sorted(by_domain_clients.items())
                        },
                    }
                },
            )
        )
        arm_summaries.append(
            {
                "tag": "global__routed_domain",
                "arm": "routed_domain",
                "forget_domain": None,
                "diagnostics": arms[-1].metadata["diagnostics"],
            }
        )

    for forget_domain in selected_domains:
        forget_clients = [
            int(cid)
            for cid in source.client_ids
            if domain_by_id[int(cid)] == forget_domain
        ]
        keep_clients = [
            int(cid)
            for cid in source.client_ids
            if domain_by_id[int(cid)] != forget_domain
        ]
        if not forget_clients or not keep_clients:
            continue

        b_forget = _weighted_mean(
            source.states_by_client,
            _renormalize(weights, forget_clients),
            b_keys,
        )
        b_loo = _weighted_mean(
            source.states_by_client,
            _renormalize(weights, keep_clients),
            b_keys,
        )
        tag_prefix = f"forget_{_safe_tag(forget_domain)}"

        def add_forget_arm(tag_suffix: str, arm: str, b_state: TensorDict, meta: dict):
            tag = f"{tag_prefix}__{tag_suffix}"
            state = _state_from_b(core, b_state)
            states = {int(cid): _clone_state(state) for cid in source.client_ids}
            payload = {
                "forget_domain": forget_domain,
                "forget_clients": forget_clients,
                "keep_clients": keep_clients,
                **meta,
            }
            arms.append(
                ArmSpec(
                    tag=tag,
                    arm=arm,
                    forget_domain=forget_domain,
                    states_by_client=states,
                    shared_state=_clone_state(core),
                    metadata=payload,
                )
            )
            arm_summaries.append({"tag": tag, "arm": arm, **payload})

        add_forget_arm(
            "pool_loo",
            "pool_loo",
            b_loo,
            {"diagnostics": _arm_diag(b_loo, b_pool_all, b_loo, b_keys)},
        )

        auto_bases, auto_ranks, auto_energy = _domain_bases(
            source.states_by_client,
            forget_clients,
            b_keys,
            energy_tau=energy_tau,
            fixed_rank=None,
        )
        rank_specs = [str(item).strip().lower() for item in projection_ranks if str(item).strip()]
        for rank_spec in rank_specs:
            if rank_spec == "auto":
                bases, ranks, captured_energy = auto_bases, auto_ranks, auto_energy
                rank_label = "auto"
                fixed_rank = None
            else:
                fixed_rank = int(rank_spec)
                bases, ranks, captured_energy = _domain_bases(
                    source.states_by_client,
                    forget_clients,
                    b_keys,
                    energy_tau=energy_tau,
                    fixed_rank=fixed_rank,
                )
                rank_label = f"rank{fixed_rank}"
            b_proj, removed_frac = _project_b(b_pool_all, bases)
            add_forget_arm(
                f"proj_{rank_label}",
                f"proj_{rank_label}",
                b_proj,
                {
                    "projection": {
                        "rank_mode": rank_label,
                        "fixed_rank": fixed_rank,
                        "energy_tau": float(energy_tau),
                        "mean_rank": _mean_of_values(ranks.values()),
                        "min_rank": int(min(ranks.values())) if ranks else None,
                        "max_rank": int(max(ranks.values())) if ranks else None,
                        "mean_captured_energy": _mean_of_values(captured_energy.values()),
                        "mean_removed_pool_all_energy_frac": _mean_of_values(removed_frac.values()),
                    },
                    "diagnostics": _arm_diag(
                        b_proj,
                        b_pool_all,
                        b_loo,
                        b_keys,
                        {"mean_removed_pool_all_energy_frac": _mean_of_values(removed_frac.values())},
                    ),
                },
            )
            for trial in range(max(0, int(random_trials))):
                random_bases = _random_bases_like(
                    b_pool_all,
                    ranks,
                    seed=int(seed),
                    forget_domain=forget_domain,
                    trial=trial,
                )
                b_random, random_removed = _project_b(b_pool_all, random_bases)
                add_forget_arm(
                    f"random_proj_{rank_label}_trial{trial}",
                    f"random_proj_{rank_label}",
                    b_random,
                    {
                        "random_projection": {
                            "rank_mode": rank_label,
                            "trial": int(trial),
                            "mean_rank": _mean_of_values(ranks.values()),
                            "mean_removed_pool_all_energy_frac": _mean_of_values(random_removed.values()),
                        },
                        "diagnostics": _arm_diag(
                            b_random,
                            b_pool_all,
                            b_loo,
                            b_keys,
                            {"mean_removed_pool_all_energy_frac": _mean_of_values(random_removed.values())},
                        ),
                    },
                )

        for lam in task_arith_lambdas:
            b_task = {
                key: (
                    b_pool_all[key].detach().float().cpu()
                    - float(lam) * b_forget[key].detach().float().cpu()
                ).to(dtype=b_pool_all[key].dtype)
                for key in b_keys
                if key in b_pool_all and key in b_forget
            }
            add_forget_arm(
                f"task_arith_lam{_float_tag(float(lam))}",
                "task_arith",
                b_task,
                {
                    "task_arith_lambda": float(lam),
                    "diagnostics": _arm_diag(b_task, b_pool_all, b_loo, b_keys),
                },
            )

    summary = {
        "schema_version": 1,
        "source_path": source.source_path,
        "num_clients": len(source.client_ids),
        "client_ids": [int(x) for x in source.client_ids],
        "domains": all_domains,
        "domain_client_counts": {
            dom: sum(1 for cid in source.client_ids if domain_by_id[int(cid)] == dom)
            for dom in all_domains
        },
        "weight_mode": weight_mode,
        "weight_min": min(weights.values()) if weights else None,
        "weight_max": max(weights.values()) if weights else None,
        "num_lora_a_keys": len(a_keys),
        "num_lora_b_keys": len(b_keys),
        "core_state_tensor_bytes": _state_nbytes(core),
        "pool_all_b_l2_norm": _state_l2(b_pool_all, b_keys),
        "forget_domains": selected_domains,
        "arms": arm_summaries,
    }
    return Phase0BuildResult(arms=arms, summary=summary)


def _default_meta_value(source_meta: Mapping[str, object], key: str, default):
    value = source_meta.get(key)
    return default if value in (None, "") else value


def _synthetic_meta(
    *,
    source: SourceStates,
    arm: ArmSpec,
    benchmark_dir: str,
    model: str,
    seed: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    rslora: bool,
    target_modules: str,
    torch_dtype: str,
    trust_remote_code: bool,
    gradient_checkpointing: bool,
) -> dict:
    source_meta = source.source_meta or {}
    effective = dict(source_meta.get("effective_hparams") or {})
    effective.update(
        {
            "synthetic_unlearning_phase0": True,
            "source_checkpoint_or_state_dir": source.source_path,
            "source_agg_type": source_meta.get("agg_type", ""),
        }
    )
    return {
        "run_checkpoint_version": 1,
        "checkpoint_phase": "final",
        "saved_after_aggregation_before_eval": False,
        "bundle_stem": f"v14_unlearning_phase0_{arm.tag}",
        # FedALT is an eval-only carrier for full A+B per-client state files.
        "agg_type": "fedalt",
        "model": str(model or _default_meta_value(source_meta, "model", "")),
        "benchmark_dir": str(benchmark_dir or _default_meta_value(source_meta, "benchmark_dir", "")),
        "seed": int(seed),
        "num_clients": int(len(source.client_ids)),
        "client_ids": [int(x) for x in source.client_ids],
        "disk_sequential_protocol": True,
        "use_ffa_peft": False,
        "train_rounds": int(_default_meta_value(source_meta, "train_rounds", 1)),
        "train_local_epochs": int(_default_meta_value(source_meta, "train_local_epochs", 1)),
        "lora_r": int(_default_meta_value(source_meta, "lora_r", lora_r)),
        "lora_alpha": int(_default_meta_value(source_meta, "lora_alpha", lora_alpha)),
        "lora_dropout": float(_default_meta_value(source_meta, "lora_dropout", lora_dropout)),
        "rslora": bool(_default_meta_value(source_meta, "rslora", rslora)),
        "target_modules": str(_default_meta_value(source_meta, "target_modules", target_modules)),
        "torch_dtype": str(_default_meta_value(source_meta, "torch_dtype", torch_dtype)),
        "trust_remote_code": bool(_default_meta_value(source_meta, "trust_remote_code", trust_remote_code)),
        "gradient_checkpointing": bool(_default_meta_value(source_meta, "gradient_checkpointing", gradient_checkpointing)),
        "metrics_path": "",
        "benchmark_fingerprint": source_meta.get("benchmark_fingerprint", {}),
        "effective_hparams": effective,
        "unlearning_phase0": {
            "schema_version": 1,
            "tag": arm.tag,
            "arm": arm.arm,
            "forget_domain": arm.forget_domain,
            "source_path": source.source_path,
            "state_container": "fedalt_eval_only_full_lora_ab",
            **arm.metadata,
        },
    }


def save_phase0_checkpoints(
    result: Phase0BuildResult,
    *,
    output_dir: str | Path,
    source: SourceStates,
    benchmark_dir: str,
    model: str,
    seed: int,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    rslora: bool = False,
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    gradient_checkpointing: bool = True,
    symlink_replicated_clients: bool = True,
    force: bool = False,
) -> dict:
    """Write synthetic FedALT-compatible checkpoint bundles to ``output_dir``."""

    th = _require_torch()
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"output_dir exists and is not empty: {root}; pass --force to overwrite")
    if root.exists() and force:
        shutil.rmtree(root)
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "created_at_unix": int(time.time()),
        "output_dir": str(root),
        "summary": result.summary,
        "checkpoints": [],
    }
    for arm in result.arms:
        arm_root = ckpt_root / arm.tag
        clients_root = arm_root / "clients"
        clients_root.mkdir(parents=True, exist_ok=True)
        th.save(_clone_state(arm.shared_state), arm_root / "global_shared.pt")
        if symlink_replicated_clients and arm.arm != "routed_domain":
            template_name = "client_template.pt"
            th.save(
                _clone_state(next(iter(arm.states_by_client.values()))),
                arm_root / template_name,
            )
            for cid in sorted(arm.states_by_client):
                dst = clients_root / f"client_{int(cid):03d}.pt"
                rel = Path("..") / template_name
                try:
                    dst.symlink_to(rel)
                except OSError:
                    shutil.copy2(arm_root / template_name, dst)
        else:
            for cid in sorted(arm.states_by_client):
                th.save(_clone_state(arm.states_by_client[cid]), clients_root / f"client_{int(cid):03d}.pt")
        meta = _synthetic_meta(
            source=source,
            arm=arm,
            benchmark_dir=benchmark_dir,
            model=model,
            seed=seed,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            rslora=rslora,
            target_modules=target_modules,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            gradient_checkpointing=gradient_checkpointing,
        )
        (arm_root / "run_checkpoint_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (arm_root / "checkpoint_ok.json").write_text(
            json.dumps(
                {"ok": True, "checkpoint_phase": "final", "saved_at_unix": int(time.time())},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (arm_root / "unlearning_phase0_meta.json").write_text(
            json.dumps(meta["unlearning_phase0"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["checkpoints"].append(
            {
                "tag": arm.tag,
                "arm": arm.arm,
                "forget_domain": arm.forget_domain,
                "checkpoint_dir": str(arm_root),
                "num_clients": len(arm.states_by_client),
                "tensor_bytes_per_client_mean": _mean_of_values(
                    _state_nbytes(state) for state in arm.states_by_client.values()
                ),
                "client_storage": (
                    "one_template_plus_relative_symlinks"
                    if symlink_replicated_clients and arm.arm != "routed_domain"
                    else "one_pt_file_per_client"
                ),
                "metadata": arm.metadata,
            }
        )

    (root / "phase0_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
