#!/usr/bin/env python3
"""Compute a per-layer/per-module LoRA-B domain-signal spectrum from states.

This post-processing step is CPU-only. It requires one saved client state per
client and never loads the base language model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from pt_reader import load_state_dict


def client_id_from_path(path: Path) -> int:
    match = re.search(r"client_(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot infer client id from {path}")
    return int(match.group(1))


def basis(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix.astype(np.float64))
    return q


def mean_cosine(left: np.ndarray, right: np.ndarray) -> float:
    values = np.linalg.svd(left.T @ right, compute_uv=False)
    return float(np.clip(values, 0.0, 1.0).mean())


def angle(values: list[float]) -> float:
    return float(np.degrees(np.arccos(np.clip(np.mean(values), -1.0, 1.0))))


def parse_projection(key: str) -> tuple[int | None, str]:
    match = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)_proj", key)
    if not match:
        return None, "unknown"
    return int(match.group(1)), str(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dir", type=Path, required=True)
    parser.add_argument("--clients_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_clients", type=int, default=35)
    args = parser.parse_args()

    files = sorted(args.state_dir.glob("client_*.pt"), key=client_id_from_path)
    if len(files) != args.expected_clients:
        raise SystemExit(
            f"[L1][error] expected {args.expected_clients} states, found {len(files)} in {args.state_dir}"
        )
    manifest = json.loads(args.clients_json.read_text(encoding="utf-8"))
    domain_by_client = {int(row["client_id"]): str(row["domain"]) for row in manifest}
    raw_ids = [client_id_from_path(path) for path in files]
    if not set(raw_ids).issubset(domain_by_client) and set(cid - 1 for cid in raw_ids).issubset(domain_by_client):
        client_ids = [cid - 1 for cid in raw_ids]
    else:
        client_ids = raw_ids
    if not set(client_ids).issubset(domain_by_client):
        raise SystemExit("[L1][error] client-state IDs do not match clients.json")

    states = [load_state_dict(str(path)) for path in files]
    b_keys = sorted(key for key in states[0] if "lora_B" in key)
    if not b_keys:
        raise SystemExit("[L1][error] no lora_B tensors found")
    missing = [key for key in b_keys if any(key not in state for state in states)]
    if missing:
        raise SystemExit(f"[L1][error] inconsistent state keys, first missing={missing[0]}")

    domains = [domain_by_client[cid] for cid in client_ids]
    rng = np.random.default_rng(args.seed)
    rows = []
    similarity_matrices = []
    for key in b_keys:
        matrices = [state[key] for state in states]
        bases = [basis(matrix) for matrix in matrices]
        output_dim, rank = matrices[0].shape
        random_bases = [basis(rng.standard_normal((output_dim, rank))) for _ in bases]
        similarity = np.eye(len(bases), dtype=np.float64)
        intra: list[float] = []
        inter: list[float] = []
        null: list[float] = []
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                value = mean_cosine(bases[i], bases[j])
                similarity[i, j] = similarity[j, i] = value
                (intra if domains[i] == domains[j] else inter).append(value)
                null.append(mean_cosine(random_bases[i], random_bases[j]))
        intra_angle, inter_angle, null_angle = angle(intra), angle(inter), angle(null)
        denominator = null_angle - intra_angle
        ratio = (
            (inter_angle - intra_angle) / denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )
        layer, module = parse_projection(key)
        rows.append(
            {
                "key": key,
                "layer": layer,
                "module": module,
                "intra_angle_deg": intra_angle,
                "inter_angle_deg": inter_angle,
                "null_angle_deg": null_angle,
                "domain_signal_ratio": ratio,
                "output_dim": int(output_dim),
                "rank": int(rank),
            }
        )
        similarity_matrices.append(similarity)

    valid = [row for row in rows if math.isfinite(float(row["domain_signal_ratio"]))]
    modules = sorted({str(row["module"]) for row in valid})
    module_summary = {}
    for module in modules:
        values = [
            float(row["domain_signal_ratio"])
            for row in valid
            if row["module"] == module
        ]
        module_summary[module] = {
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    layers = sorted({int(row["layer"]) for row in valid if row["layer"] is not None})
    layer_summary = {}
    for layer in layers:
        values = [
            float(row["domain_signal_ratio"])
            for row in valid
            if row["layer"] == layer
        ]
        layer_summary[str(layer)] = {
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    payload = {
        "config": {
            "state_dir": str(args.state_dir),
            "clients_json": str(args.clients_json),
            "seed": args.seed,
            "num_clients": len(client_ids),
            "num_projections": len(rows),
        },
        "client_ids": client_ids,
        "domains": domains,
        "rows": rows,
        "module_summary": module_summary,
        "layer_summary": layer_summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_npz,
        client_ids=np.asarray(client_ids, dtype=np.int64),
        domains=np.asarray(domains),
        keys=np.asarray(b_keys),
        similarities=np.stack(similarity_matrices),
        ratios=np.asarray([row["domain_signal_ratio"] for row in rows]),
        layers=np.asarray([row["layer"] if row["layer"] is not None else -1 for row in rows]),
        modules=np.asarray([row["module"] for row in rows]),
    )
    ratios = np.asarray([float(row["domain_signal_ratio"]) for row in valid])
    print(
        f"[L1][done] clients={len(client_ids)} projections={len(rows)} "
        f"ratio_mean={ratios.mean():.4f} ratio_min={ratios.min():.4f} "
        f"ratio_max={ratios.max():.4f} json={args.output_json} npz={args.output_npz}"
    )


if __name__ == "__main__":
    main()
