"""
Default paths for domain SFT run checkpoints (FedPLoRA repo).

Bundle directory name (no timestamp): agg + model basename + benchmark tail + r/e/seed.
Default root: <repo_parent>/trained_models/ (sibling of the FedPLoRA checkout).
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_trained_models_root(repo_root: Path, trained_models_root: str | None) -> Path:
    s = (trained_models_root or "").strip()
    if s:
        return Path(s).expanduser().resolve()
    env = (os.environ.get("TRAINED_MODELS_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root.parent / "trained_models").resolve()


def run_bundle_stem(
    agg_type: str,
    model_path: str,
    benchmark_split_dir: str,
    rounds: int,
    local_epochs: int,
    seed: int,
) -> str:
    p = Path(benchmark_split_dir).expanduser().resolve()
    parts = p.parts
    if len(parts) >= 2:
        split_tag = f"{parts[-2]}_{parts[-1]}"
    else:
        split_tag = parts[-1] if parts else "benchmark"
    model_tag = Path(str(model_path).rstrip("/")).name
    safe_agg = str(agg_type).replace(os.sep, "_").replace(":", "_")
    return f"{safe_agg}_{model_tag}_{split_tag}_r{int(rounds)}_e{int(local_epochs)}_seed{int(seed)}"


def default_save_run_checkpoint_dir(
    repo_root: Path,
    trained_models_root: str | None,
    *,
    agg_type: str,
    model_path: str,
    benchmark_split_dir: str,
    rounds: int,
    local_epochs: int,
    seed: int,
) -> str:
    root = resolve_trained_models_root(repo_root, trained_models_root)
    stem = run_bundle_stem(
        agg_type,
        model_path,
        benchmark_split_dir,
        rounds,
        local_epochs,
        seed,
    )
    return str((root / stem).resolve())


def _cli_print_bundle_dir() -> None:
    import argparse

    pa = argparse.ArgumentParser(description="Print default SFT run checkpoint bundle directory (no training).")
    pa.add_argument("--repo_root", type=str, required=True)
    pa.add_argument("--trained_models_root", type=str, default="")
    pa.add_argument("--agg_type", type=str, required=True)
    pa.add_argument("--model", type=str, required=True)
    pa.add_argument("--benchmark_dir", type=str, required=True)
    pa.add_argument("--rounds", type=int, required=True)
    pa.add_argument("--local_epochs", type=int, required=True)
    pa.add_argument("--seed", type=int, required=True)
    ns = pa.parse_args()
    print(
        default_save_run_checkpoint_dir(
            Path(ns.repo_root).resolve(),
            ns.trained_models_root or None,
            agg_type=ns.agg_type,
            model_path=ns.model,
            benchmark_split_dir=ns.benchmark_dir,
            rounds=ns.rounds,
            local_epochs=ns.local_epochs,
            seed=ns.seed,
        )
    )


if __name__ == "__main__":
    _cli_print_bundle_dir()
