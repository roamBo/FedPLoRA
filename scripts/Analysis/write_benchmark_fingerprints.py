#!/usr/bin/env python3
"""Write benchmark fingerprints for multiple FedPLoRA split directories.

This is a small 0-GPU helper for paper-table hygiene.  It avoids fragile shell
loops that accidentally write literal filenames such as ``seed_${SEED}.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utilities.benchmark_fingerprint import compute_benchmark_fingerprint


def _resolve_split_dirs(args: argparse.Namespace) -> List[Path]:
    split_dirs: List[Path] = [Path(x) for x in args.split_dirs]
    if args.split_root:
        root = Path(args.split_root)
        for seed in args.seeds:
            split_dirs.append(root / f"seed_{int(seed)}")
    # Preserve order while removing duplicates.
    seen = set()
    out: List[Path] = []
    for item in split_dirs:
        resolved = item.expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def _split_name(split_dir: Path) -> str:
    name = split_dir.name
    if not name:
        raise ValueError(f"cannot derive split name from {split_dir}")
    if "${" in name:
        raise ValueError(f"refusing literal shell-variable split name: {name}")
    return name


def write_fingerprints(split_dirs: Iterable[Path], output_dir: Path) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split_dir in split_dirs:
        if not (split_dir / "clients.json").is_file():
            raise FileNotFoundError(f"missing clients.json under split dir: {split_dir}")
        payload = compute_benchmark_fingerprint(split_dir)
        name = _split_name(split_dir)
        out = output_dir / f"{name}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(
            {
                "split_dir": str(split_dir),
                "output": str(out),
                "combined_sha256": payload.get("combined_sha256", ""),
                "train_rows": (payload.get("split_counts") or {}).get("train"),
                "clients": (payload.get("clients") or {}).get("num_clients"),
            }
        )
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(description="Write fingerprint JSON files for multiple benchmark splits.")
    ap.add_argument("--split_root", type=Path, default=None, help="Directory containing seed_42, seed_43, ...")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44], help="Seeds under --split_root.")
    ap.add_argument("--split_dirs", type=Path, nargs="*", default=[], help="Explicit split directories.")
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()

    split_dirs = _resolve_split_dirs(args)
    if not split_dirs:
        raise SystemExit("provide --split_root or --split_dirs")
    summaries = write_fingerprints(split_dirs, args.output_dir)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
