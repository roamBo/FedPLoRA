#!/usr/bin/env python3
"""Derive a mixed-richness FedPLoRA benchmark from an existing split.

The script keeps validation and test files unchanged and only subsamples
``train.jsonl`` per client.  This creates a controlled "rich/poor clients"
regime without moving the evaluation target.

Example:
  python scripts/DataProcessScripts/build_mixed_richness_benchmark.py \
    --input_benchmark_dir data/domain_benchmark_35c_dir05/seed_42 \
    --output_dir data/domain_benchmark_35c_dir05_mixrich \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _split_output_dir(output_dir: Path, seed: int) -> Path:
    if output_dir.name.startswith("seed_"):
        return output_dir
    return output_dir / f"seed_{int(seed)}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a rich/poor-client benchmark by subsampling train.jsonl."
    )
    ap.add_argument("--input_benchmark_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--rich_per_domain",
        type=int,
        default=2,
        help="Number of rich clients kept per domain.",
    )
    ap.add_argument(
        "--rich_cap",
        type=int,
        default=0,
        help="Max train rows for rich clients. 0 keeps all available rows.",
    )
    ap.add_argument("--poor_min", type=int, default=50)
    ap.add_argument("--poor_max", type=int, default=100)
    ap.add_argument(
        "--shuffle_rich_clients",
        action="store_true",
        help="If set, choose rich clients by seeded shuffle; default uses lowest client_ids per domain.",
    )
    args = ap.parse_args()

    in_dir = args.input_benchmark_dir
    out_dir = _split_output_dir(args.output_dir, args.seed)
    if not (in_dir / "train.jsonl").is_file():
        raise SystemExit(f"missing train.jsonl under {in_dir}")
    if not (in_dir / "clients.json").is_file():
        raise SystemExit(f"missing clients.json under {in_dir}")

    rng = random.Random(int(args.seed))
    train_rows = _read_jsonl(in_dir / "train.jsonl")
    clients = json.loads((in_dir / "clients.json").read_text(encoding="utf-8"))

    domain_to_clients: Dict[str, List[int]] = defaultdict(list)
    for c in clients:
        domain_to_clients[str(c.get("domain", "unknown"))].append(int(c["client_id"]))
    for domain in domain_to_clients:
        domain_to_clients[domain] = sorted(domain_to_clients[domain])

    rich_clients = set()
    for domain, cids in sorted(domain_to_clients.items()):
        candidates = list(cids)
        if bool(args.shuffle_rich_clients):
            rng.shuffle(candidates)
        rich_clients.update(candidates[: max(0, int(args.rich_per_domain))])

    rows_by_client: Dict[int, List[dict]] = defaultdict(list)
    for row in train_rows:
        rows_by_client[int(row["client_id"])].append(row)

    kept_rows = []
    new_train_counts: Dict[int, int] = {}
    richness_by_client: Dict[int, str] = {}
    for c in clients:
        cid = int(c["client_id"])
        rows = list(rows_by_client.get(cid, []))
        rng_client = random.Random(int(args.seed) * 1000003 + cid)
        rng_client.shuffle(rows)
        if cid in rich_clients:
            group = "rich"
            cap = int(args.rich_cap or 0)
            keep_n = len(rows) if cap <= 0 else min(len(rows), max(1, cap))
        else:
            group = "poor"
            lo = max(1, int(args.poor_min))
            hi = max(lo, int(args.poor_max))
            keep_n = min(len(rows), rng_client.randint(lo, hi))
        for row in rows[:keep_n]:
            row = dict(row)
            row["richness_group"] = group
            kept_rows.append(row)
        new_train_counts[cid] = int(keep_n)
        richness_by_client[cid] = group

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", kept_rows)
    for name in ("val.jsonl", "test_local.jsonl", "test_domain.jsonl", "test_global.jsonl"):
        _copy_if_exists(in_dir / name, out_dir / name)

    new_clients = []
    for c in clients:
        row = dict(c)
        cid = int(row["client_id"])
        row["n_train_original"] = int(row.get("n_train", new_train_counts.get(cid, 0)) or 0)
        row["n_train"] = int(new_train_counts.get(cid, 0))
        row["richness_group"] = richness_by_client.get(cid, "unknown")
        new_clients.append(row)
    (out_dir / "clients.json").write_text(
        json.dumps(new_clients, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    old_stats = {}
    if (in_dir / "domain_stats.json").is_file():
        old_stats = json.loads((in_dir / "domain_stats.json").read_text(encoding="utf-8"))
    train_counts_by_domain = Counter()
    group_counts_by_domain = defaultdict(Counter)
    for c in new_clients:
        d = str(c.get("domain", "unknown"))
        train_counts_by_domain[d] += int(c.get("n_train", 0) or 0)
        group_counts_by_domain[d][str(c.get("richness_group", "unknown"))] += 1
    new_stats = {}
    for d in sorted(domain_to_clients):
        item = dict(old_stats.get(d, {}))
        item["n_clients"] = len(domain_to_clients[d])
        item["n_train"] = int(train_counts_by_domain[d])
        item["richness_groups"] = dict(group_counts_by_domain[d])
        item["mixed_richness_source"] = str(in_dir)
        new_stats[d] = item
    (out_dir / "domain_stats.json").write_text(
        json.dumps(new_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "source": str(in_dir),
        "output": str(out_dir),
        "seed": int(args.seed),
        "rich_per_domain": int(args.rich_per_domain),
        "rich_cap": int(args.rich_cap or 0),
        "poor_min": int(args.poor_min),
        "poor_max": int(args.poor_max),
        "num_clients": len(new_clients),
        "total_train_rows": len(kept_rows),
        "group_counts": dict(Counter(richness_by_client.values())),
    }
    (out_dir / "mixed_richness_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] mixed-richness split: {out_dir}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

