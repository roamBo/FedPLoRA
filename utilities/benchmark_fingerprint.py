"""Benchmark fingerprint utilities for FedPLoRA domain-SFT splits.

The 2026-07-12 audit found two directories with the same logical split name
but different contents.  This module provides a small, dependency-light
fingerprint that can be embedded in run JSON/checkpoints and used by summary
scripts to refuse unsafe merges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FINGERPRINT_FILES = (
    "clients.json",
    "domain_stats.json",
    "train.jsonl",
    "val.jsonl",
    "test_local.jsonl",
    "test_domain.jsonl",
    "test_global.jsonl",
)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_count_by_client(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        if "client_id" in row:
            counts[str(int(row["client_id"]))] += 1
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))


def _row_count_by_domain(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        if "domain" in row:
            counts[str(row["domain"])] += 1
    return dict(sorted(counts.items()))


def _clients_summary(clients: List[Dict[str, Any]]) -> Dict[str, Any]:
    domains = Counter(str(row.get("domain", "")) for row in clients)
    n_train = [int(row.get("n_train", 0) or 0) for row in clients]
    n_val = [int(row.get("n_val", 0) or 0) for row in clients]
    n_local_test = [int(row.get("n_local_test", 0) or 0) for row in clients]
    per_client = {
        str(int(row.get("client_id", idx))): {
            "domain": str(row.get("domain", "")),
            "n_train": int(row.get("n_train", 0) or 0),
            "n_val": int(row.get("n_val", 0) or 0),
            "n_local_test": int(row.get("n_local_test", 0) or 0),
        }
        for idx, row in enumerate(clients)
    }
    return {
        "num_clients": len(clients),
        "domain_counts": dict(sorted(domains.items())),
        "total_n_train_manifest": int(sum(n_train)),
        "total_n_val_manifest": int(sum(n_val)),
        "total_n_local_test_manifest": int(sum(n_local_test)),
        "min_n_train": int(min(n_train)) if n_train else 0,
        "median_n_train": float(sorted(n_train)[len(n_train) // 2]) if n_train else 0.0,
        "max_n_train": int(max(n_train)) if n_train else 0,
        "per_client_manifest": dict(sorted(per_client.items(), key=lambda kv: int(kv[0]))),
    }


def compute_benchmark_fingerprint(split_dir: str | Path, benchmark: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a stable manifest for a domain benchmark split directory."""

    root = Path(split_dir).expanduser().resolve()
    file_sha256 = {name: _sha256_file(root / name) for name in FINGERPRINT_FILES}
    file_sizes = {
        name: int((root / name).stat().st_size) if (root / name).is_file() else None
        for name in FINGERPRINT_FILES
    }

    if benchmark is None:
        clients_path = root / "clients.json"
        clients = json.loads(clients_path.read_text(encoding="utf-8")) if clients_path.is_file() else []
        split_rows = {
            "train": _load_jsonl(root / "train.jsonl"),
            "val": _load_jsonl(root / "val.jsonl"),
            "test_local": _load_jsonl(root / "test_local.jsonl"),
            "test_domain": _load_jsonl(root / "test_domain.jsonl"),
            "test_global": _load_jsonl(root / "test_global.jsonl"),
        }
    else:
        clients = list(benchmark.get("clients") or [])
        split_rows = {
            "train": list(benchmark.get("train") or []),
            "val": list(benchmark.get("val") or []),
            "test_local": list(benchmark.get("test_local") or []),
            "test_domain": list(benchmark.get("test_domain") or []),
            "test_global": list(benchmark.get("test_global") or []),
        }

    split_counts = {}
    by_client = {}
    by_domain = {}
    for split, rows in split_rows.items():
        split_counts[split] = int(len(rows))
        by_client[split] = _row_count_by_client(rows)
        by_domain[split] = _row_count_by_domain(rows)

    combined = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        digest = file_sha256.get(name) or "MISSING"
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("utf-8"))
        combined.update(b"\0")

    payload = {
        "fingerprint_version": 1,
        "split_dir": str(root),
        "combined_sha256": combined.hexdigest(),
        "files_sha256": file_sha256,
        "file_sizes": file_sizes,
        "split_counts": split_counts,
        "clients": _clients_summary(clients),
        "row_counts_by_client": by_client,
        "row_counts_by_domain": by_domain,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    payload = compute_benchmark_fingerprint(args.split_dir)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

