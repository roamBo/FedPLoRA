#!/usr/bin/env python3
"""Build a lightweight manifest for copied FedPLoRA result bundles.

The goal is to catch the recurring "JSON copied but run logs/gates missing" and
"fingerprint output overwritten" issues before paper-table analysis.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


FAIL_RE = re.compile(
    r"Traceback|ModuleNotFoundError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|RuntimeError|\[[^]]+\]\[error\]",
    re.IGNORECASE,
)


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _scan_failures(log_files: List[Path]) -> List[Dict[str, Any]]:
    failures = []
    for path in log_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            failures.append({"path": str(path), "error": f"read_error:{exc!r}"})
            continue
        matches = []
        for i, line in enumerate(text.splitlines(), start=1):
            if FAIL_RE.search(line):
                matches.append({"line": i, "text": line[:500]})
                if len(matches) >= 20:
                    break
        if matches:
            failures.append({"path": str(path), "matches": matches})
    return failures


def _fingerprint_conflicts(train_jsons: List[Path]) -> Dict[str, List[str]]:
    by_split: Dict[str, set] = defaultdict(set)
    for path in train_jsons:
        data = _load_json(path)
        if not data:
            continue
        fp = data.get("benchmark_fingerprint") or {}
        sha = fp.get("combined_sha256")
        bench_dir = data.get("benchmark_dir") or (data.get("args") or {}).get("benchmark_dir")
        split = Path(str(bench_dir)).name if bench_dir else ""
        if split and sha:
            by_split[split].add(str(sha))
    return {
        split: sorted(vals)
        for split, vals in sorted(by_split.items())
        if len(vals) > 1
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Result root to scan recursively.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = args.root
    json_files = sorted(root.rglob("*.json"))
    log_files = sorted(root.rglob("*.log"))
    train_jsons = []
    personalized_jsons = []
    other_jsons = []
    parse_errors = []
    fingerprints = []
    literal_seed_files = []
    gate_files = []

    for path in json_files:
        if "${" in path.name:
            literal_seed_files.append(str(path))
        if "gates" in path.parts:
            gate_files.append(str(path))
        data = _load_json(path)
        if data is None:
            parse_errors.append(str(path))
            continue
        if "benchmark_fingerprint" in data:
            fp = data.get("benchmark_fingerprint") or {}
            fingerprints.append({
                "path": str(path),
                "split_dir": fp.get("split_dir"),
                "sha": fp.get("combined_sha256"),
                "train_rows": (fp.get("split_counts") or {}).get("train"),
                "clients": (fp.get("clients") or {}).get("num_clients"),
            })
        if "rounds" in data:
            train_jsons.append(path)
        elif "results" in data:
            personalized_jsons.append(path)
        else:
            other_jsons.append(path)

    run_roots = sorted({
        str(path.parent.parent.parent)
        for path in train_jsons
        if path.parent.parent.name == "result_logs"
    })
    missing_run_logs = []
    for run_root in run_roots:
        rr = Path(run_root)
        if not (rr / "run_logs").exists():
            missing_run_logs.append(run_root)

    payload = {
        "root": str(root),
        "json_files": len(json_files),
        "log_files": len(log_files),
        "train_jsons": len(train_jsons),
        "personalized_jsons": len(personalized_jsons),
        "other_jsons": len(other_jsons),
        "parse_errors": parse_errors,
        "failure_markers": _scan_failures(log_files),
        "fingerprint_records": fingerprints,
        "fingerprint_conflicts_by_split": _fingerprint_conflicts(train_jsons),
        "literal_seed_json_files": literal_seed_files,
        "gate_files": gate_files,
        "run_roots": run_roots,
        "run_roots_missing_run_logs": missing_run_logs,
        "ok": True,
        "reasons": [],
    }
    if parse_errors:
        payload["ok"] = False
        payload["reasons"].append("parse_errors")
    if payload["failure_markers"]:
        payload["ok"] = False
        payload["reasons"].append("failure_markers")
    if payload["fingerprint_conflicts_by_split"]:
        payload["ok"] = False
        payload["reasons"].append("fingerprint_conflicts")
    if literal_seed_files:
        payload["ok"] = False
        payload["reasons"].append("literal_seed_json_files")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "ok": payload["ok"],
        "json_files": payload["json_files"],
        "log_files": payload["log_files"],
        "reasons": payload["reasons"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
