#!/usr/bin/env python3
"""Build an eval-only benchmark that preserves source clients/train weights.

The output keeps train/val/test_local/clients/domain_stats from ``source_split``
and replaces only test_domain/test_global with ``common_test_split``.  This is
important because eval-only aggregation uses source client sizes.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path


SOURCE_FILES = ("train.jsonl", "val.jsonl", "test_local.jsonl", "clients.json", "domain_stats.json")
TEST_FILES = ("test_domain.jsonl", "test_global.jsonl")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_count(path):
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _example_key(row):
    source_id = str(row.get("source_id", "") or "").strip()
    if source_id:
        return f"source_id:{source_id}"
    prompt = " ".join(str(row.get("prompt", "") or "").split())
    return "prompt_sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _jsonl_keys(path):
    keys = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                keys.add(_example_key(json.loads(line)))
            except Exception as exc:
                raise SystemExit(f"[common-test][error] invalid JSONL {path}:{line_no}: {exc}")
    return keys


def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise SystemExit(f"[common-test][error] invalid JSONL {path}:{line_no}: {exc}")
    return rows


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_split", required=True)
    parser.add_argument("--common_test_split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--test_selection",
        choices=["full", "intersection"],
        default="full",
        help="full uses all rows from common_test_split; intersection keeps only rows that "
        "are also in source_split's original test, guaranteeing they were held out by both.",
    )
    parser.add_argument(
        "--min_test_rows_per_domain",
        type=int,
        default=20,
        help="Hard floor for intersection mode; prevents publishing an unusably tiny common test.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_split).expanduser().resolve()
    common = Path(args.common_test_split).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for root, names in ((source, SOURCE_FILES), (common, TEST_FILES)):
        for name in names:
            path = root / name
            if not path.is_file():
                raise SystemExit(f"[common-test][error] missing {path}")
    selected_tests = {}
    for name in TEST_FILES:
        rows = _load_jsonl(common / name)
        if args.test_selection == "intersection":
            source_test_keys = _jsonl_keys(source / name)
            rows = [row for row in rows if _example_key(row) in source_test_keys]
        selected_tests[name] = rows

    train_side_keys = _jsonl_keys(source / "train.jsonl") | _jsonl_keys(source / "val.jsonl")
    common_test_keys = {
        _example_key(row) for rows in selected_tests.values() for row in rows
    }
    overlap = sorted(train_side_keys & common_test_keys)
    if overlap:
        preview = "\n".join(overlap[:10])
        raise SystemExit(
            f"[common-test][error] source train/val intersects common test: {len(overlap)} examples.\n"
            f"{preview}\nFor a leakage-free eval-only salvage, use --test_selection intersection "
            "and re-evaluate every compared split on that same intersection. For a full-size test, "
            "rebuild all heterogeneity splits from one frozen shared test set and retrain."
        )
    domain_counts = Counter(
        str(row.get("domain", "?")) for row in selected_tests["test_domain.jsonl"]
    )
    if args.test_selection == "intersection":
        if not domain_counts:
            raise SystemExit("[common-test][error] empty test intersection")
        too_small = {
            domain: count for domain, count in domain_counts.items()
            if count < int(args.min_test_rows_per_domain)
        }
        if too_small:
            raise SystemExit(
                f"[common-test][error] intersection below per-domain floor "
                f"{args.min_test_rows_per_domain}: {too_small}. Rebuild from a frozen shared test and retrain."
            )
    if output.exists() and not args.force:
        raise SystemExit(f"[common-test][error] output exists: {output}; use --force")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    try:
        provenance = {
            "schema_version": 1,
            "purpose": "eval-only common-test benchmark",
            "source_split": str(source),
            "common_test_split": str(common),
            "test_selection": args.test_selection,
            "prompt_level_leakage_count": 0,
            "test_domain_counts": dict(sorted(domain_counts.items())),
            "files": {},
        }
        for name in SOURCE_FILES:
            shutil.copy2(source / name, tmp / name)
            provenance["files"][name] = {
                "role": "source_train_side",
                "sha256": _sha256(tmp / name),
            }
        for name in TEST_FILES:
            _write_jsonl(tmp / name, selected_tests[name])
            provenance["files"][name] = {
                "role": f"common_test_side:{args.test_selection}",
                "sha256": _sha256(tmp / name),
                "rows": _jsonl_count(tmp / name),
            }
        with open(tmp / "common_test_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2, sort_keys=True)
        if output.exists():
            shutil.rmtree(output)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)

    print(f"[common-test][ok] output={output}")
    print(f"[common-test][ok] source_train={source}")
    print(f"[common-test][ok] common_test={common}")
    print(f"[common-test][ok] test_selection={args.test_selection} domain_counts={dict(sorted(domain_counts.items()))}")
    print(f"[common-test][ok] test_domain_sha256={provenance['files']['test_domain.jsonl']['sha256']}")


if __name__ == "__main__":
    main()
