#!/usr/bin/env python3
"""Repartition a benchmark's non-test pool while freezing its shared test.

This is the safe way to construct IID/Dirichlet heterogeneity variants after
the reference benchmark already exists.  The output copies test_domain and
test_global byte-for-byte, combines only reference train/val/test_local into
the repartitionable pool, and never reads test rows into client allocation.
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict

import numpy as np

from build_domain_benchmark_v2 import (
    _domain_seed,
    _load_jsonl,
    _norm,
    _split_groups_by_ratio,
    _subtopic_ids,
    _write_jsonl,
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_set(rows):
    return {_norm(row.get("prompt", "")) for row in rows}


def _allocate_dirichlet(pool_keys, pool_groups, args, domain):
    rng_shard = np.random.default_rng(_domain_seed(args.seed, domain, salt=999))
    representatives = [pool_groups[key][0] for key in pool_keys]
    subtopics = _subtopic_ids(
        representatives, args.subtopic, args.n_subtopics, args.seed
    )
    client_keys = [[] for _ in range(args.num_clients_per_domain)]
    for topic in range(args.n_subtopics):
        topic_keys = [pool_keys[i] for i in range(len(pool_keys)) if subtopics[i] == topic]
        if not topic_keys:
            continue
        rng_shard.shuffle(topic_keys)
        proportions = rng_shard.dirichlet(
            [args.dirichlet_alpha] * args.num_clients_per_domain
        )
        cuts = (np.cumsum(proportions) * len(topic_keys)).astype(int)[:-1]
        chunks = np.split(np.asarray(topic_keys, dtype=object), cuts)
        for client_index, chunk in enumerate(chunks):
            client_keys[client_index].extend(list(chunk))

    def n_rows(keys):
        return sum(len(pool_groups[key]) for key in keys)

    total = sum(n_rows(keys) for keys in client_keys)
    required = args.num_clients_per_domain * args.min_samples_per_client
    if total < required:
        raise RuntimeError(
            f"domain={domain} pool_rows={total} cannot support "
            f"{args.num_clients_per_domain} clients x min_samples={args.min_samples_per_client}"
        )
    guard = 0
    while guard < 1_000_000:
        poor = [
            idx for idx, keys in enumerate(client_keys)
            if n_rows(keys) < args.min_samples_per_client
        ]
        if not poor:
            break
        rich = max(range(len(client_keys)), key=lambda idx: n_rows(client_keys[idx]))
        if not client_keys[rich] or n_rows(client_keys[rich]) <= args.min_samples_per_client:
            break
        client_keys[poor[0]].append(client_keys[rich].pop())
        guard += 1
    sizes = [n_rows(keys) for keys in client_keys]
    if min(sizes, default=0) < args.min_samples_per_client:
        raise RuntimeError(f"domain={domain} floor rebalance failed: {sizes}")
    return client_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_clients_per_domain", type=int, default=5)
    parser.add_argument("--min_samples_per_client", type=int, default=50)
    parser.add_argument("--partition", choices=["iid", "dirichlet"], default="dirichlet")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--subtopic", choices=["length", "hash", "kmeans"], default="kmeans")
    parser.add_argument("--n_subtopics", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    reference = os.path.abspath(os.path.expanduser(args.reference_split))
    required_files = (
        "train.jsonl", "val.jsonl", "test_local.jsonl", "test_domain.jsonl",
        "test_global.jsonl", "clients.json", "domain_stats.json",
    )
    for name in required_files:
        path = os.path.join(reference, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    pool_rows = []
    for name in ("train.jsonl", "val.jsonl", "test_local.jsonl"):
        for row in _load_jsonl(os.path.join(reference, name)):
            clean = dict(row)
            clean.pop("client_id", None)
            pool_rows.append(clean)
    test_domain_rows = _load_jsonl(os.path.join(reference, "test_domain.jsonl"))
    overlap = _prompt_set(pool_rows) & _prompt_set(test_domain_rows)
    if overlap:
        raise RuntimeError(f"reference split itself leaks pool/test prompts: {len(overlap)}")

    by_domain = defaultdict(list)
    for row in pool_rows:
        domain = str(row.get("domain", "") or "").strip()
        prompt = str(row.get("prompt", "") or "").strip()
        response = str(row.get("response", "") or "").strip()
        if domain and prompt and response:
            by_domain[domain].append(row)

    train_rows, val_rows, local_test_rows = [], [], []
    clients, domain_stats = [], {}
    next_client_id = 0
    test_counts = defaultdict(int)
    for row in test_domain_rows:
        test_counts[str(row.get("domain", "?"))] += 1

    for domain in sorted(by_domain):
        groups = defaultdict(list)
        for row in by_domain[domain]:
            groups[_norm(row["prompt"])].append(row)
        # Reference v2 splits are prompt-deduplicated; retain one row if a
        # legacy reference contains duplicates so a prompt stays atomic.
        groups = {key: rows[:1] for key, rows in groups.items()}
        pool_keys = list(groups)
        if args.partition == "iid":
            rng = np.random.default_rng(_domain_seed(args.seed, domain, salt=999))
            rng.shuffle(pool_keys)
            client_keys = [[] for _ in range(args.num_clients_per_domain)]
            for index, key in enumerate(pool_keys):
                client_keys[index % args.num_clients_per_domain].append(key)
        else:
            client_keys = _allocate_dirichlet(pool_keys, groups, args, domain)

        domain_client_count = 0
        for keys in client_keys:
            rng_client = np.random.default_rng(
                _domain_seed(args.seed, f"{domain}:{next_client_id}", salt=7)
            )
            (local_test, local_val), local_train = _split_groups_by_ratio(
                keys, groups, [args.test_ratio, args.val_ratio], rng_client
            )
            if len(local_train) < max(1, args.min_samples_per_client // 2):
                raise RuntimeError(
                    f"domain={domain} client={next_client_id} train too small: {len(local_train)}"
                )

            def add_client_id(rows, destination):
                for row in rows:
                    output = dict(row)
                    output["client_id"] = next_client_id
                    destination.append(output)

            add_client_id(local_train, train_rows)
            add_client_id(local_val, val_rows)
            add_client_id(local_test, local_test_rows)
            clients.append({
                "client_id": next_client_id,
                "domain": domain,
                "n_train": len(local_train),
                "n_val": len(local_val),
                "n_local_test": len(local_test),
            })
            next_client_id += 1
            domain_client_count += 1
        domain_stats[domain] = {
            "n_total": len(groups) + int(test_counts.get(domain, 0)),
            "n_domain_test": int(test_counts.get(domain, 0)),
            "n_clients": domain_client_count,
            "partition": args.partition,
            "dirichlet_alpha": args.dirichlet_alpha if args.partition == "dirichlet" else None,
            "frozen_test_reference": reference,
        }

    expected_clients = len(by_domain) * args.num_clients_per_domain
    if len(clients) != expected_clients:
        raise RuntimeError(f"expected {expected_clients} clients, built {len(clients)}")
    if _prompt_set(train_rows) & _prompt_set(test_domain_rows):
        raise RuntimeError("output train/test prompt leakage")

    split_dir = os.path.join(os.path.abspath(os.path.expanduser(args.output_dir)), f"seed_{args.seed}")
    os.makedirs(split_dir, exist_ok=True)
    _write_jsonl(os.path.join(split_dir, "train.jsonl"), train_rows)
    _write_jsonl(os.path.join(split_dir, "val.jsonl"), val_rows)
    _write_jsonl(os.path.join(split_dir, "test_local.jsonl"), local_test_rows)
    # Byte-for-byte copies make the shared-test assertion unambiguous.
    shutil.copy2(os.path.join(reference, "test_domain.jsonl"), os.path.join(split_dir, "test_domain.jsonl"))
    shutil.copy2(os.path.join(reference, "test_global.jsonl"), os.path.join(split_dir, "test_global.jsonl"))
    with open(os.path.join(split_dir, "clients.json"), "w", encoding="utf-8") as handle:
        json.dump(clients, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(split_dir, "domain_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(domain_stats, handle, ensure_ascii=False, indent=2)
    manifest = {
        "schema_version": 1,
        "reference_split": reference,
        "partition": args.partition,
        "dirichlet_alpha": args.dirichlet_alpha,
        "seed": args.seed,
        "test_domain_sha256": _sha256(os.path.join(split_dir, "test_domain.jsonl")),
        "test_global_sha256": _sha256(os.path.join(split_dir, "test_global.jsonl")),
        "num_clients": len(clients),
        "domain_test_counts": dict(sorted(test_counts.items())),
    }
    with open(os.path.join(split_dir, "frozen_test_repartition_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[frozen-test][ok] output={split_dir} clients={len(clients)}")
    print(f"[frozen-test][ok] reference={reference}")
    print(f"[frozen-test][ok] test_domain_sha256={manifest['test_domain_sha256']}")
    print(f"[frozen-test][ok] domain_test_counts={manifest['domain_test_counts']}")


if __name__ == "__main__":
    main()
