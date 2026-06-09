import argparse
import json
import os
import sys


# Repo root: scripts/DataProcessScripts/this_file -> parents[2]
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utilities.data_utils import build_domain_benchmark_from_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Build a FedPLoRA domain benchmark from unified JSONL."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Unified raw JSONL path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/domain_benchmark",
        help="Benchmark output root.",
    )
    parser.add_argument("--num_clients_per_domain", type=int, default=5)
    parser.add_argument(
        "--per_client_data_fraction",
        type=float,
        default=1.0,
        help="Fraction of each domain's non-test pool assigned to ONE client shard "
        "(use 0.2 for LW: 1 client/domain with 1/5 data, not 5 clients merged).",
    )
    parser.add_argument("--min_samples_per_client", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    info = build_domain_benchmark_from_jsonl(
        input_path=args.input_jsonl,
        output_dir=args.output_dir,
        num_clients_per_domain=args.num_clients_per_domain,
        min_samples_per_client=args.min_samples_per_client,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        per_client_data_fraction=args.per_client_data_fraction,
    )

    split_dir = info["split_dir"]
    stats_path = os.path.join(split_dir, "domain_stats.json")
    clients_path = os.path.join(split_dir, "clients.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    with open(clients_path, "r", encoding="utf-8") as f:
        clients = json.load(f)

    print(f"[ok] split_dir={split_dir}")
    print(f"[ok] num_clients={len(clients)} domains={sorted(stats.keys())}")
    for domain, item in sorted(stats.items()):
        print(
            f"[domain] {domain}: total={item['n_total']} "
            f"domain_test={item['n_domain_test']} clients={item['n_clients']}"
        )


if __name__ == "__main__":
    main()
