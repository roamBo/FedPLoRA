import argparse
import json
import os
import sys


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utilities.data_utils import build_standard_sft_benchmark_from_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Build a non-cross-domain federated SFT benchmark from unified JSONL."
    )
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/standard_benchmark_alpaca_noniid_a0.5",
        help="Benchmark output root (creates seed_<seed>/ underneath).",
    )
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--min_samples_per_client", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain_label", type=str, default="alpaca")
    parser.add_argument(
        "--partition",
        type=str,
        default="dirichlet",
        choices=["iid", "dirichlet"],
        help="Client split: iid (round-robin) or dirichlet (pseudo-label skew).",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration when partition=dirichlet (smaller => stronger skew).",
    )
    parser.add_argument(
        "--num_pseudo_labels",
        type=int,
        default=0,
        help="KMeans clusters on TF-IDF prompts; 0 => max(10, num_clients).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap on rows before split (LW pilot). 0 = use all.",
    )
    args = parser.parse_args()

    info = build_standard_sft_benchmark_from_jsonl(
        input_path=args.input_jsonl,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        min_samples_per_client=args.min_samples_per_client,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        domain_label=args.domain_label,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        num_pseudo_labels=args.num_pseudo_labels,
        max_samples=args.max_samples,
    )

    split_dir = info["split_dir"]
    with open(os.path.join(split_dir, "clients.json"), "r", encoding="utf-8") as f:
        clients = json.load(f)
    with open(os.path.join(split_dir, "domain_stats.json"), "r", encoding="utf-8") as f:
        stats = json.load(f)
    with open(os.path.join(split_dir, "partition_info.json"), "r", encoding="utf-8") as f:
        part = json.load(f)

    print(f"[ok] split_dir={split_dir}")
    print(
        f"[ok] partition={info['partition']} "
        f"alpha={info.get('dirichlet_alpha')} "
        f"num_clients={len(clients)} domains={info['domains']}"
    )
    print(f"[ok] partition_info={part}")
    for domain, item in sorted(stats.items()):
        print(
            f"[dataset] {domain}: total={item['n_total']} "
            f"held_out_test={item['n_domain_test']} clients={item['n_clients']}"
        )


if __name__ == "__main__":
    main()
