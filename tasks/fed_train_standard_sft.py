"""
Standard (non-cross-domain) federated SFT entry.

Uses a single classic instruction-tuning dataset (default: Stanford Alpaca) with
Dirichlet non-IID client shards (alpha=0.5). Reuses the federated training loop
from fed_train_sft.py but blocks
FedPLoRA-family aggregators (run those via fed_train_sft.py on the cross-domain benchmark).

Usage:
  python tasks/fed_train_standard_sft.py \\
    --model /path/to/Llama-3.1-8B \\
    --benchmark_dir data/standard_benchmark_alpaca_noniid_a0.5/seed_42 \\
    --agg_type normal
"""
import importlib.util
import os
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from utilities.data_utils import (  # noqa: E402
    build_standard_sft_benchmark_from_jsonl,
    load_domain_sft_benchmark,
)
from utilities.utils import (  # noqa: E402
    is_fedplora_agg,
    is_fedplora_oneshot_family_agg,
    is_v4_agg,
    restore_logging,
    setup_run_logging,
)

_STANDARD_ARTIFACT_ROOT = "artifacts_standard"
_DEFAULT_BENCHMARK_DIR = "data/standard_benchmark_alpaca_noniid_a0.5/seed_42"
_DEFAULT_BENCHMARK_OUTPUT = "data/standard_benchmark_alpaca_noniid_a0.5"


def _load_fed_train_sft_module():
    backup = sys.argv[:]
    try:
        sys.argv = [
            backup[0],
            "--model",
            "_import_stub_",
            "--benchmark_dir",
            "_import_stub_",
        ]
        path = _ROOT / "tasks" / "fed_train_sft.py"
        spec = importlib.util.spec_from_file_location("_fed_train_sft_impl", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = backup


def _norm_path_key(path):
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(path or ""))))


def _relocate_standard_artifact_dirs(args, num_clients: int) -> None:
    if num_clients <= 0:
        return
    root = _STANDARD_ARTIFACT_ROOT
    legacy_cs = _norm_path_key("artifacts/domain_client_states")
    legacy_met = _norm_path_key("artifacts/sft_metrics")
    cur_cs = _norm_path_key(args.client_state_dir)
    cur_met = _norm_path_key(args.metrics_output_dir)
    if cur_cs == legacy_cs:
        args.client_state_dir = os.path.join(root, "client_states")
    if cur_met == legacy_met:
        args.metrics_output_dir = os.path.join(root, "sft_metrics")


def build_or_load_standard_benchmark(args):
    if args.build_benchmark:
        if not args.benchmark_jsonl:
            raise ValueError("--build_benchmark requires --benchmark_jsonl")
        num_clients = int(getattr(args, "num_clients", 0) or 0)
        if num_clients <= 0:
            num_clients = 10
        info = build_standard_sft_benchmark_from_jsonl(
            input_path=args.benchmark_jsonl,
            output_dir=getattr(args, "benchmark_output_dir", None) or _DEFAULT_BENCHMARK_OUTPUT,
            num_clients=num_clients,
            min_samples_per_client=args.min_samples_per_client,
            val_ratio=float(getattr(args, "standard_val_ratio", 0.1)),
            test_ratio=float(getattr(args, "standard_test_ratio", 0.1)),
            seed=args.seed,
            domain_label=getattr(args, "standard_domain_label", "alpaca"),
            partition=getattr(args, "standard_partition", "dirichlet"),
            dirichlet_alpha=float(getattr(args, "standard_dirichlet_alpha", 0.5)),
            num_pseudo_labels=int(getattr(args, "standard_num_pseudo_labels", 0) or 0),
            max_samples=int(getattr(args, "standard_max_samples", 0) or 0),
        )
        split_dir = info["split_dir"]
    else:
        if not args.benchmark_dir:
            raise ValueError("provide --benchmark_dir or use --build_benchmark")
        split_dir = args.benchmark_dir
    return load_domain_sft_benchmark(split_dir), split_dir


def _validate_agg_type(agg_type: str) -> None:
    norm = (agg_type or "").strip().lower().replace("-", "_")
    if is_fedplora_agg(norm) or is_fedplora_oneshot_family_agg(norm) or is_v4_agg(norm):
        raise ValueError(
            f"agg_type={agg_type!r} is FedPLoRA/v4-only. "
            "Use tasks/fed_train_sft.py for FedPLoRA on the cross-domain benchmark, "
            "or compare against the allowed baselines here: "
            "normal, flora, flexlora, feddat, yoco, fedsa_lora, fedalt, ffa."
        )


def _apply_standard_defaults(args) -> None:
    if not getattr(args, "benchmark_dir", ""):
        args.benchmark_dir = _DEFAULT_BENCHMARK_DIR
    if not getattr(args, "benchmark_output_dir", ""):
        args.benchmark_output_dir = _DEFAULT_BENCHMARK_OUTPUT
    cs = _norm_path_key(getattr(args, "client_state_dir", ""))
    if cs == _norm_path_key("artifacts/domain_client_states"):
        args.client_state_dir = os.path.join(_STANDARD_ARTIFACT_ROOT, "client_states")
    met = _norm_path_key(getattr(args, "metrics_output_dir", ""))
    if met == _norm_path_key("artifacts/sft_metrics"):
        args.metrics_output_dir = os.path.join(_STANDARD_ARTIFACT_ROOT, "sft_metrics")


def main():
    fts = _load_fed_train_sft_module()
    fts.parser.add_argument(
        "--standard_domain_label",
        type=str,
        default="alpaca",
        help="Pseudo-domain tag when --build_benchmark (single-dataset IID split).",
    )
    fts.parser.add_argument(
        "--standard_val_ratio",
        type=float,
        default=0.1,
        help="Per-client val ratio when --build_benchmark.",
    )
    fts.parser.add_argument(
        "--standard_test_ratio",
        type=float,
        default=0.1,
        help="Global held-out + per-client local-test ratio when --build_benchmark.",
    )
    fts.parser.add_argument(
        "--standard_partition",
        type=str,
        default="dirichlet",
        choices=["iid", "dirichlet"],
        help="Client split when --build_benchmark.",
    )
    fts.parser.add_argument(
        "--standard_dirichlet_alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha when standard_partition=dirichlet.",
    )
    fts.parser.add_argument(
        "--standard_num_pseudo_labels",
        type=int,
        default=0,
        help="KMeans pseudo-label count; 0 => max(10, num_clients).",
    )
    fts.parser.add_argument(
        "--standard_max_samples",
        type=int,
        default=0,
        help="Cap rows before split when --build_benchmark (LW pilot).",
    )

    args = fts.parser.parse_args()
    _apply_standard_defaults(args)
    _validate_agg_type(args.agg_type)

    fts.build_or_load_benchmark = build_or_load_standard_benchmark
    fts._relocate_legacy_artifact_dirs = _relocate_standard_artifact_dirs

    warnings.filterwarnings("ignore")
    fts.set_seed(args.seed)
    log_file, orig_out, orig_err, _ = setup_run_logging(args, filename_prefix="standard_sft")
    try:
        if getattr(args, "eval_only_from_checkpoint", None):
            if args.build_benchmark:
                raise ValueError("--eval_only_from_checkpoint cannot be combined with --build_benchmark")
            fts.eval_only_from_checkpoint(args)
        else:
            fts.federated_sft(args)
    finally:
        restore_logging(log_file, orig_out, orig_err)


if __name__ == "__main__":
    main()
