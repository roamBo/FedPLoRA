#!/usr/bin/env python3
"""Download lm-eval benchmark datasets into the repo-local HF cache.

The cache must be built with the same ``datasets`` version as ``fedplora`` on gb
(see requirements.txt, currently 2.20.0). Caches produced by datasets 3.x/4.x
will fail offline with ``TypeError: must be called with a dataclass type``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "external_lm_eval_hf_cache"
REQUIRED_DATASETS_VERSION = "2.20.0"
TASK_DATASETS = {
    "mmlu": ("cais/mmlu", "all"),
    "pubmedqa": ("pubmed_qa", "pqa_labeled"),
    "mbpp": ("google-research-datasets/mbpp", "full"),
}


def _cache_slug(name: str) -> str:
    return name.replace("/", "___")


def _offline_env(cache_root: Path) -> dict[str, str]:
    datasets_cache = cache_root / "datasets"
    return {
        "HF_HOME": str(cache_root),
        "HF_DATASETS_CACHE": str(datasets_cache),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _assert_datasets_version() -> None:
    import datasets

    version = datasets.__version__
    if version != REQUIRED_DATASETS_VERSION:
        raise SystemExit(
            "[hf-cache][error] datasets version mismatch: "
            f"have {version}, need {REQUIRED_DATASETS_VERSION}\n"
            f"Fix: python -m pip install 'datasets=={REQUIRED_DATASETS_VERSION}'"
        )


def _download(task: str, name: str, config: str, cache_root: Path) -> None:
    from datasets import load_dataset

    datasets_cache = cache_root / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(key, None)

    print(f"[hf-cache][download] task={task} dataset={name!r} config={config!r}", flush=True)
    ds = load_dataset(name, config, trust_remote_code=False)
    print(f"[hf-cache][download][ok] {task} splits={list(ds.keys())}", flush=True)


def _verify_offline(task: str, name: str, config: str, cache_root: Path) -> None:
    from datasets import load_dataset

    env = _offline_env(cache_root)
    backup = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        print(f"[hf-cache][verify-offline] task={task}", flush=True)
        ds = load_dataset(name, config, trust_remote_code=False)
        print(f"[hf-cache][verify-offline][ok] {task} splits={list(ds.keys())}", flush=True)
    finally:
        for key, old in backup.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _remove_task_cache(task: str, cache_root: Path) -> None:
    name, config = TASK_DATASETS[task]
    slug = _cache_slug(name)
    target = cache_root / "datasets" / slug
    if target.is_dir():
        print(f"[hf-cache][purge] {target}", flush=True)
        shutil.rmtree(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_root",
        default=str(DEFAULT_CACHE_ROOT),
        help=f"Cache root (default: {DEFAULT_CACHE_ROOT})",
    )
    parser.add_argument(
        "--tasks",
        default="mmlu,pubmedqa,mbpp",
        help="Comma-separated subset of mmlu,pubmedqa,mbpp",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Delete existing cached task dirs before downloading.",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only run offline verification; do not download.",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = sorted(set(tasks) - set(TASK_DATASETS))
    if unknown:
        raise SystemExit(f"[hf-cache][error] unknown tasks: {unknown}")

    _assert_datasets_version()
    import datasets

    print(f"[hf-cache] datasets={datasets.__version__} cache_root={cache_root}", flush=True)

    if not args.verify_only:
        cache_root.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            if args.purge:
                _remove_task_cache(task, cache_root)
            name, config = TASK_DATASETS[task]
            _download(task, name, config, cache_root)

    for task in tasks:
        name, config = TASK_DATASETS[task]
        _verify_offline(task, name, config, cache_root)

    print("[hf-cache][ok] offline cache ready", flush=True)


if __name__ == "__main__":
    main()
