#!/usr/bin/env python3
"""Download lm-eval benchmark datasets into the repo-local HF cache.

Cache specs are read from the installed lm_eval task YAMLs (0.4.x), e.g.:
  pubmedqa -> bigbio/pubmed_qa (NOT pubmed_qa)
  mmlu     -> cais/mmlu per subject config (NOT config=all)

Must be built with datasets==2.20.0 (see requirements.txt).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from external_lm_eval_datasets import (  # noqa: E402
    download_task_cache,
    purge_task_cache,
    read_lm_eval_dataset_spec,
    verify_task_cache_offline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "external_lm_eval_hf_cache"
REQUIRED_DATASETS_VERSION = "2.20.0"
ALL_TASKS = ("mmlu", "pubmedqa", "mbpp")


def _assert_datasets_version() -> None:
    import datasets

    version = datasets.__version__
    if version != REQUIRED_DATASETS_VERSION:
        raise SystemExit(
            "[hf-cache][error] datasets version mismatch: "
            f"have {version}, need {REQUIRED_DATASETS_VERSION}\n"
            f"Fix: python -m pip install 'datasets=={REQUIRED_DATASETS_VERSION}'"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--tasks", default="mmlu,pubmedqa,mbpp")
    parser.add_argument("--purge", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = sorted(set(tasks) - set(ALL_TASKS))
    if unknown:
        raise SystemExit(f"[hf-cache][error] unknown tasks: {unknown}")

    _assert_datasets_version()
    import datasets

    print(f"[hf-cache] datasets={datasets.__version__} cache_root={cache_root}", flush=True)
    for task in tasks:
        if task != "mmlu":
            path, name = read_lm_eval_dataset_spec(task)
            print(f"[hf-cache] lm_eval {task} -> {path!r} config={name!r}", flush=True)

    if not args.verify_only:
        cache_root.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            if args.purge:
                purge_task_cache(task, cache_root)
            download_task_cache(task, cache_root)

    for task in tasks:
        verify_task_cache_offline(task, cache_root)

    print("[hf-cache][ok] offline cache ready", flush=True)


if __name__ == "__main__":
    main()
