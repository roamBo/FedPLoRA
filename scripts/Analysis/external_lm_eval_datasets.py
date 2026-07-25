"""Dataset specs used by external lm-eval (offline cache + preflight)."""

from __future__ import annotations

from pathlib import Path

MMLU_PATH = "cais/mmlu"
MMLU_CACHE_SLUG = "cais___mmlu"
MMLU_SKIP_CONFIGS = frozenset({"all", "auxiliary_train"})
MMLU_MIN_SUBJECTS = 57
TASK_DATASETS = {
    "pubmedqa": ("pubmed_qa", "pqa_labeled"),
    "mbpp": ("google-research-datasets/mbpp", "full"),
}


def cache_slug(name: str) -> str:
    return name.replace("/", "___")


def _mmlu_subject_configs_online() -> list[str]:
    try:
        from datasets import get_dataset_config_names

        configs = get_dataset_config_names(MMLU_PATH, trust_remote_code=False)
    except Exception:
        from datasets import load_dataset_builder

        # datasets 2.20.0 requires an explicit config when probing cais/mmlu.
        builder = load_dataset_builder(MMLU_PATH, "abstract_algebra", trust_remote_code=False)
        configs = list(builder.builder_configs.keys())
    return sorted(name for name in configs if name not in MMLU_SKIP_CONFIGS)


def mmlu_subject_configs(*, online: bool = False, cache_root: Path | None = None) -> list[str]:
    if online:
        return _mmlu_subject_configs_online()
    if cache_root is None:
        raise ValueError("cache_root is required when online=False")
    root = cache_root / "datasets" / MMLU_CACHE_SLUG
    if not root.is_dir():
        return []
    return sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and path.name not in MMLU_SKIP_CONFIGS
    )


def assert_mmlu_cache_complete(cache_root: Path) -> list[str]:
    configs = mmlu_subject_configs(online=False, cache_root=cache_root)
    if "abstract_algebra" not in configs:
        raise SystemExit(
            "[external-eval][error] MMLU offline cache incomplete.\n"
            "lm_eval loads cais/mmlu per subject (e.g. abstract_algebra), not config='all'.\n"
            f"Cached subjects: {configs or '(none)'}\n"
            "Fix: python scripts/Analysis/prepare_external_lm_eval_hf_cache.py --tasks mmlu --purge"
        )
    if len(configs) < MMLU_MIN_SUBJECTS:
        raise SystemExit(
            "[external-eval][error] MMLU offline cache incomplete: "
            f"have {len(configs)} subjects, need >= {MMLU_MIN_SUBJECTS}.\n"
            "Fix: python scripts/Analysis/prepare_external_lm_eval_hf_cache.py --tasks mmlu --purge"
        )
    return configs
