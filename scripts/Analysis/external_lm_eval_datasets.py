"""Offline HF cache specs aligned with the installed lm_eval task YAMLs."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

MMLU_PATH = "cais/mmlu"
MMLU_CACHE_SLUG = "cais___mmlu"
MMLU_SKIP_CONFIGS = frozenset({"all", "auxiliary_train"})
MMLU_MIN_SUBJECTS = 57

# lm_eval 0.4.x defaults (used if task yaml cannot be read).
LM_EVAL_TASK_DATASETS = {
    "pubmedqa": ("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source"),
    "mbpp": ("google-research-datasets/mbpp", "full"),
}


def cache_slug(name: str) -> str:
    return name.replace("/", "___")


def read_lm_eval_dataset_spec(task_name: str) -> tuple[str, str | None]:
    """Return (dataset_path, dataset_name) exactly as lm_eval task yaml declares."""
    try:
        import lm_eval
        import yaml
    except ImportError:
        if task_name in LM_EVAL_TASK_DATASETS:
            path, config = LM_EVAL_TASK_DATASETS[task_name]
            return path, config
        raise

    tasks_root = Path(lm_eval.__file__).resolve().parent / "tasks"
    for yaml_path in tasks_root.rglob("*.yaml"):
        try:
            cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(cfg, dict) or cfg.get("task") != task_name:
            continue
        path = cfg.get("dataset_path")
        if not path:
            continue
        return str(path), cfg.get("dataset_name")

    if task_name in LM_EVAL_TASK_DATASETS:
        path, config = LM_EVAL_TASK_DATASETS[task_name]
        return path, config
    raise KeyError(f"unknown lm_eval task dataset spec: {task_name}")


def _mmlu_subject_configs_online() -> list[str]:
    try:
        from datasets import get_dataset_config_names

        configs = get_dataset_config_names(MMLU_PATH, trust_remote_code=False)
    except Exception:
        from datasets import load_dataset_builder

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
            "lm_eval loads cais/mmlu per subject (e.g. abstract_algebra).\n"
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


def dataset_cache_dir(cache_root: Path, dataset_path: str) -> Path:
    return cache_root / "datasets" / cache_slug(dataset_path)


@contextmanager
def temporary_environ(values: dict[str, str]):
    backup = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, old in backup.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _offline_load_dataset(path: str, name: str | None, env: dict[str, str]):
    from datasets import load_dataset

    kwargs = {"trust_remote_code": False}
    if name is None:
        return load_dataset(path, **kwargs)
    return load_dataset(path, name, **kwargs)


def preflight_offline_tasks(task_names: list[str], env: dict[str, str], cache_root: Path) -> None:
    from datasets import load_dataset
    import datasets as datasets_pkg

    print(
        "[external-eval][preflight] datasets="
        f"{datasets_pkg.__version__} cache={env['HF_DATASETS_CACHE']}",
        flush=True,
    )
    for task in sorted(set(task_names)):
        if task == "mmlu":
            configs = assert_mmlu_cache_complete(cache_root)
            print(
                f"[external-eval][preflight] mmlu subjects={len(configs)} "
                "(lm_eval uses per-subject configs)",
                flush=True,
            )
            with temporary_environ(env):
                for config in configs:
                    load_dataset(MMLU_PATH, config, trust_remote_code=False)
            print(f"[external-eval][preflight][ok] mmlu subjects={len(configs)}", flush=True)
            continue

        path, name = read_lm_eval_dataset_spec(task)
        print(
            f"[external-eval][preflight] lm_eval task={task!r} -> "
            f"load_dataset({path!r}, {name!r})",
            flush=True,
        )
        with temporary_environ(env):
            _offline_load_dataset(path, name, env)
        print(f"[external-eval][preflight][ok] {task}", flush=True)


def download_task_cache(task: str, cache_root: Path) -> None:
    from datasets import load_dataset

    datasets_cache = cache_root / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(key, None)

    if task == "mmlu":
        configs = mmlu_subject_configs(online=True)
        print(f"[hf-cache][download] mmlu subjects={len(configs)}", flush=True)
        for config in configs:
            print(f"[hf-cache][download] mmlu config={config!r}", flush=True)
            ds = load_dataset(MMLU_PATH, config, trust_remote_code=False)
            print(f"[hf-cache][download][ok] mmlu {config} splits={list(ds.keys())}", flush=True)
        return

    path, name = read_lm_eval_dataset_spec(task)
    print(f"[hf-cache][download] task={task} dataset={path!r} config={name!r}", flush=True)
    ds = _offline_load_dataset(path, name, {})
    print(f"[hf-cache][download][ok] {task} splits={list(ds.keys())}", flush=True)


def verify_task_cache_offline(task: str, cache_root: Path) -> None:
    env = {
        "HF_HOME": str(cache_root),
        "HF_DATASETS_CACHE": str(cache_root / "datasets"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    preflight_offline_tasks([task], env, cache_root)


def purge_task_cache(task: str, cache_root: Path) -> None:
    import shutil

    if task == "mmlu":
        target = dataset_cache_dir(cache_root, MMLU_PATH)
    else:
        path, _name = read_lm_eval_dataset_spec(task)
        target = dataset_cache_dir(cache_root, path)
    if target.is_dir():
        print(f"[hf-cache][purge] {target}", flush=True)
        shutil.rmtree(target)
