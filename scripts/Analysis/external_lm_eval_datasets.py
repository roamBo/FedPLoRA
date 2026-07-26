"""Offline HF cache specs aligned with the installed lm_eval task YAMLs."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path

MMLU_PATH = "cais/mmlu"
MMLU_CACHE_SLUG = "cais___mmlu"
MMLU_SKIP_CONFIGS = frozenset({"all", "auxiliary_train"})
MMLU_MIN_SUBJECTS = 57
PUBMEDQA_HUB_PATH = "bigbio/pubmed_qa"
PUBMEDQA_HUB_CONFIG = "pubmed_qa_labeled_fold0_source"

LM_EVAL_TASK_DATASETS = {
    "pubmedqa": (PUBMEDQA_HUB_PATH, PUBMEDQA_HUB_CONFIG),
    "mbpp": ("google-research-datasets/mbpp", "full"),
}


def cache_slug(name: str) -> str:
    return name.replace("/", "___")


def pubmedqa_export_dir(cache_root: Path) -> Path:
    return cache_root / "export" / "pubmedqa_bigbio"


def pubmedqa_lm_eval_override_dir(cache_root: Path) -> Path:
    return cache_root / "lm_eval_task_overrides"


def bundled_preprocess_pubmedqa_path() -> Path:
    """Repo-local helper imported by lm_eval pubmedqa.yaml (!function doc_to_text)."""
    return Path(__file__).resolve().parent / "lm_eval_task_overrides" / "pubmedqa" / "preprocess_pubmedqa.py"


def read_lm_eval_dataset_spec(task_name: str) -> tuple[str, str | None]:
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


def assert_pubmedqa_export_exists(cache_root: Path) -> Path:
    export_dir = pubmedqa_export_dir(cache_root)
    marker = export_dir / "dataset_dict.json"
    if not marker.is_file():
        raise SystemExit(
            "[external-eval][error] PubMedQA offline export missing.\n"
            f"Expected: {marker}\n"
            "bigbio/pubmed_qa cannot load offline from Hub scripts alone.\n"
            "Fix: python scripts/Analysis/prepare_external_lm_eval_hf_cache.py --tasks pubmedqa --purge"
        )
    return export_dir


def assert_pubmedqa_export_schema(export_dir: Path) -> None:
    from datasets import load_from_disk

    ds = load_from_disk(str(export_dir))
    split_name = "test" if "test" in ds else "train" if "train" in ds else next(iter(ds.keys()))
    doc = ds[split_name][0]
    keys = set(doc.keys())
    has_context = "CONTEXTS" in keys or "context" in keys
    has_question = "QUESTION" in keys or "question" in keys
    if not has_context or not has_question:
        raise SystemExit(
            "[external-eval][error] PubMedQA export schema unexpected.\n"
            f"split={split_name} keys={sorted(keys)}\n"
            "Fix: python scripts/Analysis/prepare_external_lm_eval_hf_cache.py --tasks pubmedqa --purge"
        )


def _sync_pubmedqa_preprocess_helper(out_dir: Path) -> Path:
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    helper_src = _resolve_preprocess_pubmedqa_source()
    helper_dst = out_dir / "preprocess_pubmedqa.py"
    shutil.copy2(helper_src, helper_dst)
    return helper_dst


def ensure_pubmedqa_lm_eval_override(cache_root: Path) -> Path:
    out_dir = pubmedqa_lm_eval_override_dir(cache_root)
    override = out_dir / "pubmedqa.yaml"
    if not override.is_file():
        materialize_pubmedqa_lm_eval_yaml(cache_root)
    else:
        _sync_pubmedqa_preprocess_helper(out_dir)
    return override


def assert_pubmedqa_export_complete(cache_root: Path) -> Path:
    export_dir = assert_pubmedqa_export_exists(cache_root)
    ensure_pubmedqa_lm_eval_override(cache_root)
    assert_pubmedqa_export_schema(export_dir)
    return export_dir


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


def _load_dataset(path: str, name: str | None, *, trust_remote_code: bool = False):
    from datasets import load_dataset

    kwargs = {"trust_remote_code": trust_remote_code}
    if name is None:
        return load_dataset(path, **kwargs)
    return load_dataset(path, name, **kwargs)


def _resolve_preprocess_pubmedqa_source() -> Path:
    bundled = bundled_preprocess_pubmedqa_path()
    if bundled.is_file():
        return bundled
    import lm_eval

    installed = Path(lm_eval.__file__).resolve().parent / "tasks" / "pubmedqa" / "preprocess_pubmedqa.py"
    if installed.is_file():
        return installed
    raise SystemExit(
        "[external-eval][error] preprocess_pubmedqa.py missing.\n"
        f"Expected bundled helper at: {bundled}\n"
        f"Or installed lm_eval helper at: {installed}"
    )


def materialize_pubmedqa_lm_eval_yaml(cache_root: Path) -> Path:
    """Patch installed pubmedqa.yaml to load from local save_to_disk export."""
    import lm_eval

    export_dir = assert_pubmedqa_export_exists(cache_root)
    src = Path(lm_eval.__file__).resolve().parent / "tasks" / "pubmedqa" / "pubmedqa.yaml"
    text = src.read_text(encoding="utf-8")
    text = re.sub(
        r"^dataset_path:.*$",
        f"dataset_path: {export_dir.resolve().as_posix()}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^dataset_name:.*\n", "", text, count=1, flags=re.MULTILINE)
    out_dir = pubmedqa_lm_eval_override_dir(cache_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / "pubmedqa.yaml"
    out_yaml.write_text(text, encoding="utf-8")
    helper_dst = _sync_pubmedqa_preprocess_helper(out_dir)
    print(f"[hf-cache][pubmedqa] lm_eval override -> {out_yaml}", flush=True)
    print(f"[hf-cache][pubmedqa] helper module -> {helper_dst}", flush=True)
    return out_dir


def export_pubmedqa_offline(cache_root: Path) -> Path:
    import shutil
    from datasets import load_from_disk

    export_dir = pubmedqa_export_dir(cache_root)
    marker = export_dir / "dataset_dict.json"
    if marker.is_file():
        print(f"[hf-cache][download][skip] pubmedqa export exists: {export_dir}", flush=True)
        ds = load_from_disk(str(export_dir))
        print(f"[hf-cache][download][ok] pubmedqa splits={list(ds.keys())}", flush=True)
        ensure_pubmedqa_lm_eval_override(cache_root)
        return export_dir

    path, name = read_lm_eval_dataset_spec("pubmedqa")
    print(
        f"[hf-cache][download] pubmedqa online fetch {path!r} config={name!r} "
        "(trust_remote_code=True)",
        flush=True,
    )
    ds = _load_dataset(path, name, trust_remote_code=True)
    if export_dir.is_dir():
        shutil.rmtree(export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(export_dir))
    print(f"[hf-cache][download][ok] pubmedqa exported -> {export_dir}", flush=True)
    ensure_pubmedqa_lm_eval_override(cache_root)
    return export_dir


def preflight_offline_tasks(task_names: list[str], env: dict[str, str], cache_root: Path) -> None:
    from datasets import load_dataset, load_from_disk
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

        if task == "pubmedqa":
            export_dir = assert_pubmedqa_export_complete(cache_root)
            print(
                f"[external-eval][preflight] pubmedqa local export -> {export_dir}",
                flush=True,
            )
            with temporary_environ(env):
                ds = load_from_disk(str(export_dir))
            print(f"[external-eval][preflight][ok] pubmedqa splits={list(ds.keys())}", flush=True)
            continue

        path, name = read_lm_eval_dataset_spec(task)
        print(
            f"[external-eval][preflight] lm_eval task={task!r} -> "
            f"load_dataset({path!r}, {name!r})",
            flush=True,
        )
        with temporary_environ(env):
            _load_dataset(path, name, trust_remote_code=False)
        print(f"[external-eval][preflight][ok] {task}", flush=True)


def download_task_cache(task: str, cache_root: Path) -> None:
    from datasets import load_dataset

    datasets_cache = cache_root / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
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

    if task == "pubmedqa":
        export_pubmedqa_offline(cache_root)
        return

    path, name = read_lm_eval_dataset_spec(task)
    print(f"[hf-cache][download] task={task} dataset={path!r} config={name!r}", flush=True)
    ds = _load_dataset(path, name, trust_remote_code=False)
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

    targets = []
    if task == "mmlu":
        targets.append(dataset_cache_dir(cache_root, MMLU_PATH))
    elif task == "pubmedqa":
        targets.append(dataset_cache_dir(cache_root, PUBMEDQA_HUB_PATH))
        targets.append(cache_root / "hub" / "datasets--bigbio--pubmed_qa")
        targets.append(pubmedqa_export_dir(cache_root))
        targets.append(pubmedqa_lm_eval_override_dir(cache_root))
    else:
        path, _name = read_lm_eval_dataset_spec(task)
        targets.append(dataset_cache_dir(cache_root, path))
        hub_slug = path.replace("/", "--")
        targets.append(cache_root / "hub" / f"datasets--{hub_slug}")
    for target in targets:
        if target.is_dir():
            print(f"[hf-cache][purge] {target}", flush=True)
            shutil.rmtree(target)


def lm_eval_include_path_for_tasks(task_names: list[str], cache_root: Path) -> str | None:
    if "pubmedqa" not in task_names:
        return None
    assert_pubmedqa_export_complete(cache_root)
    return str(pubmedqa_lm_eval_override_dir(cache_root).resolve())
