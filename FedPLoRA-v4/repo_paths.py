"""Import path bootstrap for FedPLoRA-v4 (repo root + v4 root, no utilities shadow)."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap(v4_entry_file: str | Path) -> tuple[Path, Path]:
    """
    Prepare sys.path for v4 training entry.

    - Repo root (parent of FedPLoRA-v4/) must provide utilities/*.py and methods/*.
    - FedPLoRA-v4/ provides methods/fedplora_v4_* only; its ``utilities/`` folder must
      NOT shadow repo ``utilities`` (no package __init__ there).
    """
    v4_root = Path(v4_entry_file).resolve().parents[1]
    repo_root = v4_root.parent
    data_utils = repo_root / "utilities" / "data_utils.py"
    if not data_utils.is_file():
        raise RuntimeError(
            f"[FedPLoRA-v4] Missing {data_utils}. "
            f"Ensure the full FedPLoRA checkout exists at {repo_root} "
            "(FedPLoRA-v4 must live inside that repo, not as a standalone copy)."
        )

    repo_s = str(repo_root)
    v4_s = str(v4_root)
    sys.path[:] = [p for p in sys.path if p not in (repo_s, v4_s)]
    sys.path.insert(0, repo_s)
    sys.path.insert(1, v4_s)

    _purge_shadowed_utilities(repo_root)
    return v4_root, repo_root


def _purge_shadowed_utilities(repo_root: Path) -> None:
    mod = sys.modules.get("utilities")
    if mod is None:
        return
    paths = [str(p) for p in getattr(mod, "__path__", []) or []]
    repo_util = str((repo_root / "utilities").resolve())
    if paths and not any(repo_util in str(p) or str(p) == repo_util for p in paths):
        for key in list(sys.modules):
            if key == "utilities" or key.startswith("utilities."):
                del sys.modules[key]
