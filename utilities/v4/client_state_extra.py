"""Branch D — persistent per-client A_local storage on disk.

Used by `fed_train_sft_v4.py` when `agg_type` starts with `v4_mix_`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch


def save_client_a_local(state: Dict[str, torch.Tensor], save_dir: str | Path, client_idx: int):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_dir / f"client_{client_idx:03d}_A_local.pt")


def load_client_a_local(save_dir: str | Path, client_idx: int):
    path = Path(save_dir) / f"client_{client_idx:03d}_A_local.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")
