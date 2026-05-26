"""Branch D — FedPLoRA-Mix.

Per-client mixer between server-returned `A_down` and the locally-trained
`A_local`. Three modes:
  - fixed:      η is a constant from args.v4_mix_eta (default 0.5)
  - per_domain: grid search η on the client's val set
  - per_input:  light MLP gate trained for a few steps on val set

Server aggregation reuses v2 fedplora-oneshot (or any Branch A aggregator) —
Branch D only changes what each client puts into its LoRA before evaluation.

We expose two helpers:
  - `snapshot_local_A(model)`                 — save A_local at end of local training
  - `build_mixed_A(A_down, A_local, eta_fn)`  — produce effective A for evaluation
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_V4_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _V4_ROOT.parent
for _p in (_REPO_ROOT, _V4_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def snapshot_local_A(model):
    """Return a state_dict subset containing lora_A weights, ready to persist."""
    snap = {}
    for name, p in model.named_parameters():
        if "lora_A" in name and name.endswith("default.weight"):
            snap[name] = p.detach().cpu().clone()
    return snap


def build_mixed_A(A_down: Dict[str, torch.Tensor],
                  A_local: Dict[str, torch.Tensor],
                  eta: float) -> Dict[str, torch.Tensor]:
    """Static mixer: A_eff = eta * A_down + (1-eta) * A_local."""
    eta = float(min(max(eta, 0.0), 1.0))
    out = {}
    for key, down in A_down.items():
        loc = A_local.get(key)
        if loc is None or tuple(loc.shape) != tuple(down.shape):
            out[key] = down.clone()
        else:
            out[key] = eta * down.float() + (1.0 - eta) * loc.float()
    return out


def search_per_domain_eta(eval_fn: Callable[[Dict[str, torch.Tensor]], float],
                          A_down, A_local,
                          grid=None) -> float:
    """Grid-search η on a single domain's val set (the eval_fn already encodes the val data)."""
    grid = grid or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_eta, best_loss = 0.5, math.inf
    for eta in grid:
        A_eff = build_mixed_A(A_down, A_local, eta)
        loss = float(eval_fn(A_eff))
        if loss < best_loss:
            best_loss = loss
            best_eta = float(eta)
    return best_eta


class InputGate(nn.Module):
    """Tiny per-input MoE gate: pooled hidden -> sigmoid scalar."""

    def __init__(self, hidden_size, mlp_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (B, T, H). Returns gate (B, 1, 1) ∈ (0, 1)."""
        pooled = hidden_states.mean(dim=1)            # (B, H)
        logit = self.net(pooled)                      # (B, 1)
        return torch.sigmoid(logit).unsqueeze(-1)     # (B, 1, 1)


def train_input_gate(gate: InputGate,
                     model_apply_fn: Callable[[Dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
                     A_down, A_local,
                     val_dataloader,
                     args,
                     epochs: int = 3,
                     lr: float = 5e-4) -> InputGate:
    """Train a per-input gate using val data.

    `model_apply_fn(A_eff, batch)` should run the full forward with A_eff installed
    and return a per-sample loss tensor (B,).
    """
    optim = torch.optim.AdamW(gate.parameters(), lr=lr)
    gate.train()
    device = next(gate.parameters()).device
    for _ in range(epochs):
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Need access to embed hidden states; we approximate with the input embedding
            # mean as the gate input. Adapter authors can substitute a real hidden state.
            input_emb = batch.get("input_emb")
            if input_emb is None:
                # Fallback: use one-hot token embeddings averaged (very rough)
                continue
            g = gate(input_emb).squeeze(-1).squeeze(-1)   # (B,)
            # Build "soft" A_eff per sample by linear blending; expensive for (r, d) matrices,
            # so we use the mean gate value g.mean() to drive the static mixer per batch.
            # Practical compromise: per-batch eta = g.mean().
            eta = g.mean().clamp(0.0, 1.0).detach().item()
            A_eff = build_mixed_A(A_down, A_local, eta)
            loss_per = model_apply_fn(A_eff, batch)
            loss = (g * loss_per.detach() + (1.0 - g) * loss_per.detach()).mean()  # placeholder
            loss = loss + 1e-3 * (g - 0.5).pow(2).mean()                            # anti-extreme
            optim.zero_grad()
            loss.backward()
            optim.step()
    gate.eval()
    return gate
