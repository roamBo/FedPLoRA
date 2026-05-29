"""Branch C — FedPLoRA-Sign.

Two complementary local regularizers (run inside `train_client`):

  L_bsign  = ||tanh(γ B_i) - sign(B_anchor)||_1
  L_asparse= ||A_i||_1

The server-side aggregation is reused from v2 (`aggregate_models_fedplora_oneshot`)
unchanged — Branch C is a *local-only* change.

Reads (via args):
  - v4_bsign_lambda     float    weight on B-sign regularizer (default 1e-3)
  - v4_bsign_gamma      float    sharpness of tanh (default 5.0)
  - v4_bsign_anchor_steps int    delay before snapshotting B as anchor (default 1)
  - v4_asparse_lambda   float    L1 weight on A (default 1e-4); shares with `yoco_sparse_lambda`

PEFT initialises lora_B as zero matrices, so `sign(B_0) = 0` everywhere. We work
around this by snapshotting B *after* the first `v4_bsign_anchor_steps` local
batches, where B has accumulated meaningful direction. The anchor is then frozen
for the rest of local training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def maybe_init_bsign_anchor(model, args):
    """Reset the snapshot cache at the start of each client's local training."""
    setattr(args, "_v4_bsign_anchor", None)
    setattr(args, "_v4_bsign_step", 0)


def update_bsign_anchor(model, args):
    """Snapshot current B parameters once we've taken `v4_bsign_anchor_steps` batches.

    Call this *after* each optimizer step. Once snapshot is populated, the
    regularizer becomes active.
    """
    anchor = getattr(args, "_v4_bsign_anchor", None)
    if anchor is not None:
        return  # already snapshotted
    step = int(getattr(args, "_v4_bsign_step", 0)) + 1
    setattr(args, "_v4_bsign_step", step)
    delay = int(getattr(args, "v4_bsign_anchor_steps", 1))
    if step < max(delay, 1):
        return
    snap = {}
    for name, p in model.named_parameters():
        if "lora_B" in name and name.endswith("default.weight"):
            snap[name] = p.detach().cpu().clone().float()
    setattr(args, "_v4_bsign_anchor", snap)


def add_bsign_regularizer(loss, model, args):
    lam = float(getattr(args, "v4_bsign_lambda", 0.0) or 0.0)
    if lam <= 0:
        return loss
    anchor = getattr(args, "_v4_bsign_anchor", None)
    if not anchor:
        return loss
    gamma = float(getattr(args, "v4_bsign_gamma", 5.0))
    terms = []
    for name, p in model.named_parameters():
        if "lora_B" not in name or not name.endswith("default.weight"):
            continue
        if not p.requires_grad:
            continue
        ref = anchor.get(name)
        if ref is None:
            continue
        ref = ref.to(device=p.device, dtype=p.dtype)
        if tuple(ref.shape) != tuple(p.shape):
            continue
        target_sign = torch.sign(ref)
        soft = torch.tanh(gamma * p.float())
        terms.append((soft - target_sign).abs().mean())
    if terms:
        loss = loss + lam * torch.stack(terms).mean()
    return loss


def add_asparse_regularizer(loss, model, args):
    """L1 on A, equivalent to YOCO's sparse-A prior."""
    lam = float(getattr(args, "v4_asparse_lambda", 0.0) or 0.0)
    if lam <= 0:
        return loss
    terms = []
    for name, p in model.named_parameters():
        if "lora_A" not in name or not name.endswith("default.weight"):
            continue
        if not p.requires_grad:
            continue
        terms.append(p.abs().mean())
    if terms:
        loss = loss + lam * torch.stack(terms).mean()
    return loss


def apply_sign_regularizers(loss, model, args):
    """Both regularizers in one entry point, suitable for hooking into train_client."""
    loss = add_bsign_regularizer(loss, model, args)
    loss = add_asparse_regularizer(loss, model, args)
    return loss
