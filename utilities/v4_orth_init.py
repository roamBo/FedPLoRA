"""Branch B: row-orthonormal initialization for LoRA A matrices."""
from __future__ import annotations

import torch


def orthogonalize_lora_A_in_model(model, eps: float = 1e-6) -> int:
    """QR-orthogonalize each LoRA A (r, d). Returns number of matrices updated."""
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_A" not in name or not name.endswith("weight"):
                continue
            if p.ndim != 2:
                continue
            orig_norm = torch.linalg.vector_norm(p.float()).clamp_min(eps)
            pt = p.transpose(0, 1).float()
            try:
                q, _ = torch.linalg.qr(pt, mode="reduced")
            except RuntimeError:
                continue
            q = q.transpose(0, 1).contiguous()
            q_norm = torch.linalg.vector_norm(q).clamp_min(eps)
            p.copy_((q * (orig_norm / q_norm)).to(dtype=p.dtype))
            n += 1
    return n
