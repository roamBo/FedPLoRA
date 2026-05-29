"""Branch F — FedPLoRA-AdaRank (per-domain heterogeneous rank).

STUB. The full implementation requires:

  1. Per-domain LoRA rank assignment at client init (see DOMAIN_RANK below).
  2. PEFT LoRA factory variant that accepts a domain-specific rank.
  3. Server-side padding + masked weighted average across heterogeneous ranks.
  4. Per-client downlink truncation back to that client's domain rank.

Currently this file documents the interface; integration will happen in v4
Stage 4. The compute saved on low-rank domains is ~25%, and the worst-domain
capacity boost is the headline metric.
"""

from __future__ import annotations

DOMAIN_RANK = {
    "general":   4,
    "education": 4,
    "math":      8,
    "code":      8,
    "medical":  16,
    "legal":    16,
    "finance":  16,
}


def get_domain_rank(domain: str, default: int = 8) -> int:
    return int(DOMAIN_RANK.get(str(domain).lower(), default))


def aggregate_models_v4_adarank(global_model, client_uploads, args):
    raise NotImplementedError(
        "Branch F (heterogeneous rank) is a Stage-4 task. It requires changing "
        "the PEFT LoRA factory; see branches/BRANCH_F_AdaRank.md for the plan."
    )
