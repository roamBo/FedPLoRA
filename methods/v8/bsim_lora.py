"""FedPLoRA-v8 B-SimLoRA aggregation facade.

v8 makes the FedPLoRA paper-facing protocol explicit:
  - shared LoRA A as the coordinate system;
  - B-only client uploads for the main method;
  - LoRA-B principal-angle clustering without domain labels;
  - one routed B pool per client for personalization/cold-start evaluation.

The heavy lifting is centralized in ``methods.lora_expert_baselines`` so v8
and the expert baselines share the same distance, clustering, and routing code.
"""

from __future__ import annotations

from methods.lora_expert_baselines import (
    aggregate_models_lora_expert_baseline,
    build_lora_expert_upload_package,
)
from utilities.utils import (
    FEDPLORA_V8_B_ONLY_AGGS,
    FEDPLORA_V8_FAMILY_AGGS,
)


SUPPORTED_V8_AGGS = set(FEDPLORA_V8_FAMILY_AGGS)


def _norm_agg(agg_type) -> str:
    return (agg_type or "").strip().lower().replace("-", "_")


def is_fedplora_v8_agg(agg_type) -> bool:
    return _norm_agg(agg_type) in SUPPORTED_V8_AGGS


def is_fedplora_v8_b_only_agg(agg_type) -> bool:
    return _norm_agg(agg_type) in FEDPLORA_V8_B_ONLY_AGGS


def build_fedplora_v8_upload_package(*args, **kwargs):
    """Build the v8 client payload.

    ``fedplora_v8`` / ``v8_bsim`` are static B-only. Scheduled variants set a
    transient ``args._v8_current_b_only_upload`` flag each round.
    """

    return build_lora_expert_upload_package(*args, **kwargs)


def aggregate_models_fedplora_v8(global_model, client_uploads, args):
    """Aggregate one FedPLoRA-v8 round and materialize routed client B states."""

    return aggregate_models_lora_expert_baseline(global_model, client_uploads, args)
