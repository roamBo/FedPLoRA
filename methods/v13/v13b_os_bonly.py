"""FedPLoRA-v13b one-shot B-only attribution branch.

This branch preserves the v8 routed-B mechanism but gives it a new algorithm
name so the 20260711 low-communication accounting can be tested without
rewriting historical v8 results.
"""

from __future__ import annotations

from .v13_common import (
    SUPPORTED_V13B_BONLY_AGGS,
    aggregate_models_fedplora_v13,
    apply_fedplora_v13_runtime_defaults,
    build_fedplora_v13_upload_package,
    is_fedplora_v13b_os_bonly_agg,
)


def build_fedplora_v13b_os_bonly_upload_package(*args, **kwargs):
    return build_fedplora_v13_upload_package(*args, **kwargs)


def aggregate_models_fedplora_v13b_os_bonly(global_model, client_uploads, args):
    apply_fedplora_v13_runtime_defaults(args)
    return aggregate_models_fedplora_v13(global_model, client_uploads, args)


__all__ = [
    "SUPPORTED_V13B_BONLY_AGGS",
    "aggregate_models_fedplora_v13b_os_bonly",
    "build_fedplora_v13b_os_bonly_upload_package",
    "is_fedplora_v13b_os_bonly_agg",
]

