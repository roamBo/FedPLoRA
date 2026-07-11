"""FedPLoRA-v13a one-shot main branch.

Named wrapper for the 20260711 main algorithm:
true low-rank A-delta sketch + routed B, alpha=1.0, no μ, no local A/B
regularizers.
"""

from __future__ import annotations

from .v13_common import (
    SUPPORTED_V13A_OS_AGGS,
    aggregate_models_fedplora_v13,
    apply_fedplora_v13_runtime_defaults,
    build_fedplora_v13_upload_package,
    is_fedplora_v13a_os_agg,
)


def build_fedplora_v13a_os_upload_package(*args, **kwargs):
    return build_fedplora_v13_upload_package(*args, **kwargs)


def aggregate_models_fedplora_v13a_os(global_model, client_uploads, args):
    apply_fedplora_v13_runtime_defaults(args)
    return aggregate_models_fedplora_v13(global_model, client_uploads, args)


__all__ = [
    "SUPPORTED_V13A_OS_AGGS",
    "aggregate_models_fedplora_v13a_os",
    "build_fedplora_v13a_os_upload_package",
    "is_fedplora_v13a_os_agg",
]

