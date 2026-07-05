"""FedPLoRA-v11 2026-07-05 branch entry points."""

from .v11a_relaxed_a import (
    SUPPORTED_V11A_AGGS,
    aggregate_models_fedplora_v11a,
    build_fedplora_v11a_upload_package,
    is_fedplora_v11a_agg,
)
from .v11c_gmix import (
    SUPPORTED_V11C_AGGS,
    aggregate_models_fedplora_v11c,
    build_fedplora_v11c_upload_package,
    is_fedplora_v11c_agg,
)

SUPPORTED_V11_AGGS = set(SUPPORTED_V11A_AGGS) | set(SUPPORTED_V11C_AGGS)


def is_fedplora_v11_agg(agg_type) -> bool:
    return is_fedplora_v11a_agg(agg_type) or is_fedplora_v11c_agg(agg_type)


def build_fedplora_v11_upload_package(*args, **kwargs):
    agg_type = getattr(kwargs.get("args", None), "agg_type", "")
    if is_fedplora_v11c_agg(agg_type):
        return build_fedplora_v11c_upload_package(*args, **kwargs)
    return build_fedplora_v11a_upload_package(*args, **kwargs)


def aggregate_models_fedplora_v11(global_model, client_uploads, args):
    if is_fedplora_v11c_agg(getattr(args, "agg_type", "")):
        return aggregate_models_fedplora_v11c(global_model, client_uploads, args)
    return aggregate_models_fedplora_v11a(global_model, client_uploads, args)


__all__ = [
    "SUPPORTED_V11_AGGS",
    "SUPPORTED_V11A_AGGS",
    "SUPPORTED_V11C_AGGS",
    "aggregate_models_fedplora_v11",
    "aggregate_models_fedplora_v11a",
    "aggregate_models_fedplora_v11c",
    "build_fedplora_v11_upload_package",
    "build_fedplora_v11a_upload_package",
    "build_fedplora_v11c_upload_package",
    "is_fedplora_v11_agg",
    "is_fedplora_v11a_agg",
    "is_fedplora_v11c_agg",
]

