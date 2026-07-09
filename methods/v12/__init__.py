"""FedPLoRA-v12 2026-07-08 branch entry points."""

from .v12a_sched_gmix import (
    SUPPORTED_V12A_AGGS,
    aggregate_models_fedplora_v12a,
    build_fedplora_v12a_upload_package,
    is_fedplora_v12a_agg,
)
from .v12b_nmi_guard_gmix import (
    SUPPORTED_V12B_AGGS,
    aggregate_models_fedplora_v12b,
    build_fedplora_v12b_upload_package,
    is_fedplora_v12b_agg,
)

SUPPORTED_V12_AGGS = set(SUPPORTED_V12A_AGGS) | set(SUPPORTED_V12B_AGGS)


def is_fedplora_v12_agg(agg_type) -> bool:
    return is_fedplora_v12a_agg(agg_type) or is_fedplora_v12b_agg(agg_type)


def build_fedplora_v12_upload_package(*args, **kwargs):
    agg_type = getattr(kwargs.get("args", None), "agg_type", "")
    if is_fedplora_v12b_agg(agg_type):
        return build_fedplora_v12b_upload_package(*args, **kwargs)
    return build_fedplora_v12a_upload_package(*args, **kwargs)


def aggregate_models_fedplora_v12(global_model, client_uploads, args):
    if is_fedplora_v12b_agg(getattr(args, "agg_type", "")):
        return aggregate_models_fedplora_v12b(global_model, client_uploads, args)
    return aggregate_models_fedplora_v12a(global_model, client_uploads, args)


__all__ = [
    "SUPPORTED_V12_AGGS",
    "SUPPORTED_V12A_AGGS",
    "SUPPORTED_V12B_AGGS",
    "aggregate_models_fedplora_v12",
    "aggregate_models_fedplora_v12a",
    "aggregate_models_fedplora_v12b",
    "build_fedplora_v12_upload_package",
    "build_fedplora_v12a_upload_package",
    "build_fedplora_v12b_upload_package",
    "is_fedplora_v12_agg",
    "is_fedplora_v12a_agg",
    "is_fedplora_v12b_agg",
]

