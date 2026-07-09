"""FedPLoRA-v12b: NMI-guarded global-routed B mixing.

This is a deliberately small adaptive branch: use a higher μ only while the
pre-mix B-subspace clustering still agrees with domain geometry.  If NMI drops
below ``--v12_nmi_guard_threshold``, the next aggregation backs off to the low
μ endpoint.  It is meant as an attribution test, not an oracle-heavy final
method.
"""

from __future__ import annotations

from methods.v11.v11_common import build_v11_upload_package, norm_agg

from .v12_common import aggregate_models_v12_gmix, nmi_guard_mu


SUPPORTED_V12B_AGGS = {
    "fedplora_v12b_nmi_guard_gmix",
    "fedplora_v12_adaptive_gmix",
    "v12b_nmi_guard_gmix",
    "v12_adaptive_gmix",
}


def is_fedplora_v12b_agg(agg_type) -> bool:
    return norm_agg(agg_type) in SUPPORTED_V12B_AGGS


def build_fedplora_v12b_upload_package(*args, **kwargs):
    return build_v11_upload_package(*args, **kwargs)


def aggregate_models_fedplora_v12b(global_model, client_uploads, args):
    return aggregate_models_v12_gmix(
        global_model,
        client_uploads,
        args,
        branch_name="nmi_guarded_global_B_mixing",
        mu_resolver=nmi_guard_mu,
    )

