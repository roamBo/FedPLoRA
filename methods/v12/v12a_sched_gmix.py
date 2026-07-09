"""FedPLoRA-v12a: round-scheduled global-routed B mixing.

This branch tests the 2026-07-08 hypothesis that μ=0.6 recovers macro accuracy
but may collapse B geometry because it is applied from the beginning.  v12a
keeps early rounds near ``--v12_mu_start`` and moves toward ``--v12_mu_end`` by
``--v12_mu_schedule``.
"""

from __future__ import annotations

from methods.v11.v11_common import build_v11_upload_package, norm_agg

from .v12_common import aggregate_models_v12_gmix, scheduled_mu


SUPPORTED_V12A_AGGS = {
    "fedplora_v12a_sched_gmix",
    "fedplora_v12_sched_gmix",
    "v12a_sched_gmix",
    "v12_sched_gmix",
}


def is_fedplora_v12a_agg(agg_type) -> bool:
    return norm_agg(agg_type) in SUPPORTED_V12A_AGGS


def build_fedplora_v12a_upload_package(*args, **kwargs):
    return build_v11_upload_package(*args, **kwargs)


def aggregate_models_fedplora_v12a(global_model, client_uploads, args):
    return aggregate_models_v12_gmix(
        global_model,
        client_uploads,
        args,
        branch_name="round_scheduled_global_B_mixing",
        mu_resolver=scheduled_mu,
    )

