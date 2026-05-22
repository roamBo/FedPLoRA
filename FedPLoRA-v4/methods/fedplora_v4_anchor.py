"""Branch E — FedPLoRA-Anchor (server-side anchor calibration).

STUB. The full implementation requires:

  1. `utilities/anchor_data.py` to load 50/domain public anchor samples.
  2. A candidate config grid (lam_min, lam_max, kappa, cluster_k).
  3. A server-side `eval_on_anchor(model, A_down, anchor_loaders)` that loads
     each candidate's A_down into the model and computes mean loss across the
     7 × 50 anchor samples.

Currently this file documents the interface; integration with the main loop
will happen in v4 Stage 5 (see README §6).
"""

from __future__ import annotations

from methods.common_v4 import M  # noqa: F401  (re-export to keep symmetry)


def aggregate_models_v4_anchor(global_model, client_uploads, args):
    raise NotImplementedError(
        "Branch E (anchor calibration) is a Stage-5 task. Implement after "
        "Branches A/C/D have been validated."
    )
