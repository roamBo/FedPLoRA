"""FedPLoRA v5-merge: interference-aware ΔW merging for one-shot federated LoRA.

agg_type family (prefix ``v5m_``):
- ``v5m_mean``        weighted mean of ΔW_i = B_i A_i, rank-r SVD refactor (≈ FlexLoRA; sanity).
- ``v5m_ties``        entry-wise TIES (trim → elect sign → disjoint mean) on ΔW.
- ``v5m_dare_ties``   DARE random-drop-and-rescale before TIES.
- ``v5m_knots_ties``  subspace-aligned TIES: shared right basis from stacked client
                      factors (KnOTS-style), TIES on aligned coefficients, exact
                      factored reconstruction (never materializes Nm×n).

Rank policy (orthogonal to operator): ``--v5m_rank_policy fixed|energy``.
"""

from methods.v5.fedplora_v5_merge import aggregate_models_v5_merge  # noqa: F401
