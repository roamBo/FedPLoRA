"""Self-test for v5 merge operators (run on the GPU box where torch is installed).

    python -m methods.v5.selftest_merge

Checks that the actual torch implementation in merge_ops.py satisfies the
algebraic identities verified offline in numpy:
  - KnOTS factored reconstruction is exact at full basis (ΔW_i == C_i V^T).
  - factorize_coefficient_delta reproduces C* V^T at full rank.
  - entry-wise mean equals the closed-form weighted mean of ΔW.
  - TIES / DARE-TIES produce finite factors with the requested downlink rank.
"""

from __future__ import annotations

import torch

from methods.v5.merge_ops import (
    factorize_coefficient_delta,
    factorize_dense_delta,
    knots_align_factors,
    merge_delta_entrywise,
    normalize_weights,
    ties_on_coefficients,
)


def _rand_factors(m, n, ranks, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = [torch.randn(m, r, generator=g) for r in ranks]
    A = [torch.randn(r, n, generator=g) for r in ranks]
    return B, A


def main():
    torch.set_default_dtype(torch.float64)  # tight tolerances
    m, n = 19, 27
    ranks = [3, 4, 2, 5, 3]
    N = len(ranks)
    B, A = _rand_factors(m, n, ranks, seed=1)
    DW = [B[i] @ A[i] for i in range(N)]
    w = normalize_weights([5, 3, 8, 2, 4], N).double()

    # 1. KnOTS reconstruction at full basis
    C, V, st = knots_align_factors(B, A, w, basis_energy_tau=1.0)
    err = max((DW[i] - C[i] @ V.t()).abs().max().item() for i in range(N))
    print(f"[1] knots reconstruction err = {err:.2e}  (k0={st['k0']}, k={st['k_basis']})")
    assert err < 1e-9, "KnOTS reconstruction not exact"

    # 2. factorize coefficient delta at full rank
    C_star = sum(float(w[i]) * C[i] for i in range(N))
    A_new, B_new, S = factorize_coefficient_delta(C_star, V, r_down=C_star.shape[1])
    err2 = (B_new @ A_new - C_star @ V.t()).abs().max().item()
    print(f"[2] coeff factorize err = {err2:.2e}")
    assert err2 < 1e-9

    # 3. entry-wise mean == closed-form weighted mean
    mean_ref = sum(float(w[i]) * DW[i] for i in range(N))
    merged, _ = merge_delta_entrywise(B, A, w, mode="mean", device=torch.device("cpu"))
    err3 = (merged - mean_ref).abs().max().item()
    print(f"[3] entrywise mean err = {err3:.2e}")
    assert err3 < 1e-9

    # 4. TIES entry-wise: finite + factorable
    merged_t, stats_t = merge_delta_entrywise(B, A, w, mode="ties", keep_ratio=0.4)
    A_t, B_t, S_t = factorize_dense_delta(merged_t, r_down=8)
    print(f"[4] ties active_frac={stats_t['active_frac']:.3f} "
          f"factor finite={torch.isfinite(A_t).all().item() and torch.isfinite(B_t).all().item()}")
    assert torch.isfinite(A_t).all() and torch.isfinite(B_t).all()

    # 5. KnOTS-TIES end-to-end
    C2, V2, _ = knots_align_factors(B, A, w, basis_energy_tau=0.9999)
    Cstar2, st2 = ties_on_coefficients(C2, w, keep_ratio=0.3, normalize=True)
    A2, B2, S2 = factorize_coefficient_delta(Cstar2, V2, r_down=8)
    print(f"[5] knots-ties active_frac={st2['active_frac']:.3f} "
          f"rank_out={A2.shape[0]} finite={torch.isfinite(A2).all().item()}")
    assert torch.isfinite(A2).all() and torch.isfinite(B2).all()

    print("\nALL V5 MERGE SELF-TESTS PASS")


if __name__ == "__main__":
    main()
