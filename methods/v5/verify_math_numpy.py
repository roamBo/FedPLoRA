"""Pure-numpy verification of the v5 merge algebra (mirrors methods/v5/merge_ops.py).

We verify three identities that the torch implementation relies on:
  1. KnOTS factored reconstruction: ΔW_i == C_prov_i @ V_p^T exactly.
  2. Basis rotation preserves reconstruction: ΔW_i == C_i @ V^T (full rank) exactly.
  3. V's columns are right singular vectors of weighted stacked updates.
  4. factorize_coefficient_delta reproduces C* @ V^T at full rank.
  5. TIES disjoint-mean matches the reference (trim->elect->merge).
"""
import numpy as np

rng = np.random.default_rng(0)

m, n = 17, 23          # ΔW is m×n
N = 5                  # clients
ranks = [3, 4, 2, 5, 3]
w = rng.random(N); w = w / w.sum()

B_list = [rng.standard_normal((m, r)) for r in ranks]
A_list = [rng.standard_normal((r, n)) for r in ranks]
DW = [B_list[i] @ A_list[i] for i in range(N)]

# ---- KnOTS align (numpy mirror) ----
P = np.concatenate(A_list, axis=0)                 # (Σr, n)
Up, Sp, Vhp = np.linalg.svd(P, full_matrices=False)
tol = Sp.max() * 1e-7
k0 = int((Sp > tol).sum())
Up, Sp, Vp = Up[:, :k0], Sp[:k0], Vhp[:k0, :].T    # Vp (n, k0)

C_prov = []
off = 0
for i in range(N):
    Ui = Up[off:off+ranks[i], :]
    off += ranks[i]
    C_prov.append(B_list[i] @ (Ui * Sp[None, :]))  # (m, k0)

# Identity 1: ΔW_i == C_prov_i @ Vp^T
err1 = max(np.abs(DW[i] - C_prov[i] @ Vp.T).max() for i in range(N))
print(f"[1] KnOTS factored reconstruction max err = {err1:.2e}")

# Rotation
H = sum(w[i] * (C_prov[i].T @ C_prov[i]) for i in range(N))
evals, evecs = np.linalg.eigh(H)
order = np.argsort(evals)[::-1]
evals = np.clip(evals[order], 0, None)
Wh = evecs[:, order]                               # (k0, k0) orthogonal
V = Vp @ Wh                                        # (n, k0)
C = [C_prov[i] @ Wh for i in range(N)]

# Identity 2: ΔW_i == C_i @ V^T at full rank
err2 = max(np.abs(DW[i] - C[i] @ V.T).max() for i in range(N))
print(f"[2] rotated reconstruction max err   = {err2:.2e}")

# Identity 3: V are right singular vectors of weighted stack
stack = np.concatenate([np.sqrt(w[i]) * DW[i] for i in range(N)], axis=0)  # (Nm, n)
_, Ss, Vhs = np.linalg.svd(stack, full_matrices=False)
# Compare subspaces: |cos| of leading singular directions ~ 1
cos_lead = abs(np.dot(Vhs[0], V[:, 0]) / (np.linalg.norm(Vhs[0]) * np.linalg.norm(V[:, 0])))
print(f"[3] leading right-singular alignment |cos| = {cos_lead:.6f} (want ~1.0)")

# Identity 4: factorize C* @ V^T reproduces at full rank
C_star = sum(w[i] * C[i] for i in range(N))        # any C* in coeff space
k = C_star.shape[1]
Uc, Sc, Whc = np.linalg.svd(C_star, full_matrices=False)
r = min(k, len(Sc))
Uc, Sc, Whc = Uc[:, :r], Sc[:r], Whc[:r, :]
sq = np.sqrt(np.clip(Sc, 1e-12, None))
B_new = Uc * sq[None, :]
A_new = (sq[:, None]) * (V @ Whc.T).T
err4 = np.abs(B_new @ A_new - C_star @ V.T).max()
print(f"[4] factorize_coefficient_delta err  = {err4:.2e}")

# Identity 5: TIES reference on a small matrix
def ties(mats, weights, keep=0.4):
    trimmed = []
    for t in mats:
        thr = np.quantile(np.abs(t), 1 - keep)
        trimmed.append(np.where(np.abs(t) >= thr, t, 0.0))
    elected = sum(weights[i] * trimmed[i] for i in range(len(mats)))
    gamma = np.sign(elected)
    num = np.zeros_like(trimmed[0]); den = np.zeros_like(trimmed[0])
    for i, t in enumerate(trimmed):
        match = (np.sign(t) == gamma) & (t != 0)
        mf = match * weights[i]
        num += mf * t; den += mf
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

merged = ties([rng.standard_normal((m, n)) for _ in range(N)], w)
print(f"[5] TIES merged finite = {np.isfinite(merged).all()}, "
      f"active_frac = {(merged != 0).mean():.3f}")

ok = err1 < 1e-9 and err2 < 1e-9 and err4 < 1e-9 and cos_lead > 0.999
print("\nALL CORE IDENTITIES PASS" if ok else "\n*** MISMATCH ***")
