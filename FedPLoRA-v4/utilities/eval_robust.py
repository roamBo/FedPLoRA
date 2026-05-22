"""Robust evaluation utilities for v4.

Bootstrap-based mean ± 95% CI on per-domain per-client batches.
"""

from __future__ import annotations

import math
import numpy as np

_DEFAULT_BOOTSTRAP = 1000


def bootstrap_mean_ci(values, n_boot: int = _DEFAULT_BOOTSTRAP, alpha: float = 0.05, seed: int = 0):
    """Return (mean, lo, hi) with `100(1-alpha)%` BCa-style percentile CI."""
    arr = np.asarray([float(v) for v in values if v is not None and not math.isnan(float(v))])
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=arr.size, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def summarize_per_domain_per_client(per_client_per_domain):
    """Given a dict[client_id][domain] -> metric (e.g. token_accuracy), compute:
      - per-domain mean ± 95% CI across clients
      - macro = mean over domains
      - worst = min over domains
    """
    domains = sorted({d for v in per_client_per_domain.values() for d in v.keys()})
    per_domain_summary = {}
    for d in domains:
        vals = [v[d] for v in per_client_per_domain.values() if d in v]
        mean, lo, hi = bootstrap_mean_ci(vals)
        per_domain_summary[d] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}
    macro = float(np.mean([per_domain_summary[d]["mean"] for d in domains]))
    worst = float(min(per_domain_summary[d]["mean"] for d in domains))
    return {
        "macro": macro,
        "worst": worst,
        "per_domain": per_domain_summary,
        "domains": domains,
    }
