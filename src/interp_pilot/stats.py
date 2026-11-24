"""Paired statistics across seeds (pure numpy/scipy).

For one arm, one method and one metric, d_s = metric(method, seed s) - metric(unedited model, seed s).
Reported: n, mean and median of d, a percentile bootstrap 95% CI of the mean (resampling seeds, B = 10,000,
fixed seed), the exact two-sided paired sign-flip permutation p-value of the mean (all 2^n sign patterns
for n <= 20, enumerated in blocks; Monte Carlo only above 20 seeds, which the frozen design does not reach), and the Wilcoxon signed-rank p-value as a secondary test.
Holm's step-down correction is applied over the family of primary contrasts. No seed-count rules are used.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_ci(d: Sequence[float], n_boot: int = 10000, seed: int = 12345, level: float = 0.95, stat=np.mean):
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = stat(d[idx], axis=1)
    a = (1 - level) / 2
    return (float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a)))


EXACT_MAX_N = 20
_BLOCK = 1 << 16


def sign_flip_p(d: Sequence[float], n_mc: int = 100000, seed: int = 12345) -> float | None:
    """Two-sided p-value of H0: d is symmetric about 0, statistic |mean(d)|. Exact for n <= EXACT_MAX_N (20):
    all 2^n sign patterns, enumerated in blocks of 2^16. The observed pattern is included, so p >= 2^-n."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return None
    obs = abs(d.mean())
    if n <= EXACT_MAX_N:
        total, hits, bits = 1 << n, 0, np.arange(n, dtype=np.int64)
        for start in range(0, total, _BLOCK):
            k = np.arange(start, min(start + _BLOCK, total), dtype=np.int64)
            signs = 1.0 - 2.0 * ((k[:, None] >> bits) & 1)
            hits += int(np.count_nonzero(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))
        return hits / total
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_mc, n))
    null = np.abs((signs * d).mean(axis=1))
    return float(np.mean(null >= obs - 1e-12))


def wilcoxon_p(d: Sequence[float]) -> float | None:
    from scipy.stats import wilcoxon
    d = np.asarray(d, dtype=float)
    if len(d) == 0 or np.all(d == 0):
        return None
    try:
        return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return None


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for {name: p}. None values are passed through."""
    items = sorted(((k, p) for k, p in pvals.items() if p is not None), key=lambda t: t[1])
    m = len(items)
    out, running = {k: None for k in pvals}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def paired_summary(method_vals: Sequence[float], base_vals: Sequence[float], n_boot: int = 10000, seed: int = 12345) -> dict:
    a, b = np.asarray(method_vals, dtype=float), np.asarray(base_vals, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return {"n": int(len(d)), "mean_method": float(a[ok].mean()) if ok.any() else None,
            "mean_base": float(b[ok].mean()) if ok.any() else None,
            "mean_delta": float(d.mean()) if len(d) else None, "median_delta": float(np.median(d)) if len(d) else None,
            "ci95_low": lo, "ci95_high": hi, "p_signflip": sign_flip_p(d), "p_wilcoxon": wilcoxon_p(d),
            "deltas": d.tolist()}
