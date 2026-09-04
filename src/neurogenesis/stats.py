"""Estimation-first statistics.

The house style here is deliberate: report effect sizes and bootstrap intervals,
show every per-seed point, and use a significance test only where it answers a
question that was asked in advance. With four conditions and ten seeds, ``p <
0.001`` is trivially purchasable by adding seeds, so a small p-value would be
evidence of nothing except that we ran the experiment. Hence: no stars anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Estimate:
    """A point estimate with a bootstrap interval."""

    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}] (n={self.n})"


def bootstrap_mean(
    values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Estimate:
    """Percentile bootstrap CI for the mean."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(v.mean()), float(lo), float(hi), len(v))


def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Estimate:
    """Percentile bootstrap CI for ``mean(a) - mean(b)``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    da = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    d = da - db
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(a.mean() - b.mean()), float(lo), float(hi), min(len(a), len(b)))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b). Non-parametric, in [-1, 1].

    Chosen over Cohen's d because the outcome here is bimodal by construction
    (a shortcut scores exactly 0), which makes a standard-deviation-based effect
    size close to meaningless.
    """
    a = np.asarray(a, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[None, :]
    return float((a > b).mean() - (a < b).mean())


def permutation_trend_test(groups: list[np.ndarray], n_perm: int = 100_000, seed: int = 0) -> float:
    """Permutation p-value for a monotone decreasing trend across ordered groups.

    Statistic: Spearman correlation between group index and value, pooled over all
    observations. The null shuffles group membership. Reported once, for the single
    ordering question asked in the preregistration.
    """
    values = np.concatenate(groups)
    labels = np.concatenate([np.full(len(g), i, dtype=float) for i, g in enumerate(groups)])
    if len(np.unique(values)) == 1:
        return 1.0

    def stat(vals: np.ndarray) -> float:
        rv = _rankdata(vals)
        rl = _rankdata(labels)
        rv = rv - rv.mean()
        rl = rl - rl.mean()
        denom = np.sqrt((rv**2).sum() * (rl**2).sum())
        return float((rv * rl).sum() / denom) if denom > 0 else 0.0

    observed = stat(values)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        if stat(rng.permutation(values)) <= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0
