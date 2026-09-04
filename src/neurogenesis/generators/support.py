"""Support-restriction generators: vary identifiability while controlling data volume.

Thinning the support is the most direct way to move a task along the
identifiability spectrum. The trap is that it also removes *data*, so any
correlation between ``|RS|`` and grounding could just be "fewer distinct
situations seen". These generators therefore always report ``support_size``
separately, and the experiment resamples to equal example counts, so constraint
strength and data quantity are separable covariates rather than one confounded one.
"""

from __future__ import annotations

import numpy as np

from ..tasks import Task


def thin_support(
    task: Task, density: float, rng: np.random.Generator, name: str | None = None
) -> Task:
    """Keep a random ``density`` fraction of the support."""
    grid = task.support
    n_keep = max(1, int(round(density * len(grid))))
    keep = grid[rng.choice(len(grid), size=n_keep, replace=False)]
    return Task(
        name=name or f"{task.name}_d{density:.2f}",
        space=task.space,
        label_table=task.label_table,
        support=keep,
        n_labels=task.n_labels,
        meta={**task.meta, "support_density": density, "generator": "support"},
    )


def density_sweep(task: Task, densities: list[float], seed: int = 0) -> list[Task]:
    """One thinned task per requested density, from a single seed."""
    rng = np.random.default_rng(seed)
    return [thin_support(task, d, rng) for d in densities]


def greedy_minimal_identifying_support(
    task: Task, rng: np.random.Generator | None = None, max_rounds: int = 4
) -> Task:
    """Greedily drop support tuples while identifiability is preserved.

    This is the ``threadbare`` construction: the result is *provably* identifiable
    but hangs on as few support tuples as possible, so its **margin is tiny**.
    These are precisely the tasks where the binary property should stop predicting
    grounding while the margin should keep predicting it -- the decisive cell for H2.

    Greedy, hence not guaranteed minimum; several random orders are tried and the
    smallest result kept. Exactness is not needed -- what matters is landing in the
    low-margin regime, which is verified by measuring the margin afterwards.
    """
    from ..oracle import enumerate as en

    args = dict(mode="shared", closure="total", allow_noninjective=True)
    rng = rng or np.random.default_rng(0)
    if not en.rs_set(task, **args).is_identifiable:
        raise ValueError(f"{task.name} is not identifiable to begin with")

    best = task.support
    for _ in range(max_rounds):
        support = task.support.copy()
        order = rng.permutation(len(support))
        for idx in order:
            if len(support) <= 1:
                break
            row = task.support[idx]
            keep = np.array([not np.array_equal(r, row) for r in support], dtype=bool)
            if keep.sum() == len(support):
                continue  # already dropped
            candidate = Task(
                name="probe",
                space=task.space,
                label_table=task.label_table,
                support=support[keep],
                n_labels=task.n_labels,
            )
            if en.rs_set(candidate, **args).is_identifiable:
                support = support[keep]
        if len(support) < len(best):
            best = support

    return Task(
        name=f"{task.name}_threadbare",
        space=task.space,
        label_table=task.label_table,
        support=best,
        n_labels=task.n_labels,
        meta={
            **task.meta,
            "generator": "threadbare",
            "original_support_size": int(len(task.support)),
        },
    )


def refuting_tuples(task: Task, alpha: np.ndarray) -> np.ndarray:
    """Indices of support tuples whose label ``alpha`` fails to preserve.

    These are exactly the evidence that rules ``alpha`` out. Their total
    probability mass is ``alpha``'s violation mass, and the margin is the minimum
    of that over all ``alpha != id``.
    """
    alpha = np.asarray(alpha, dtype=np.int64)
    mapped = alpha[task.support.astype(np.int64)]
    return np.flatnonzero(task.label_of(mapped) != task.support_labels)


def swap_map(k: int, a: int, b: int) -> np.ndarray:
    """The transposition exchanging concepts ``a`` and ``b``."""
    alpha = np.arange(k)
    alpha[a], alpha[b] = b, a
    return alpha


def rarefy_against(
    task: Task,
    alpha: np.ndarray,
    rarity: float = 1e-3,
    name: str | None = None,
) -> Task:
    """Keep the full support but starve the evidence that refutes ``alpha``.

    This is the sharpest form of the ``threadbare`` idea, and the experiment's
    decisive cell for H2.

    Thinning the support cannot produce a very small margin: with ``|S|`` tuples of
    equal mass the margin floors at ``1/|S|``, so a *small* support is not a *weak*
    one. Instead we keep every support tuple -- support size, data volume and label
    distribution are untouched -- and move probability mass **away** from the tuples
    that refute one chosen relabelling.

    The result is provably identifiable (those tuples still carry non-zero mass, so
    ``alpha`` is still excluded) while a model trained on finite data may never see
    the evidence that makes it so. If binary identifiability still predicted
    grounding here, H2 would be wrong. If the margin predicts it and the binary
    property does not, that is H2's central claim.
    """
    idx = refuting_tuples(task, alpha)
    if len(idx) == 0:
        raise ValueError("alpha is already a shortcut of this task; nothing to rarefy")
    w = np.ones(len(task.support), dtype=float)
    w[idx] = rarity
    return Task(
        name=name or f"{task.name}_rare{rarity:g}",
        space=task.space,
        label_table=task.label_table,
        support=task.support,
        n_labels=task.n_labels,
        support_weights=w / w.sum(),
        meta={
            **task.meta,
            "generator": "rarefied",
            "n_refuting": int(len(idx)),
            "rarity": rarity,
        },
    )
