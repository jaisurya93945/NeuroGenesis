"""Uniformly random tasks: a null family with no designed structure.

Every other generator plants something -- an algebraic law, a chosen symmetry, a
thinned support. This one plants nothing: a random total label table over a random
support. Two uses:

1. **Differential testing.** Structured tasks share the symmetries the oracle
   backends might both get wrong in the same way; random tasks do not, which is
   what makes agreement between independent implementations meaningful evidence.
2. **A null arm for the correlational study (E2).** If ``|RS|`` predicts grounding
   only on hand-designed families, that is a fact about the families rather than
   about identifiability.

This lives in the package rather than in ``tests/`` deliberately: a cross-test
import (``from tests.x import y``) works under ``python -m pytest``, which puts the
working directory on ``sys.path``, and fails under a bare ``pytest``, which does
not. That difference broke CI while every local run passed.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..concepts import ConceptSpace
from ..tasks import Task


def random_task(
    rng: np.random.Generator,
    k: int,
    n: int,
    density: float,
    *,
    n_labels: int | None = None,
    name: str | None = None,
) -> Task:
    """A task with a random total label table and a random support.

    Args:
        rng: source of randomness; pass a seeded generator for reproducibility.
        k: concept vocabulary size.
        n: number of slots.
        density: fraction of the concept grid carrying probability mass.
        n_labels: label alphabet size; drawn from ``[2, max(3, k)]`` when omitted.
        name: optional override for the task name.
    """
    if not 0 < density <= 1:
        raise ValueError(f"density must be in (0, 1], got {density}")
    if n_labels is None:
        n_labels = int(rng.integers(2, max(3, k) + 1))

    table = rng.integers(0, n_labels, size=(k,) * n).astype(np.int16)
    grid = np.array(list(itertools.product(range(k), repeat=n)), dtype=np.int16)
    n_keep = max(1, int(round(density * len(grid))))
    support = grid[rng.choice(len(grid), size=n_keep, replace=False)]

    return Task(
        name=name or f"rand_k{k}_n{n}_d{density:g}",
        space=ConceptSpace(k=k, n_slots=n),
        label_table=table,
        support=support,
        n_labels=n_labels,
        meta={"generator": "random", "density": density},
    )


def random_task_batch(
    rng: np.random.Generator,
    count: int,
    *,
    k_range: tuple[int, int] = (2, 7),
    n_choices: tuple[int, ...] = (2, 3),
    densities: tuple[float, ...] = (0.2, 0.5, 1.0),
) -> list[Task]:
    """A batch of random tasks spanning the given shape ranges."""
    out = []
    for _ in range(count):
        k = int(rng.integers(*k_range))
        n = int(rng.choice(n_choices))
        density = float(rng.choice(densities))
        out.append(random_task(rng, k, n, density))
    return out
