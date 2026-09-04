"""Build ``(images, concepts, label)`` datasets from a Task.

Concept tuples are drawn from the task's support (with its weights), then each
slot is rendered by sampling an image of that concept from the requested MNIST
split. Because images are sampled *within* a split and splits are index-disjoint,
no image can appear in two splits -- asserted in ``tests/test_data_splits.py``.

The ``data_seed`` controlling tuple sampling is deliberately separate from the
``init_seed`` controlling weight initialisation, so run-to-run variance can be
decomposed into "which data" versus "which initialisation".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tasks import Task
from . import mnist


@dataclass
class TupleDataset:
    """A rendered dataset. ``x`` is ``(N, n_slots, 28, 28)`` or ``(N, n_slots, d)``."""

    x: np.ndarray
    concepts: np.ndarray  # (N, n_slots) ground truth -- for evaluation only
    labels: np.ndarray  # (N,)
    task_name: str
    split: str

    def __len__(self) -> int:
        return len(self.labels)


def sample_concept_tuples(task: Task, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` concept tuples from the support, according to its weights."""
    idx = rng.choice(len(task.support), size=n, p=task.support_weights)
    return task.support[idx].astype(np.int64)


def render_mnist(
    task: Task,
    n: int,
    split: str,
    data: mnist.MNIST,
    rng: np.random.Generator,
) -> TupleDataset:
    """Render concept tuples as MNIST digit images (Tier M)."""
    if task.space.k > 10:
        raise ValueError("MNIST rendering supports at most 10 concepts")
    images, _ = data.split(split)
    by_digit = data.indices_by_digit(split, k=task.space.k)
    for d, idxs in enumerate(by_digit):
        if len(idxs) == 0:
            raise ValueError(f"digit {d} absent from split {split!r}")

    concepts = sample_concept_tuples(task, n, rng)
    n_slots = task.space.n_slots
    picked = np.empty((n, n_slots), dtype=np.int64)
    for j in range(n_slots):
        for d in range(task.space.k):
            sel = concepts[:, j] == d
            if sel.any():
                picked[sel, j] = rng.choice(by_digit[d], size=int(sel.sum()))
    x = images[picked]  # (n, n_slots, 28, 28)
    return TupleDataset(
        x=x.astype(np.float32),
        concepts=concepts,
        labels=task.label_of(concepts).astype(np.int64),
        task_name=task.name,
        split=split,
    )


def make_synthetic_codebook(k: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """A random near-orthonormal codebook: one vector per concept (Tier S)."""
    a = rng.standard_normal((dim, k))
    q, _ = np.linalg.qr(a)
    return q.T[:k]  # (k, dim)


def render_synthetic(
    task: Task,
    n: int,
    codebook: np.ndarray,
    noise: float,
    rng: np.random.Generator,
    split: str = "train",
) -> TupleDataset:
    """Render tuples as ``codebook[c] + noise * eps`` (Tier S).

    Perceptual difficulty becomes an explicit, tunable axis. That matters because
    the leading rival explanation for any ``|RS|`` effect is that inductive bias --
    a function of encoder and data geometry -- is doing the work instead.
    """
    concepts = sample_concept_tuples(task, n, rng)
    vecs = codebook[concepts]  # (n, n_slots, dim)
    x = vecs + noise * rng.standard_normal(vecs.shape)
    return TupleDataset(
        x=x.astype(np.float32),
        concepts=concepts,
        labels=task.label_of(concepts).astype(np.int64),
        task_name=task.name,
        split=split,
    )
