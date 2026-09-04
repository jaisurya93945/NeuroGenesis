"""Tasks: a total symbolic map ``f`` plus the support of the data distribution.

The separation between ``label_table`` (total, defined on every concept tuple) and
``support`` (where the data actually lives) is load-bearing and is the single most
important definitional choice in this project -- see ``RESEARCH.md`` and the
``closure`` argument in ``oracle.base``.

Rationale: a neural encoder can emit *any* concept tuple, including tuples never
seen in training. So ``f(alpha(c))`` is always well-defined and the question "does
this shortcut encoder still achieve zero loss?" has a definite answer. Restricting
``f`` to the support instead (the ``partial`` reading) is a different, defensible
question, and it changes the counts -- so the oracle makes the caller name which
one they mean rather than silently defaulting.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .concepts import ConceptSpace


@dataclass(frozen=True)
class Task:
    """A neuro-symbolic task: ``y = f(c_1, ..., c_n)`` observed on ``support``.

    Attributes:
        name: human-readable identifier, used in run records.
        space: the concept space.
        label_table: dense array of shape ``(k,) * n_slots`` mapping every concept
            tuple to its label. ``f`` is total by construction.
        support: ``(S, n_slots)`` array of the distinct concept tuples that carry
            probability mass. Rows must be unique.
        support_weights: ``(S,)`` probability mass per support row, summing to 1.
            Uniform by default. Used by the *margin* computation, where "how much
            evidence rules out this shortcut" is a probability, not a count.
        n_labels: size of the label alphabet ``|Y|``.
        meta: generator parameters, analytic RS counts, cost, provenance.
    """

    name: str
    space: ConceptSpace
    label_table: np.ndarray
    support: np.ndarray
    n_labels: int
    support_weights: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        k, n = self.space.k, self.space.n_slots
        if self.label_table.shape != (k,) * n:
            raise ValueError(f"label_table shape {self.label_table.shape} != expected {(k,) * n}")
        if self.support.ndim != 2 or self.support.shape[1] != n:
            raise ValueError(f"support must be (S, {n}), got {self.support.shape}")
        if self.support.shape[0] == 0:
            raise ValueError("support must be non-empty")
        if self.support.min() < 0 or self.support.max() >= k:
            raise ValueError("support entries must lie in range(k)")
        if self.label_table.min() < 0 or self.label_table.max() >= self.n_labels:
            raise ValueError("label_table entries must lie in range(n_labels)")

        # Rows must be unique: duplicated support rows would silently double-count
        # evidence in the margin computation.
        flat = self._flatten(self.support, k)
        if len(np.unique(flat)) != len(flat):
            raise ValueError("support rows must be unique")

        object.__setattr__(self, "label_table", np.ascontiguousarray(self.label_table, np.int16))
        object.__setattr__(self, "support", np.ascontiguousarray(self.support, np.int16))

        if self.support_weights is None:
            w = np.full(len(self.support), 1.0 / len(self.support), dtype=np.float64)
        else:
            w = np.ascontiguousarray(self.support_weights, np.float64)
            if w.shape != (len(self.support),):
                raise ValueError("support_weights must have one entry per support row")
            if w.min() < 0:
                raise ValueError("support_weights must be non-negative")
            total = w.sum()
            if total <= 0:
                raise ValueError("support_weights must have positive total mass")
            w = w / total
        object.__setattr__(self, "support_weights", w)

    # ---- basic accessors -------------------------------------------------

    @staticmethod
    def _flatten(tuples: np.ndarray, k: int) -> np.ndarray:
        """Mixed-radix encode ``(N, n)`` concept tuples into ``(N,)`` integers."""
        out = np.zeros(len(tuples), dtype=np.int64)
        for j in range(tuples.shape[1]):
            out = out * k + tuples[:, j].astype(np.int64)
        return out

    def label_of(self, tuples: np.ndarray) -> np.ndarray:
        """Label each concept tuple. ``tuples`` is ``(N, n_slots)``."""
        tuples = np.asarray(tuples)
        return self.label_table[tuple(tuples[:, j] for j in range(self.space.n_slots))]

    @property
    def support_labels(self) -> np.ndarray:
        """Labels of the support tuples."""
        return self.label_of(self.support)

    def dense_mask(self) -> np.ndarray:
        """``(k,)*n_slots + (n_labels,)`` bool mask: which tuples produce which label.

        This is what the marginalised NeSy loss consumes.
        """
        return self.label_table[..., None] == np.arange(self.n_labels, dtype=np.int16)

    def all_tuples(self) -> np.ndarray:
        """Every concept tuple in ``[k]^n``, row-major, shape ``(k**n, n_slots)``."""
        k, n = self.space.k, self.space.n_slots
        return np.array(list(itertools.product(range(k), repeat=n)), dtype=np.int16)

    @property
    def support_density(self) -> float:
        """Fraction of the concept-tuple grid carrying mass."""
        return len(self.support) / self.space.n_tuples

    def label_entropy(self) -> float:
        """Shannon entropy (nats) of the label marginal under the support weights.

        A key covariate: ``|RS|`` must be shown to predict grounding *beyond* the
        trivial explanation that harder-to-guess labels carry more information.
        """
        y = self.support_labels
        p = np.bincount(y, weights=self.support_weights, minlength=self.n_labels)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def content_hash(self) -> str:
        """Stable hash of the task's mathematical content (not its name or meta).

        Two tasks with the same hash have the same RS set, so this is the oracle
        cache key.
        """
        h = hashlib.sha256()
        h.update(b"neurogenesis-task-v1")
        h.update(np.array([self.space.k, self.space.n_slots, self.n_labels], np.int64).tobytes())
        h.update(self.label_table.tobytes())
        # Canonicalise support row order so permutations hash identically.
        order = np.argsort(self._flatten(self.support, self.space.k))
        h.update(self.support[order].tobytes())
        h.update(np.round(self.support_weights[order], 12).tobytes())
        return h.hexdigest()[:32]

    def __repr__(self) -> str:
        return (
            f"Task({self.name!r}, k={self.space.k}, n={self.space.n_slots}, "
            f"|Y|={self.n_labels}, |supp|={len(self.support)})"
        )


def task_from_fn(
    name: str,
    k: int,
    n_slots: int,
    fn,
    *,
    support: np.ndarray | None = None,
    n_labels: int | None = None,
    support_weights: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
) -> Task:
    """Build a Task from a python function ``fn(c: tuple[int, ...]) -> int``.

    ``fn`` is evaluated on the *whole* grid ``[k]^n`` (``f`` is total); ``support``
    restricts only where data is drawn from, and defaults to the full grid.
    """
    space = ConceptSpace(k=k, n_slots=n_slots)
    grid = np.array(list(itertools.product(range(k), repeat=n_slots)), dtype=np.int16)
    labels = np.array([fn(tuple(int(v) for v in row)) for row in grid], dtype=np.int64)
    if labels.min() < 0:
        raise ValueError("fn must return non-negative labels")
    table = labels.reshape((k,) * n_slots).astype(np.int16)
    return Task(
        name=name,
        space=space,
        label_table=table,
        support=grid if support is None else np.asarray(support, np.int16),
        n_labels=int(labels.max()) + 1 if n_labels is None else n_labels,
        support_weights=support_weights,
        meta=meta or {},
    )
