"""Shared types for reasoning-shortcut oracles.

A *reasoning shortcut* (RS) for a task ``T = (f, supp)`` is a relabelling of the
concept vocabulary that the data cannot distinguish from the truth::

    RS(T) = { alpha : [k] -> [k]  |  for all c in supp,  f(alpha(c)) = f(c) }

Two properties make this object worth computing:

1. ``RS`` is a **monoid**: it contains the identity and is closed under
   composition.
2. ``RS(T1 and T2) = RS(T1) intersect RS(T2)``.

(2) is why multitask constraint *selection* is a set-cover problem, and it is the
formal basis of ``neurogenesis.selection``.

Note that ``alpha`` need not be a permutation. Non-injective ``alpha`` are concept
*collapses*, which is a real and separately-interesting failure mode, so
``allow_noninjective`` is a first-class switch rather than an assumption.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from ..tasks import Task

#: Whether one shared relabelling is applied to every slot, or one per slot.
RSMode = Literal["shared", "per_slot"]

#: How ``f`` is read off the support.
#:
#: ``"total"``  -- ``f`` is defined on all of ``[k]^n``; ``f(alpha(c))`` always
#:                 evaluates. This matches what a neural encoder can actually emit
#:                 and is the project default.
#: ``"partial"`` -- ``alpha`` is disqualified if it maps a support tuple outside
#:                 the support. A different, defensible question; it yields
#:                 different counts, which is exactly why it must be named.
RSClosure = Literal["total", "partial"]


@dataclass
class RSResult:
    """The reasoning-shortcut set of a task (or of a conjunction of tasks)."""

    maps: np.ndarray  # (R, k) int8 -- each row is one alpha
    count: int
    truncated: bool
    backend: str
    elapsed_s: float
    mode: RSMode = "shared"
    closure: RSClosure = "total"

    @property
    def is_identifiable(self) -> bool:
        """True iff the only shortcut is the identity, i.e. the task pins ``f``."""
        return self.count == 1 and not self.truncated

    @property
    def n_permutations(self) -> int:
        """How many shortcuts are bijections (pure relabellings, no collapse)."""
        if self.count == 0:
            return 0
        k = self.maps.shape[1]
        return int(sum(len(np.unique(m)) == k for m in self.maps))

    @property
    def n_collapsing(self) -> int:
        """How many shortcuts merge two or more concepts."""
        return self.count - self.n_permutations

    def contains(self, alpha: np.ndarray) -> bool:
        """Whether a given relabelling is in this set."""
        alpha = np.asarray(alpha, dtype=np.int8)
        return bool((self.maps == alpha).all(axis=1).any())

    def sorted_maps(self) -> np.ndarray:
        """Maps in canonical lexicographic order, for differential testing."""
        if self.count == 0:
            return self.maps
        order = np.lexsort(self.maps.T[::-1])
        return self.maps[order]

    def __and__(self, other: RSResult) -> RSResult:
        """Intersect two RS sets -- the fast path used by the selection loop."""
        if self.truncated or other.truncated:
            raise ValueError("cannot intersect truncated RS sets without loss of soundness")
        a = {m.tobytes() for m in self.maps}
        keep = np.array([m.tobytes() in a for m in other.maps], dtype=bool)
        maps = other.maps[keep] if len(other.maps) else other.maps
        return RSResult(
            maps=maps,
            count=int(len(maps)),
            truncated=False,
            backend=f"({self.backend}&{other.backend})",
            elapsed_s=self.elapsed_s + other.elapsed_s,
            mode=self.mode,
            closure=self.closure,
        )

    def __repr__(self) -> str:
        trunc = ", TRUNCATED" if self.truncated else ""
        return (
            f"RSResult(count={self.count}, perms={self.n_permutations}, "
            f"collapsing={self.n_collapsing}, backend={self.backend!r}{trunc})"
        )


class RSOracle(Protocol):
    """Interface every backend implements.

    ``mode`` and ``closure`` are deliberately **mandatory keyword arguments** with
    no defaults. A silent default here would be the single most dangerous bug in
    the project: every downstream number inherits the choice, and the failure is
    invisible (you get plausible counts that answer a different question).
    """

    def rs_set(
        self,
        tasks: Task | Sequence[Task],
        *,
        mode: RSMode,
        closure: RSClosure,
        allow_noninjective: bool,
        limit: int | None = 200_000,
    ) -> RSResult: ...


def as_task_list(tasks: Task | Sequence[Task]) -> list[Task]:
    """Normalise the ``Task | Sequence[Task]`` argument, checking compatibility."""
    lst = [tasks] if isinstance(tasks, Task) else list(tasks)
    if not lst:
        raise ValueError("need at least one task")
    k, n = lst[0].space.k, lst[0].space.n_slots
    for t in lst[1:]:
        if t.space.k != k or t.space.n_slots != n:
            raise ValueError("all tasks must share the same concept space")
    return lst
