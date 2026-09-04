"""Exhaustive RS enumeration by depth-first search with support-bucketed pruning.

The search space of shared relabellings is ``k ** k`` -- 10**10 for MNIST digits,
far too large to enumerate naively. The trick is that a relabelling can be
*refuted* long before it is fully specified: once ``alpha(0..d)`` are fixed, every
support tuple whose entries all lie in ``[0, d]`` is already checkable.

So we assign ``alpha(0), alpha(1), ...`` in order and, immediately after assigning
``alpha(d)``, verify every support tuple with ``max(c) == d`` (those are exactly
the tuples that became checkable at this depth). Under full support this refutes
almost every branch within two assignments, and ``k = 10`` runs in milliseconds.

Worst case is genuinely bad: an adversarially sparse support constrains nothing
early and the search degrades toward ``k ** k``. That is not hidden -- it is why
``limit`` and ``RSResult.truncated`` exist, and why the ASP backend is the
fallback for the hard shapes (per-slot maps, relational knowledge, margins).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from ..tasks import Task
from .base import RSClosure, RSMode, RSResult, as_task_list


def _buckets(task: Task) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group support rows by their maximum entry.

    Returns a list indexed by depth ``d``; entry ``d`` is ``(rows, labels)`` for
    the support tuples whose largest concept is exactly ``d`` -- i.e. those that
    become checkable the moment ``alpha(d)`` is assigned.
    """
    k = task.space.k
    supp = task.support.astype(np.int64)
    labels = task.support_labels.astype(np.int64)
    mx = supp.max(axis=1)
    out = []
    for d in range(k):
        sel = mx == d
        out.append((supp[sel], labels[sel]))
    return out


def _support_codes(task: Task) -> set[int]:
    """Mixed-radix codes of support tuples, for the ``partial`` closure check."""
    return set(Task._flatten(task.support, task.space.k).tolist())


def rs_set(
    tasks: Task | Sequence[Task],
    *,
    mode: RSMode,
    closure: RSClosure,
    allow_noninjective: bool,
    limit: int | None = 200_000,
) -> RSResult:
    """Enumerate the reasoning-shortcut set by pruned DFS.

    See ``oracle.base.RSOracle`` for the argument contract. ``mode`` and
    ``closure`` are mandatory on purpose.
    """
    t0 = time.perf_counter()
    task_list = as_task_list(tasks)
    if mode != "shared":
        raise NotImplementedError(
            "the enumeration backend implements shared maps only; "
            "use the ASP backend for per-slot maps"
        )

    k = task_list[0].space.k
    per_task = [(_buckets(t), t.label_table, _support_codes(t), t.space.n_slots) for t in task_list]

    alpha = np.zeros(k, dtype=np.int64)
    used = np.zeros(k, dtype=bool)
    found: list[np.ndarray] = []
    truncated = False

    def consistent_at(depth: int) -> bool:
        """Check every constraint that became decidable after assigning alpha[depth]."""
        for buckets, table, codes, n_slots in per_task:
            rows, labels = buckets[depth]
            if len(rows) == 0:
                continue
            mapped = alpha[rows]  # (m, n_slots)
            got = table[tuple(mapped[:, j] for j in range(n_slots))]
            if not np.array_equal(got, labels):
                return False
            if closure == "partial":
                code = np.zeros(len(mapped), dtype=np.int64)
                for j in range(n_slots):
                    code = code * k + mapped[:, j]
                if any(int(c) not in codes for c in code):
                    return False
        return True

    def dfs(depth: int) -> None:
        nonlocal truncated
        if truncated:
            return
        if depth == k:
            found.append(alpha.copy())
            if limit is not None and len(found) >= limit:
                truncated = True
            return
        for v in range(k):
            if not allow_noninjective and used[v]:
                continue
            alpha[depth] = v
            if not allow_noninjective:
                used[v] = True
            if consistent_at(depth):
                dfs(depth + 1)
            if not allow_noninjective:
                used[v] = False
            if truncated:
                return

    dfs(0)

    maps = np.array(found, dtype=np.int8) if found else np.zeros((0, k), dtype=np.int8)
    return RSResult(
        maps=maps,
        count=len(found),
        truncated=truncated,
        backend="enumerate",
        elapsed_s=time.perf_counter() - t0,
        mode=mode,
        closure=closure,
    )
