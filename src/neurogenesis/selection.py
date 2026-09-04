"""Active selection of auxiliary tasks to collapse the reasoning-shortcut space.

This is the design/intervention question the project targets, and it has a clean
formal shape thanks to one fact::

    RS(T_1 and ... and T_m) = RS(T_1) intersect ... intersect RS(T_m)

So each candidate task *eliminates* a set of non-identity relabellings, and
choosing a cheap set of tasks that eliminates all of them is exactly **weighted
set cover**. Greedy therefore inherits the standard ``ln(n) + 1`` approximation
guarantee for unweighted coverage, and is compared here against an exhaustive
optimum on small pools so that *algorithm* quality and *objective* quality can be
told apart -- if greedy matches the optimum but both lose to concept supervision,
the objective is wrong, not the search.

Baselines live here too, deliberately. The one that matters most is
``information_greedy``: if picking tasks by mutual information does as well as
picking them by shortcut coverage, the whole RS apparatus is decoration. That
ablation is meant to be run early and reported whatever it says.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .oracle import enumerate as en
from .oracle.base import RSResult
from .tasks import Task

ORACLE_ARGS = dict(mode="shared", closure="total", allow_noninjective=True)


@dataclass
class SelectionResult:
    """Which tasks were chosen, and what that bought."""

    chosen: list[int]
    chosen_names: list[str]
    rs_counts: list[int]  # |RS| after each addition, starting from the base task
    final_rs_count: int
    identifiable: bool
    total_cost: float
    method: str
    trace: list[dict] = field(default_factory=list)


def _rs_of(tasks: list[Task]) -> RSResult:
    return en.rs_set(tasks, **ORACLE_ARGS)


def _damage(maps: np.ndarray, k: int) -> np.ndarray:
    """Fraction of concepts each relabelling corrupts -- how bad this shortcut is."""
    ident = np.arange(k)
    return np.array([(m != ident).mean() for m in maps], dtype=float)


def greedy_cover(
    base: Task,
    pool: list[Task],
    budget: int,
    *,
    weight_by_damage: bool = True,
    costs: np.ndarray | None = None,
) -> SelectionResult:
    """Greedily add the task eliminating the most shortcut mass per unit cost.

    ``weight_by_damage=True`` values eliminating a shortcut that corrupts every
    concept above one that merely swaps a pair -- the quantity we actually care
    about is concept accuracy, not the raw count of surviving relabellings.
    """
    k = base.space.k
    current = _rs_of([base])
    remaining = current.maps
    chosen: list[int] = []
    counts = [current.count]
    trace: list[dict] = []
    total_cost = 0.0
    costs = np.ones(len(pool)) if costs is None else np.asarray(costs, float)

    for _ in range(budget):
        ident = np.arange(k)
        non_id = np.array([not np.array_equal(m, ident) for m in remaining], dtype=bool)
        if not non_id.any():
            break  # already identifiable
        w = _damage(remaining, k) if weight_by_damage else np.ones(len(remaining))
        w = w * non_id

        best, best_gain, best_keep = None, -1.0, None
        for j, cand in enumerate(pool):
            if j in chosen:
                continue
            survives = _survivors(remaining, base, cand)
            eliminated = w.sum() - w[survives].sum()
            gain = eliminated / max(costs[j], 1e-9)
            if gain > best_gain:
                best, best_gain, best_keep = j, gain, survives

        if best is None or best_gain <= 0:
            break  # no candidate helps
        chosen.append(best)
        remaining = remaining[best_keep]
        total_cost += float(costs[best])
        counts.append(len(remaining))
        trace.append({"added": pool[best].name, "rs_count": len(remaining), "gain": best_gain})

    return SelectionResult(
        chosen=chosen,
        chosen_names=[pool[j].name for j in chosen],
        rs_counts=counts,
        final_rs_count=len(remaining),
        identifiable=len(remaining) == 1,
        total_cost=total_cost,
        method="greedy_cover" + ("_damage" if weight_by_damage else "_count"),
        trace=trace,
    )


def _survivors(maps: np.ndarray, base: Task, cand: Task) -> np.ndarray:
    """Boolean mask: which relabellings still preserve ``cand``'s labels."""
    support = base.support.astype(np.int64)
    true_labels = cand.label_of(support)
    keep = np.zeros(len(maps), dtype=bool)
    for i, alpha in enumerate(maps):
        mapped = np.asarray(alpha, dtype=np.int64)[support]
        keep[i] = bool(np.array_equal(cand.label_of(mapped), true_labels))
    return keep


def random_selection(
    base: Task, pool: list[Task], budget: int, rng: np.random.Generator
) -> SelectionResult:
    """Pick ``budget`` tasks uniformly at random -- guards against 'more tasks is just better'."""
    idx = list(rng.choice(len(pool), size=min(budget, len(pool)), replace=False))
    rs = _rs_of([base] + [pool[j] for j in idx])
    return SelectionResult(
        chosen=[int(j) for j in idx],
        chosen_names=[pool[j].name for j in idx],
        rs_counts=[_rs_of([base]).count, rs.count],
        final_rs_count=rs.count,
        identifiable=rs.is_identifiable,
        total_cost=float(len(idx)),
        method="random",
    )


def information_greedy(base: Task, pool: list[Task], budget: int) -> SelectionResult:
    """Pick tasks by label entropy, ignoring shortcut structure entirely.

    **The ablation that decides whether this project has a contribution.** If
    choosing informative tasks does as well as choosing shortcut-eliminating ones,
    the oracle machinery adds nothing and that is the headline result, reported as
    such rather than buried.
    """
    scores = [t.label_entropy() for t in pool]
    idx = list(np.argsort(scores)[::-1][:budget])
    rs = _rs_of([base] + [pool[j] for j in idx])
    return SelectionResult(
        chosen=[int(j) for j in idx],
        chosen_names=[pool[j].name for j in idx],
        rs_counts=[_rs_of([base]).count, rs.count],
        final_rs_count=rs.count,
        identifiable=rs.is_identifiable,
        total_cost=float(len(idx)),
        method="information_greedy",
    )


def exhaustive_optimal(base: Task, pool: list[Task], budget: int) -> SelectionResult:
    """Smallest achievable ``|RS|`` over all subsets of size <= budget.

    Only for small pools. Its purpose is to separate search quality from objective
    quality when greedy underperforms.
    """
    from itertools import combinations

    best_idx: tuple[int, ...] = ()
    best_count = _rs_of([base]).count
    for size in range(1, budget + 1):
        for combo in combinations(range(len(pool)), size):
            c = _rs_of([base] + [pool[j] for j in combo]).count
            if c < best_count:
                best_count, best_idx = c, combo
            if best_count == 1:
                break
        if best_count == 1:
            break
    return SelectionResult(
        chosen=list(best_idx),
        chosen_names=[pool[j].name for j in best_idx],
        rs_counts=[_rs_of([base]).count, best_count],
        final_rs_count=best_count,
        identifiable=best_count == 1,
        total_cost=float(len(best_idx)),
        method="exhaustive_optimal",
    )
