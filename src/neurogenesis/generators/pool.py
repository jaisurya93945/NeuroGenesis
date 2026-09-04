"""Candidate auxiliary-task pools, and what they cost.

The design question this project targets -- "which tasks should you acquire to
collapse the shortcut space?" -- needs a pool to choose from and a cost model to
choose against. Both are stated explicitly here because both are contestable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tasks import Task, task_from_fn


@dataclass(frozen=True)
class CostModel:
    """What each kind of supervision costs.

    ``tau`` is the fixed charge for *authoring a rule* -- writing down an auxiliary
    task's symbolic knowledge -- as a multiple of the cost of one concept label.
    Its true value depends entirely on who is doing the annotating, so no single
    value is defensible. Results are therefore always reported as a curve over a
    **band** of ``tau``, never at one point; a headline number at a chosen ``tau``
    would be the easiest possible way to manufacture a favourable comparison.
    """

    tau: float = 100.0
    per_example: float = 0.0
    concept_label: float = 1.0

    def task_cost(self, n_examples: int = 0) -> float:
        return self.tau + self.per_example * n_examples

    def supervision_cost(self, n_labels: int) -> float:
        return self.concept_label * n_labels


def candidate_predicates(k: int, n_slots: int = 2) -> list[tuple[str, callable]]:
    """A varied pool of auxiliary predicates over the same concept space.

    Deliberately heterogeneous in how much they constrain: some (modular sums)
    pin concepts tightly, others (comparisons, membership) barely constrain at all.
    A selection method that cannot tell them apart has nothing to offer.
    """
    out: list[tuple[str, callable]] = []
    for q in (2, 3, 4, 5):
        out.append((f"mod{q}_c0", lambda c, q=q: c[0] % q))
        out.append((f"mod{q}_sum", lambda c, q=q: sum(c) % q))
    for w in range(1, min(k, 10)):
        out.append((f"lin_w{w}", lambda c, w=w: (c[0] + w * c[1]) % k))
    out.append(("greater", lambda c: int(c[0] > c[1])))
    out.append(("equal", lambda c: int(c[0] == c[1])))
    out.append(("maxc", lambda c: max(c)))
    out.append(("minc", lambda c: min(c)))
    out.append(("absdiff", lambda c: abs(c[0] - c[1])))
    out.append(("sum_int", lambda c: sum(c)))
    for t in (k // 3, k // 2, 2 * k // 3):
        if 0 < t < k:
            out.append((f"thresh{t}", lambda c, t=t: int(c[0] >= t)))
    return out


def build_pool(k: int, n_slots: int = 2, seed: int = 0) -> list[Task]:
    """Materialise the candidate pool as Tasks over the full concept grid."""
    rng = np.random.default_rng(seed)
    tasks = []
    for name, fn in candidate_predicates(k, n_slots):
        t = task_from_fn(f"aux_{name}", k=k, n_slots=n_slots, fn=fn, meta={"pool": True})
        tasks.append(t)
    # A few random partitions, so the pool is not entirely hand-designed.
    for i in range(4):
        n_lab = int(rng.integers(2, max(3, k)))
        table = rng.integers(0, n_lab, size=(k,) * n_slots)
        tasks.append(
            task_from_fn(
                f"aux_rand{i}",
                k=k,
                n_slots=n_slots,
                fn=lambda c, tb=table: int(tb[c]),
                meta={"pool": True, "random": True},
            )
        )
    return tasks


def divisor_modular_pool(k: int, n_slots: int = 2) -> list[Task]:
    """A pool where no single task suffices, so selection is a real problem.

    Discovered empirically (see DECISIONS.md D7): with a generic pool of
    predicates, *every* selection method -- including uniform random -- makes
    ``y = (c1 + 9 c2) mod 10`` identifiable with a single auxiliary task. A
    predicate like ``c0 mod 3`` is not invariant under any cyclic shift of a
    10-element space, so it annihilates all nine non-identity shortcuts at once and
    every method looks equally good. An experiment on such a pool measures nothing.

    A predicate ``c0 mod q`` is invariant under the shift ``u`` exactly when
    ``q | u``. Restricting the pool to moduli that **divide** ``k`` therefore gives
    every candidate a non-trivial, partial invariance: each kills some shifts and
    spares others, and only a combination reaches identifiability. That is a
    genuine weighted set-cover instance, which is what E3 needs in order to
    distinguish selection strategies at all.
    """
    divisors = [q for q in range(2, k) if k % q == 0]
    tasks = []
    for q in divisors:
        tasks.append(
            task_from_fn(
                f"aux_div_mod{q}_c0",
                k=k,
                n_slots=n_slots,
                fn=lambda c, q=q: c[0] % q,
                meta={"pool": True, "modulus": q, "kills_shifts_not_divisible_by": q},
            )
        )
        tasks.append(
            task_from_fn(
                f"aux_div_mod{q}_sum",
                k=k,
                n_slots=n_slots,
                fn=lambda c, q=q: sum(c) % q,
                meta={"pool": True, "modulus": q},
            )
        )
    # Shift-invariant distractors: cost budget, eliminate nothing. A method that
    # cannot tell these from useful tasks will waste its entire budget.
    for name, fn in (
        ("diff", lambda c: (c[0] - c[1]) % k),
        ("diff2", lambda c: (2 * (c[0] - c[1])) % k),
    ):
        tasks.append(
            task_from_fn(
                f"aux_inv_{name}",
                k=k,
                n_slots=n_slots,
                fn=fn,
                meta={"pool": True, "shift_invariant": True},
            )
        )
    return tasks


def individually_insufficient(base: Task, pool: list[Task]) -> list[Task]:
    """Keep only candidates that do *not* make the base task identifiable alone.

    A generic filter for turning any pool into one where selection matters.
    """
    from ..oracle import enumerate as en

    args = dict(mode="shared", closure="total", allow_noninjective=True)
    return [t for t in pool if not en.rs_set([base, t], **args).is_identifiable]
