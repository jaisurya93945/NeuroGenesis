"""Graded measures of shortcut vulnerability.

Binary identifiability is coarse. A task can be identifiable only because of a
handful of very low-probability support tuples -- formally shortcut-free, but with
almost no evidence actually ruling the shortcuts out. Finite data and finite
training see the *amount* of evidence, not the logical fact.

The **margin** is that amount::

    margin(T) = min over alpha != id of  Pr_{c ~ D} [ f(alpha(c)) != f(c) ]

It is the probability mass of the evidence that refutes the *cheapest-to-satisfy*
wrong relabelling. Two properties make it the natural graded refinement:

- ``margin(T) > 0`` **iff** ``T`` is identifiable (a non-identity shortcut is
  exactly an ``alpha != id`` with zero violation mass), so the margin strictly
  subsumes the binary property;
- it generalises to *approximate* shortcuts, ``RS_eps(T) = {alpha : violation <= eps}``,
  which is what a model trained on finite data can actually distinguish.

Hypothesis H2 is that the margin predicts empirical grounding better than the
binary property does. Computing it is a constrained minimisation, not an
enumeration, which is why clingo's ``#minimize`` earns its place here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import clingo
import numpy as np

from ..tasks import Task
from .asp import _facts

#: Support masses are rationals; clingo optimises over integers. Scaling by 1e6
#: and rounding gives ~6 decimal places of resolution, far finer than any
#: difference we could resolve empirically with tens of runs.
MASS_SCALE = 1_000_000


@dataclass
class MarginResult:
    """The margin of a task, and the relabelling that achieves it."""

    margin: float
    argmin_alpha: np.ndarray | None
    is_identifiable: bool
    optimal: bool  # whether clingo proved optimality
    elapsed_s: float

    def __repr__(self) -> str:
        return (
            f"MarginResult(margin={self.margin:.6f}, identifiable={self.is_identifiable}, "
            f"optimal={self.optimal})"
        )


def margin(task: Task, *, allow_noninjective: bool = True, timeout_s: float = 60.0) -> MarginResult:
    """Compute ``margin(T)`` exactly by weighted ASP optimisation."""
    t0 = time.perf_counter()
    k = task.space.k
    n = task.space.n_slots

    weights = np.maximum(1, np.round(task.support_weights * MASS_SCALE).astype(np.int64))
    mass_facts = "\n".join(
        f"mass({','.join(str(int(v)) for v in row)},{int(w)})."
        for row, w in zip(task.support, weights, strict=True)
    )

    cs = ",".join(f"C{i}" for i in range(n))
    ds = ",".join(f"D{i}" for i in range(n))
    alpha_lits = ",".join(f"alpha(C{i},D{i})" for i in range(n))

    parts = [
        f"concept(0..{k - 1}).",
        _facts([task]),
        mass_facts,
        "1 { alpha(D,E) : concept(E) } 1 :- concept(D).",
        # A shortcut candidate must differ from the identity somewhere.
        ":- alpha(D,D) : concept(D).",
        f"violated({cs}) :- sup(0,{cs}), flab(0,{cs},Y), {alpha_lits}, flab(0,{ds},Y2), Y != Y2.",
        f"#minimize {{ W,{cs} : violated({cs}), mass({cs},W) }}.",
        "#show alpha/2.",
    ]
    if not allow_noninjective:
        parts.append(":- alpha(D1,E), alpha(D2,E), D1 != D2.")

    ctl = clingo.Control(["--models=0", "--warn=none", "--opt-mode=optN"])
    ctl.add("base", [], "\n".join(parts))
    ctl.ground([("base", [])])

    best_cost: int | None = None
    best_alpha: np.ndarray | None = None

    def on_model(m: clingo.Model) -> None:
        nonlocal best_cost, best_alpha
        cost = m.cost[0] if m.cost else 0
        if best_cost is None or cost < best_cost:
            best_cost = cost
            arr = np.zeros(k, dtype=np.int8)
            for sym in m.symbols(shown=True):
                d, e = sym.arguments
                arr[d.number] = e.number
            best_alpha = arr

    with ctl.solve(on_model=on_model, async_=True) as handle:
        handle.wait(timeout_s)
        handle.cancel()
        optimal = handle.get().exhausted

    if best_cost is None:
        # No alpha != id exists at all -- only possible when k == 1.
        return MarginResult(float("inf"), None, True, True, time.perf_counter() - t0)

    total = float(weights.sum())
    value = best_cost / total
    return MarginResult(
        margin=value,
        argmin_alpha=best_alpha,
        is_identifiable=value > 0,
        optimal=optimal,
        elapsed_s=time.perf_counter() - t0,
    )


def margin_bruteforce(task: Task) -> float:
    """Reference implementation: enumerate every ``alpha != id`` and take the minimum.

    Only tractable for ``k <= 6``; exists purely to validate the ASP version.
    """
    import itertools

    k = task.space.k
    support = task.support.astype(np.int64)
    weights = task.support_weights
    true_labels = task.support_labels
    ident = tuple(range(k))
    best = float("inf")
    for alpha_t in itertools.product(range(k), repeat=k):
        if alpha_t == ident:
            continue
        alpha = np.array(alpha_t, dtype=np.int64)
        mapped = alpha[support]
        got = task.label_of(mapped)
        viol = float(weights[got != true_labels].sum())
        best = min(best, viol)
    return best


def damage_profile(task: Task, rs_maps: np.ndarray) -> np.ndarray:
    """Per-shortcut concept damage: the fraction of concepts each shortcut mislabels.

    Used to weight the greedy set-cover objective in ``selection``: eliminating a
    shortcut that corrupts every concept is worth more than eliminating one that
    swaps a single pair.
    """
    k = task.space.k
    ident = np.arange(k)
    if len(rs_maps) == 0:
        return np.zeros(0)
    return np.array([(m != ident).mean() for m in rs_maps], dtype=float)
