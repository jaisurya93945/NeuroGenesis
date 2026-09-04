"""Answer-set-programming backend for reasoning-shortcut enumeration.

This is the *general* oracle. The pruned-DFS backend in ``enumerate.py`` is faster
for shared maps over a functional ``f``, but it cannot express:

- **per-slot** relabellings (search space ``(k**k) ** n_slots``),
- **margin optimisation** -- "which non-identity shortcut is cheapest to satisfy?",
  which is a weighted `#minimize` and has no natural DFS formulation,
- relational (non-functional) knowledge, planned.

Beyond capability, the two backends exist so they can be **cross-checked**. They
share no code and no algorithm, so agreement between them on thousands of random
tasks is real evidence that the definition in ``base.py`` is implemented faithfully.
That differential test is the project's main defence against its #1 risk.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import clingo
import numpy as np

from ..tasks import Task
from .base import RSClosure, RSMode, RSResult, as_task_list


def _facts(tasks: list[Task]) -> str:
    """Emit the task facts: the total label table and the support, per task."""
    lines: list[str] = []
    for j, t in enumerate(tasks):
        grid = t.all_tuples()
        labels = t.label_of(grid)
        for row, y in zip(grid, labels, strict=True):
            args = ",".join(str(int(v)) for v in row)
            lines.append(f"flab({j},{args},{int(y)}).")
        for row in t.support:
            args = ",".join(str(int(v)) for v in row)
            lines.append(f"sup({j},{args}).")
    return "\n".join(lines)


def _program(
    tasks: list[Task],
    mode: RSMode,
    closure: RSClosure,
    allow_noninjective: bool,
) -> str:
    """Build the ASP program encoding ``RS(T_1 and ... and T_m)``."""
    k = tasks[0].space.k
    n = tasks[0].space.n_slots

    parts = [f"concept(0..{k - 1}).", _facts(tasks)]

    if mode == "shared":
        # alpha is a total function [k] -> [k]
        parts.append("1 { alpha(D,E) : concept(E) } 1 :- concept(D).")
        amap = [("alpha", "C{i}", "D{i}") for _ in range(n)]
    else:
        for s in range(n):
            parts.append(f"1 {{ alpha{s}(D,E) : concept(E) }} 1 :- concept(D).")
        amap = [(f"alpha{s}", "C{i}", "D{i}") for s in range(n)]

    cs = ",".join(f"C{i}" for i in range(n))
    ds = ",".join(f"D{i}" for i in range(n))
    alpha_lits = ",".join(
        f"{name}({src.format(i=i)},{dst.format(i=i)})" for i, (name, src, dst) in enumerate(amap)
    )

    # The core constraint: a shortcut may never change the label of a support tuple.
    for j in range(len(tasks)):
        parts.append(
            f":- sup({j},{cs}), flab({j},{cs},Y), {alpha_lits}, flab({j},{ds},Y2), Y != Y2."
        )
        if closure == "partial":
            # alpha must also keep support tuples inside the support.
            parts.append(f":- sup({j},{cs}), {alpha_lits}, not sup({j},{ds}).")

    if not allow_noninjective:
        if mode == "shared":
            parts.append(":- alpha(D1,E), alpha(D2,E), D1 != D2.")
        else:
            for s in range(n):
                parts.append(f":- alpha{s}(D1,E), alpha{s}(D2,E), D1 != D2.")

    shown = "alpha" if mode == "shared" else "alpha0"
    parts.append(f"#show {shown}/2." if mode == "shared" else "")
    if mode == "per_slot":
        parts.extend(f"#show alpha{s}/2." for s in range(n))
    return "\n".join(p for p in parts if p)


def rs_set(
    tasks: Task | Sequence[Task],
    *,
    mode: RSMode,
    closure: RSClosure,
    allow_noninjective: bool,
    limit: int | None = 200_000,
) -> RSResult:
    """Enumerate the reasoning-shortcut set with clingo.

    See ``oracle.base.RSOracle`` for the argument contract.
    """
    t0 = time.perf_counter()
    task_list = as_task_list(tasks)
    k = task_list[0].space.k
    n = task_list[0].space.n_slots

    program = _program(task_list, mode, closure, allow_noninjective)
    ctl = clingo.Control(["--models=0", "--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])

    n_maps = 1 if mode == "shared" else n
    found: list[np.ndarray] = []
    truncated = False

    def on_model(model: clingo.Model) -> bool:
        nonlocal truncated
        arr = np.zeros((n_maps, k), dtype=np.int8)
        for sym in model.symbols(shown=True):
            name = sym.name
            slot = 0 if name == "alpha" else int(name[5:])
            d, e = sym.arguments
            arr[slot, d.number] = e.number
        found.append(arr[0] if mode == "shared" else arr)
        if limit is not None and len(found) >= limit:
            truncated = True
            return False
        return True

    ctl.solve(on_model=on_model)

    if found:
        maps = np.array(found, dtype=np.int8)
    else:
        maps = np.zeros((0, k) if mode == "shared" else (0, n, k), dtype=np.int8)

    return RSResult(
        maps=maps,
        count=len(found),
        truncated=truncated,
        backend="asp",
        elapsed_s=time.perf_counter() - t0,
        mode=mode,
        closure=closure,
    )
