"""A third, deliberately stupid reference implementation.

The DFS backend is fast because it prunes, and the ASP backend is fast because
clingo is clever. Both could in principle be *identically* wrong about what the
pruning or the encoding means. This module defines the RS set the most literal way
possible -- enumerate every map in ``[k]**[k]``, check every support tuple, no
cleverness at all -- and asserts the two real backends agree with it.

Restricted to ``k <= 5`` (``5**5 = 3125`` maps) so it stays fast.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from neurogenesis.generators.algebraic import addition_task, modular_task
from neurogenesis.generators.random_tasks import random_task
from neurogenesis.oracle import asp
from neurogenesis.oracle import enumerate as en
from neurogenesis.tasks import Task


def naive_rs_set(task: Task, *, closure: str = "total") -> np.ndarray:
    """Literal enumeration. No pruning, no solver, no shared code with the backends."""
    k = task.space.k
    support = [tuple(int(v) for v in row) for row in task.support]
    support_set = set(support)
    out = []
    for alpha in itertools.product(range(k), repeat=k):
        ok = True
        for c in support:
            mapped = tuple(alpha[v] for v in c)
            if task.label_table[mapped] != task.label_table[c]:
                ok = False
                break
            if closure == "partial" and mapped not in support_set:
                ok = False
                break
        if ok:
            out.append(alpha)
    return np.array(sorted(out), dtype=np.int8).reshape(-1, k)


@pytest.mark.parametrize("closure", ["total", "partial"])
def test_naive_matches_both_backends_on_random_tasks(closure):
    rng = np.random.default_rng(7)
    args = dict(mode="shared", closure=closure, allow_noninjective=True, limit=200_000)
    for i in range(60):
        k = int(rng.integers(2, 6))
        n = int(rng.choice([2, 3]))
        task = random_task(rng, k, n, float(rng.choice([0.3, 0.6, 1.0])))
        want = naive_rs_set(task, closure=closure)
        np.testing.assert_array_equal(
            en.rs_set(task, **args).sorted_maps(), want, err_msg=f"dfs {i}"
        )
        np.testing.assert_array_equal(
            asp.rs_set(task, **args).sorted_maps(), want, err_msg=f"asp {i}"
        )


def test_naive_matches_on_structured_tasks():
    """Also check the tasks we actually care about, not only random ones."""
    args = dict(mode="shared", closure="total", allow_noninjective=True)
    for task in [addition_task(k=5, n_slots=2), modular_task([1, 4], 5), modular_task([1, 1], 5)]:
        want = naive_rs_set(task)
        np.testing.assert_array_equal(en.rs_set(task, **args).sorted_maps(), want)
        np.testing.assert_array_equal(asp.rs_set(task, **args).sorted_maps(), want)
