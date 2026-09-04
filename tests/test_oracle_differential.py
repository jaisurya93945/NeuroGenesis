"""Differential test: the DFS and ASP backends must agree exactly.

The two implementations share no code and no algorithm -- one is a hand-written
pruned depth-first search over relabellings, the other is a declarative ASP
encoding solved by clingo. Agreement across thousands of randomly generated tasks
is the strongest cheap evidence that the definition in ``oracle/base.py`` is
implemented faithfully, and it is the project's main defence against its #1 risk
(a *definitional* oracle error that yields plausible numbers for the wrong question).
"""

from __future__ import annotations

import numpy as np
import pytest

from neurogenesis.generators.random_tasks import random_task
from neurogenesis.oracle import asp
from neurogenesis.oracle import enumerate as en

# k <= 6 keeps the worst case (k**k = 46656 shortcuts under a nearly-empty support)
# comfortably below the truncation limit, so counts are always exact and comparable.
LIMIT = 200_000


def _cases():
    """500 random tasks crossed with both injectivity settings and both closures."""
    rng = np.random.default_rng(20260904)
    out = []
    for i in range(500):
        k = int(rng.integers(2, 7))
        n = int(rng.choice([2, 3]))
        density = float(rng.choice([0.2, 0.5, 1.0]))
        out.append((i, random_task(rng, k, n, density)))
    return out


CASES = _cases()


@pytest.mark.parametrize(
    ("closure", "allow_noninjective"),
    [("total", True), ("total", False), ("partial", True), ("partial", False)],
)
def test_backends_agree_on_random_tasks(closure, allow_noninjective):
    args = dict(mode="shared", closure=closure, allow_noninjective=allow_noninjective, limit=LIMIT)
    for i, task in CASES:
        a = asp.rs_set(task, **args)
        e = en.rs_set(task, **args)
        assert not a.truncated and not e.truncated, f"case {i} truncated; raise LIMIT"
        assert a.count == e.count, (
            f"case {i} ({task.name}, closure={closure}, "
            f"noninj={allow_noninjective}): asp={a.count} dfs={e.count}"
        )
        np.testing.assert_array_equal(
            a.sorted_maps(), e.sorted_maps(), err_msg=f"case {i}: map sets differ"
        )


def test_identity_always_present_in_both_backends():
    """A sanity invariant that must hold for every task, under every setting."""
    args = dict(mode="shared", closure="total", allow_noninjective=True, limit=LIMIT)
    for i, task in CASES[:100]:
        k = task.space.k
        ident = np.arange(k, dtype=np.int8)
        assert asp.rs_set(task, **args).contains(ident), f"case {i}"
        assert en.rs_set(task, **args).contains(ident), f"case {i}"


def test_partial_closure_is_a_subset_of_total():
    """The partial reading can only ever disqualify maps, never admit new ones."""
    for i, task in CASES[:150]:
        tot = en.rs_set(task, mode="shared", closure="total", allow_noninjective=True, limit=LIMIT)
        par = en.rs_set(
            task, mode="shared", closure="partial", allow_noninjective=True, limit=LIMIT
        )
        assert par.count <= tot.count, f"case {i}"
        total_set = {m.tobytes() for m in tot.maps}
        assert all(m.tobytes() in total_set for m in par.maps), f"case {i}"


def test_injective_restriction_is_a_subset():
    """Permutation-only shortcuts are a subset of all shortcuts."""
    for i, task in CASES[:150]:
        allm = en.rs_set(task, mode="shared", closure="total", allow_noninjective=True, limit=LIMIT)
        perm = en.rs_set(
            task, mode="shared", closure="total", allow_noninjective=False, limit=LIMIT
        )
        assert perm.count == allm.n_permutations, f"case {i}"
