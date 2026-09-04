"""Known-answer tests binding the oracle to analytically-derived RS counts.

These are CI gates. If any of them fails, nothing downstream in the project means
anything, so they are deliberately cheap and run first.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from neurogenesis.generators.algebraic import addition_task, analytic_rs_count, modular_task
from neurogenesis.oracle import enumerate as en

ARGS = dict(mode="shared", closure="total", allow_noninjective=True)


def test_mnist_addition_is_identifiable():
    """Full-support integer addition admits only the identity (published result)."""
    r = en.rs_set(addition_task(k=10, n_slots=2), **ARGS)
    assert r.count == 1
    assert r.is_identifiable
    np.testing.assert_array_equal(r.maps[0], np.arange(10))


@pytest.mark.parametrize("w", range(1, 10))
def test_mod10_family_matches_gcd(w):
    """The E1 headline family: |RS| = gcd(1 + w, 10), all shortcuts cyclic shifts."""
    t = modular_task([1, w], 10)
    r = en.rs_set(t, **ARGS)
    assert r.count == math.gcd(1 + w, 10)
    for m in r.maps:
        shift = (m[0] - 0) % 10
        np.testing.assert_array_equal(m, (np.arange(10) + shift) % 10)


@pytest.mark.parametrize("m", range(5, 13))
@pytest.mark.parametrize("n", [2, 3])
def test_gcd_formula_grid(m, n):
    """|RS| = gcd(sum w, m) across the full (m, n, w) grid where the precondition holds."""
    coprime = [w for w in range(1, m) if math.gcd(w, m) == 1]
    for weights in itertools.product(coprime, repeat=n):
        predicted = analytic_rs_count(weights, m)
        assert predicted is not None
        r = en.rs_set(modular_task(list(weights), m), **ARGS)
        assert r.count == predicted, f"m={m} w={weights}: {r.count} != {predicted}"


def test_non_coprime_weights_have_no_closed_form():
    """The closed form correctly declines when its precondition fails."""
    assert analytic_rs_count([2, 4], 10) is None


def test_rs_is_a_monoid():
    """RS contains the identity and is closed under composition."""
    r = en.rs_set(modular_task([1, 9], 10), **ARGS)
    k = 10
    assert r.contains(np.arange(k))
    for a in r.maps:
        for b in r.maps:
            assert r.contains(a[b]), "RS must be closed under composition"


def test_intersection_law():
    """RS(T1 and T2) == RS(T1) intersect RS(T2) -- the basis of set-cover selection."""
    t1, t2 = modular_task([1, 9], 10), modular_task([1, 4], 10)
    joint = en.rs_set([t1, t2], **ARGS)
    r1 = en.rs_set(t1, **ARGS)
    r2 = en.rs_set(t2, **ARGS)
    np.testing.assert_array_equal(joint.sorted_maps(), (r1 & r2).sorted_maps())
    # gcd(10,10)=10 shifts, gcd(5,10)=5 shifts -> intersection is the 5 common ones
    assert joint.count == 5


def test_adding_support_never_grows_rs():
    """Monotonicity: more evidence can only shrink the shortcut set."""
    full = addition_task(k=6, n_slots=2)
    rng = np.random.default_rng(0)
    grid = full.all_tuples()
    small = grid[rng.choice(len(grid), size=12, replace=False)]
    partial = addition_task(k=6, n_slots=2, support=small)
    assert en.rs_set(partial, **ARGS).count >= en.rs_set(full, **ARGS).count
