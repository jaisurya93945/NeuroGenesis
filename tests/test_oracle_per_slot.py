"""Per-slot relabellings: the setting only the ASP backend supports."""

from __future__ import annotations

import itertools

import pytest

from neurogenesis.generators.algebraic import addition_task, modular_task
from neurogenesis.oracle import asp
from neurogenesis.oracle import enumerate as en

PER_SLOT = dict(mode="per_slot", closure="total", allow_noninjective=True)
SHARED = dict(mode="shared", closure="total", allow_noninjective=True)


def naive_per_slot_count(task) -> int:
    """Literal enumeration over independent per-slot maps, for k**k*n small."""
    k, n = task.space.k, task.space.n_slots
    support = [tuple(int(v) for v in row) for row in task.support]
    count = 0
    for maps in itertools.product(itertools.product(range(k), repeat=k), repeat=n):
        if all(
            task.label_table[tuple(maps[j][c[j]] for j in range(n))] == task.label_table[c]
            for c in support
        ):
            count += 1
    return count


def test_dfs_backend_refuses_per_slot_rather_than_guessing():
    with pytest.raises(NotImplementedError):
        en.rs_set(addition_task(k=4), **PER_SLOT)


@pytest.mark.parametrize("task", [addition_task(k=3, n_slots=2), modular_task([1, 2], 3)])
def test_asp_per_slot_matches_naive(task):
    r = asp.rs_set(task, **PER_SLOT)
    assert r.maps.shape == (r.count, task.space.n_slots, task.space.k)
    assert r.count == naive_per_slot_count(task)


def test_per_slot_is_at_least_as_permissive_as_shared():
    """Every shared shortcut is a per-slot shortcut with all slots equal."""
    for task in [addition_task(k=4, n_slots=2), modular_task([1, 3], 4)]:
        assert asp.rs_set(task, **PER_SLOT).count >= asp.rs_set(task, **SHARED).count


def test_integer_addition_stays_per_slot_identifiable():
    """Integer addition resists the opposing-shift attack, because of its range.

    The natural per-slot shortcut for y = c1 + c2 is alpha0(x) = x+t paired with
    alpha1(x) = x-t, which preserves every label. But alpha must map [k] into [k],
    and a non-zero shift sends an endpoint out of range, so full-support integer
    addition is identifiable under per-slot maps too. (Verified against the naive
    enumerator above.)
    """
    task = addition_task(k=4, n_slots=2)
    assert asp.rs_set(task, **SHARED).count == 1
    assert asp.rs_set(task, **PER_SLOT).count == 1


def test_modular_addition_is_not_per_slot_identifiable():
    """Wrap-around removes the range obstruction, so opposing shifts survive.

    This is the clean demonstration that shared-identifiable does not imply
    per-slot identifiable -- and hence why `mode` is a mandatory argument rather
    than something the oracle guesses.
    """
    task = modular_task([1, 1], 4)
    shared = asp.rs_set(task, **SHARED)
    per_slot = asp.rs_set(task, **PER_SLOT)
    assert per_slot.count > shared.count
    assert per_slot.count == naive_per_slot_count(task)


def test_per_slot_permutation_count_handles_stacked_maps():
    """Regression: n_permutations must not treat an (n_slots, k) stack as a vector."""
    r = asp.rs_set(modular_task([1, 1], 4), **PER_SLOT)
    assert r.maps.ndim == 3
    assert r.n_permutations + r.n_collapsing == r.count
    # every modular opposing-shift shortcut is a genuine permutation pair
    assert r.n_permutations >= 4
