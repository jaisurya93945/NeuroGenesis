"""Selection: the set-cover structure, and that the baselines are real competition."""

from __future__ import annotations

import numpy as np

from neurogenesis.generators.algebraic import modular_task
from neurogenesis.generators.pool import (
    build_pool,
    divisor_modular_pool,
    individually_insufficient,
)
from neurogenesis.oracle import enumerate as en
from neurogenesis.selection import (
    exhaustive_optimal,
    greedy_cover,
    information_greedy,
    random_selection,
)

A = dict(mode="shared", closure="total", allow_noninjective=True)


def test_conjunction_shrinks_rs_monotonically():
    """Adding a task can only remove shortcuts -- the property greedy relies on."""
    base = modular_task([1, 9], 10)
    pool = divisor_modular_pool(10)
    prev = en.rs_set(base, **A).count
    chosen = [base]
    for t in pool:
        chosen.append(t)
        cur = en.rs_set(chosen, **A).count
        assert cur <= prev
        prev = cur


def test_greedy_reaches_identifiability_and_matches_optimum():
    base = modular_task([1, 9], 10)
    pool = divisor_modular_pool(10)
    g = greedy_cover(base, pool, budget=3)
    e = exhaustive_optimal(base, pool, budget=3)
    assert g.identifiable
    assert g.final_rs_count == e.final_rs_count, "greedy must match the optimum here"


def test_selection_result_counts_match_a_fresh_oracle_call():
    """Guard against the incremental survivor filter drifting from the real oracle."""
    base = modular_task([1, 9], 10)
    pool = divisor_modular_pool(10)
    g = greedy_cover(base, pool, budget=3)
    fresh = en.rs_set([base] + [pool[j] for j in g.chosen], **A)
    assert fresh.count == g.final_rs_count


def test_information_greedy_is_a_real_baseline_not_a_strawman():
    """It must be able to win somewhere, or it proves nothing when it loses.

    On the generic pool every method succeeds immediately; that is precisely why
    the generic pool cannot be used for E3 (DECISIONS.md D7).
    """
    base = modular_task([1, 9], 10)
    generic = build_pool(10)
    assert information_greedy(base, generic, budget=1).identifiable
    assert greedy_cover(base, generic, budget=1).identifiable


def test_shift_invariant_distractors_eliminate_nothing():
    """The distractors must genuinely be distractors, not accidentally useful."""
    base = modular_task([1, 9], 10)
    n0 = en.rs_set(base, **A).count
    for t in divisor_modular_pool(10):
        if t.meta.get("shift_invariant"):
            assert en.rs_set([base, t], **A).count == n0


def test_individually_insufficient_filter():
    base = modular_task([1, 9], 10)
    kept = individually_insufficient(base, build_pool(10))
    for t in kept:
        assert not en.rs_set([base, t], **A).is_identifiable


def test_random_selection_is_reproducible():
    base = modular_task([1, 9], 10)
    pool = divisor_modular_pool(10)
    a = random_selection(base, pool, 2, np.random.default_rng(7))
    b = random_selection(base, pool, 2, np.random.default_rng(7))
    assert a.chosen == b.chosen
