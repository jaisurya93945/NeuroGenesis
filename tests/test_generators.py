"""Generator contracts: what each family guarantees about its shortcut structure."""

from __future__ import annotations

import numpy as np
import pytest

from neurogenesis.generators.algebraic import addition_task, modular_task
from neurogenesis.generators.planted import cyclic_group, planted_task, random_planted
from neurogenesis.generators.support import (
    greedy_minimal_identifying_support,
    rarefy_against,
    refuting_tuples,
    swap_map,
    thin_support,
)
from neurogenesis.oracle import enumerate as en
from neurogenesis.oracle.measures import margin, margin_bruteforce

A = dict(mode="shared", closure="total", allow_noninjective=True)


def test_margin_is_positive_iff_identifiable():
    """The defining property that makes margin a refinement of the binary test."""
    for w in (2, 1, 4, 9):
        t = modular_task([1, w], 10)
        ident = en.rs_set(t, **A).is_identifiable
        assert (margin(t).margin > 0) == ident


@pytest.mark.parametrize(
    "task",
    [addition_task(k=4), addition_task(k=5), modular_task([1, 2], 5), modular_task([1, 4], 5)],
    ids=lambda t: t.name,
)
def test_margin_asp_matches_bruteforce(task):
    assert margin(task).margin == pytest.approx(margin_bruteforce(task), abs=1e-9)


def test_thin_support_never_shrinks_rs():
    t = addition_task(k=6)
    rng = np.random.default_rng(0)
    base = en.rs_set(t, **A).count
    for d in (0.8, 0.5, 0.3):
        assert en.rs_set(thin_support(t, d, rng), **A).count >= base


def test_threadbare_preserves_identifiability_and_shrinks_support():
    t = addition_task(k=8)
    tb = greedy_minimal_identifying_support(t, np.random.default_rng(0))
    assert en.rs_set(tb, **A).is_identifiable
    assert len(tb.support) < len(t.support)


def test_rarefy_keeps_identifiability_while_driving_margin_down():
    """The decisive H2 instrument: same support, same data volume, tiny margin."""
    t = addition_task(k=10)
    alpha = swap_map(10, 0, 1)
    prev = margin(t).margin
    for rarity in (0.1, 0.01, 0.001):
        r = rarefy_against(t, alpha, rarity)
        assert len(r.support) == len(t.support), "support size must be untouched"
        assert en.rs_set(r, **A).is_identifiable, "must stay provably identifiable"
        m = margin(r).margin
        assert 0 < m < prev, "margin must strictly decrease and stay positive"
        prev = m


def test_rarefy_rejects_an_alpha_that_is_already_a_shortcut():
    t = modular_task([1, 9], 10)  # every cyclic shift is a shortcut
    shift = (np.arange(10) + 1) % 10
    assert len(refuting_tuples(t, shift)) == 0
    with pytest.raises(ValueError, match="already a shortcut"):
        rarefy_against(t, shift)


def test_planted_rs_contains_the_planted_maps():
    """RS >= G by construction. RS may be strictly larger; that is recorded, not rejected."""
    rng = np.random.default_rng(0)
    for _ in range(8):
        task, g = random_planted(5, 2, rng)
        rs = en.rs_set(task, **A)
        assert rs.count >= g


def test_planted_maps_are_actually_shortcuts():
    """Stronger than the count check: each planted map must be *in* the recovered set."""
    rng = np.random.default_rng(1)
    maps = cyclic_group(6, 2)
    task = planted_task(6, 2, maps, rng)
    rs = en.rs_set(task, **A)
    for m in maps:
        assert rs.contains(m.astype(np.int8)), f"planted map {m.tolist()} missing from RS"
