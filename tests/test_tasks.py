"""Task semantics: construction, validation, hashing, and derived quantities."""

from __future__ import annotations

import numpy as np
import pytest

from neurogenesis.concepts import ConceptSpace
from neurogenesis.generators.algebraic import addition_task, modular_task
from neurogenesis.tasks import Task, task_from_fn


def test_label_of_matches_generating_function():
    t = addition_task(k=6, n_slots=2)
    grid = t.all_tuples()
    np.testing.assert_array_equal(t.label_of(grid), grid.sum(axis=1))


def test_dense_mask_selects_exactly_the_right_tuples():
    t = addition_task(k=5, n_slots=2)
    mask = t.dense_mask()
    assert mask.shape == (5, 5, t.n_labels)
    # each tuple contributes to exactly one label
    assert mask.sum() == 25
    for y in range(t.n_labels):
        got = set(map(tuple, np.argwhere(mask[..., y])))
        want = {(a, b) for a in range(5) for b in range(5) if a + b == y}
        assert got == want


def test_content_hash_is_order_invariant_but_content_sensitive():
    t = addition_task(k=5, n_slots=2)
    grid = t.all_tuples()
    shuffled = Task(
        name="other-name",
        space=t.space,
        label_table=t.label_table,
        support=grid[::-1].copy(),
        n_labels=t.n_labels,
    )
    assert t.content_hash() == shuffled.content_hash(), "row order must not change the hash"
    assert modular_task([1, 9], 10).content_hash() != modular_task([1, 4], 10).content_hash()


def test_duplicate_support_rows_rejected():
    """Duplicated rows would silently double-count evidence in the margin."""
    with pytest.raises(ValueError, match="unique"):
        Task(
            name="dup",
            space=ConceptSpace(k=3, n_slots=2),
            label_table=np.zeros((3, 3), np.int16),
            support=np.array([[0, 0], [0, 0]], np.int16),
            n_labels=1,
        )


def test_out_of_range_support_rejected():
    with pytest.raises(ValueError, match="range"):
        Task(
            name="oob",
            space=ConceptSpace(k=3, n_slots=2),
            label_table=np.zeros((3, 3), np.int16),
            support=np.array([[0, 5]], np.int16),
            n_labels=1,
        )


def test_support_weights_normalise():
    t = task_from_fn(
        "w", k=3, n_slots=1, fn=lambda c: c[0], support_weights=np.array([2.0, 1.0, 1.0])
    )
    np.testing.assert_allclose(t.support_weights.sum(), 1.0)
    np.testing.assert_allclose(t.support_weights, [0.5, 0.25, 0.25])


def test_label_entropy_of_uniform_family_is_log_k():
    """Every member of the mod-10 family has identical label entropy -- the E1 design claim."""
    ents = [modular_task([1, w], 10).label_entropy() for w in range(1, 10)]
    np.testing.assert_allclose(ents, np.full(9, np.log(10)), atol=1e-12)
