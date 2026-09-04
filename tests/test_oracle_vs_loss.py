"""The test that binds the oracle to the objective. If this fails, nothing else matters.

The oracle makes a *combinatorial* claim: these relabellings preserve every label
on the support. The trainer optimises a *numerical* objective: the marginalised
negative log-likelihood. Those are only the same statement if the definitional
choices in ``oracle/base.py`` -- total vs partial ``f``, shared vs per-slot maps,
what "the support" means -- line up exactly with how the loss is computed.

So this module hand-constructs, for a given ``alpha``, the encoder that emits
``alpha(ground truth)`` exactly, pushes it through the **real** ``semantic_nll``,
and asserts:

    loss == 0 and Acc(Y) == 1     if and only if     alpha in RS(T)

A definitional error anywhere in the oracle shows up here as a shortcut that the
oracle blesses but the loss punishes, or vice versa. This is the reason the
project builds the oracle before it builds any experiment.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.needs_torch

from tests.test_oracle_differential import random_task  # noqa: E402

from neurogenesis.generators.algebraic import addition_task, modular_task  # noqa: E402
from neurogenesis.models.nesy import NeSyModel, TabularEncoder  # noqa: E402
from neurogenesis.oracle import enumerate as en  # noqa: E402

ARGS = dict(mode="shared", closure="total", allow_noninjective=True, limit=200_000)


def _loss_and_label_acc(task, alpha: np.ndarray) -> tuple[float, float]:
    """Realise ``alpha`` as an encoder and evaluate the real loss on the support."""
    k = task.space.k
    model = NeSyModel(TabularEncoder(torch.from_numpy(np.asarray(alpha)), k), task)
    x = torch.tensor(np.asarray(task.support), dtype=torch.long)
    y = torch.tensor(np.asarray(task.support_labels), dtype=torch.long)
    loss = float(model.loss(x, y))
    pred, _ = model.predict(x)
    return loss, float((pred == y).float().mean())


TASKS = [
    addition_task(k=6, n_slots=2),
    addition_task(k=10, n_slots=2),
    modular_task([1, 9], 10),
    modular_task([1, 4], 10),
    modular_task([1, 1], 6),
]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.name)
def test_every_oracle_shortcut_achieves_zero_loss(task):
    """Necessity: everything the oracle calls a shortcut really is loss-free."""
    rs = en.rs_set(task, **ARGS)
    assert rs.count >= 1
    for alpha in rs.maps:
        loss, acc_y = _loss_and_label_acc(task, alpha)
        assert loss < 1e-5, f"oracle blessed alpha={alpha.tolist()} but loss={loss:.4g}"
        assert acc_y == 1.0, f"oracle blessed alpha={alpha.tolist()} but Acc(Y)={acc_y}"


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.name)
def test_every_non_shortcut_incurs_loss(task):
    """Sufficiency: everything the oracle *excludes* really does cost loss.

    Without this direction the oracle could pass the test above by simply
    declaring every map a shortcut.
    """
    rs = en.rs_set(task, **ARGS)
    in_rs = {m.tobytes() for m in rs.maps}
    k = task.space.k
    rng = np.random.default_rng(0)

    checked = 0
    for _ in range(300):
        alpha = rng.integers(0, k, size=k).astype(np.int8)
        if alpha.tobytes() in in_rs:
            continue
        loss, _ = _loss_and_label_acc(task, alpha)
        assert loss > 1e-3, f"alpha={alpha.tolist()} not in RS but loss={loss:.4g}"
        checked += 1
        if checked >= 40:
            break
    assert checked > 0, "no non-shortcut sampled; test would be vacuous"


def test_exhaustive_equivalence_on_small_tasks():
    """The full iff, checked over *every* map, on tasks small enough to enumerate."""
    for task in [addition_task(k=4, n_slots=2), modular_task([1, 2], 4), modular_task([1, 3], 4)]:
        k = task.space.k
        in_rs = {m.tobytes() for m in en.rs_set(task, **ARGS).maps}
        for alpha_t in itertools.product(range(k), repeat=k):
            alpha = np.array(alpha_t, dtype=np.int8)
            loss, acc_y = _loss_and_label_acc(task, alpha)
            oracle_says = alpha.tobytes() in in_rs
            loss_says = loss < 1e-5
            assert oracle_says == loss_says, (
                f"{task.name}: alpha={alpha_t} oracle={oracle_says} loss={loss:.4g} -> disagreement"
            )
            if oracle_says:
                assert acc_y == 1.0


def test_equivalence_on_random_sparse_support_tasks():
    """The riskiest case: sparse support, where total-vs-partial semantics could diverge."""
    rng = np.random.default_rng(11)
    for i in range(25):
        k = int(rng.integers(3, 6))
        task = random_task(rng, k, 2, float(rng.choice([0.25, 0.5])))
        in_rs = {m.tobytes() for m in en.rs_set(task, **ARGS).maps}
        for alpha_t in itertools.product(range(k), repeat=k):
            alpha = np.array(alpha_t, dtype=np.int8)
            loss, _ = _loss_and_label_acc(task, alpha)
            assert (alpha.tobytes() in in_rs) == (loss < 1e-5), (
                f"case {i}: alpha={alpha_t} disagreement (loss={loss:.4g})"
            )
