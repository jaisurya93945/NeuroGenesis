"""Run reproducibility: a recorded seed must actually determine the run.

Regression test for a real bug. ``runner.run`` constructed the encoder *before*
``train`` called ``torch.manual_seed``, so weight initialisation was drawn from
whatever global RNG state happened to exist. Runs recorded an ``init_seed`` that
did not in fact determine them, and two invocations of an identical config could
differ by 0.9 in ``Acc(C)`` -- which is how it was noticed.

The whole provenance story depends on this: a result that cannot be regenerated
from its recorded seed is not reproducible, whatever the manifest claims.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytestmark = [pytest.mark.needs_torch]

from neurogenesis.config import DataSpec, ModelSpec, OptimSpec, RunConfig, TaskSpec  # noqa: E402
from neurogenesis.runner import run  # noqa: E402


def _cfg(init_seed: int, data_seed: int = 0) -> RunConfig:
    return RunConfig(
        experiment="determinism_test",
        task=TaskSpec(family="algebraic", weights=(1, 2), m=10),
        data=DataSpec(
            tier="S", n_train=1500, n_val=300, n_test=300, noise=0.3, dim=32, data_seed=data_seed
        ),
        model=ModelSpec(encoder="mlp", init_seed=init_seed),
        optim=OptimSpec(epochs=3),
    )


def test_same_seed_reproduces_exactly():
    a = run(_cfg(0), store=None)["metrics"]["test"]
    b = run(_cfg(0), store=None)["metrics"]["test"]
    assert a["acc_c"] == b["acc_c"]
    assert a["acc_y"] == b["acc_y"]
    assert a["alpha_hat"] == b["alpha_hat"]


def test_different_init_seed_gives_a_different_run():
    """Otherwise 'N seeds' would be N copies of one run, and every CI would be fake."""
    a = run(_cfg(0), store=None)["metrics"]["test"]
    b = run(_cfg(1), store=None)["metrics"]["test"]
    assert (a["acc_c"], a["alpha_hat"]) != (b["acc_c"], b["alpha_hat"])


def test_different_data_seed_gives_a_different_run():
    a = run(_cfg(0, data_seed=0), store=None)["metrics"]["test"]
    b = run(_cfg(0, data_seed=1), store=None)["metrics"]["test"]
    assert a["acc_c"] != b["acc_c"] or a["alpha_hat"] != b["alpha_hat"]


def test_build_encoder_seed_controls_initialisation():
    """The structural guard: seeding must be an argument, not a caller convention.

    This project shipped the same reproducibility bug twice -- ``runner.run`` and
    then the E3 driver both built the encoder before seeding torch, leaving weight
    initialisation uncontrolled. Both times it was invisible until two supposedly
    identical configs disagreed. ``build_encoder(seed=...)`` removes the ordering
    hazard; this test keeps it removed.
    """
    import torch

    from neurogenesis.models.encoders import build_encoder

    def weights(seed: int | None) -> torch.Tensor:
        enc = build_encoder("mlp", k=6, in_dim=32, seed=seed)
        return next(enc.parameters()).detach().clone()

    torch.manual_seed(999)
    a = weights(0)
    torch.manual_seed(12345)  # deliberately disturb global RNG state in between
    b = weights(0)
    assert torch.equal(a, b), "same seed must give identical initial weights"
    assert not torch.equal(a, weights(1)), "different seeds must differ"


def test_multitask_training_is_deterministic():
    """The E3 path, which is where the bug recurred."""
    import numpy as np

    from neurogenesis.data.tuples import make_synthetic_codebook, render_synthetic
    from neurogenesis.generators.algebraic import modular_task
    from neurogenesis.models.encoders import build_encoder
    from neurogenesis.train.loop import TrainConfig, train_multitask

    base = modular_task([1, 5], 6)
    book = make_synthetic_codebook(6, 32, np.random.default_rng(12345))

    def once() -> float:
        rng = np.random.default_rng(0)
        tr = render_synthetic(base, 1200, book, 0.1, rng)
        te = render_synthetic(base, 400, book, 0.1, rng, "test")
        enc = build_encoder("mlp", k=6, in_dim=32, seed=0)
        res = train_multitask(
            [base], enc, tr, {"test": te}, TrainConfig(epochs=3, encoder="mlp"), 0
        )
        return res.metrics["test"].acc_c

    assert once() == once()
