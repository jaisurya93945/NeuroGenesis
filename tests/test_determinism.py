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
