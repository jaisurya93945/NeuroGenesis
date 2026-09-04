"""Config identity, the leakage guard, and the statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from neurogenesis.config import (
    CONFIRMATORY_SEED_FLOOR,
    DataSpec,
    ModelSpec,
    RunConfig,
    TaskSpec,
    environment_fingerprint,
)
from neurogenesis.stats import (
    bootstrap_diff,
    bootstrap_mean,
    cliffs_delta,
    permutation_trend_test,
    spearman,
)


def test_config_hash_is_stable_and_content_sensitive():
    a = RunConfig(task=TaskSpec(weights=(1, 9)))
    b = RunConfig(task=TaskSpec(weights=(1, 9)))
    c = RunConfig(task=TaskSpec(weights=(1, 4)))
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()


def test_seed_changes_the_hash():
    """Otherwise a sweep would dedup distinct seeds into one run."""
    a = RunConfig(model=ModelSpec(init_seed=0))
    b = RunConfig(model=ModelSpec(init_seed=1))
    assert a.config_hash() != b.config_hash()
    assert RunConfig(data=DataSpec(data_seed=0)).config_hash() != (
        RunConfig(data=DataSpec(data_seed=1)).config_hash()
    )


def test_confirmatory_seeds_cannot_be_used_for_tuning():
    """The leakage guard must raise, not warn -- it is the only thing enforcing this."""
    with pytest.raises(ValueError, match="confirmatory"):
        RunConfig(task=TaskSpec(seed=CONFIRMATORY_SEED_FLOOR), tuning_mode=True)
    # allowed: confirmatory seed outside tuning mode, and dev seed inside it
    RunConfig(task=TaskSpec(seed=CONFIRMATORY_SEED_FLOOR), tuning_mode=False)
    RunConfig(task=TaskSpec(seed=5), tuning_mode=True)


def test_environment_fingerprint_has_provenance_fields():
    env = environment_fingerprint()
    assert set(env) >= {"git_commit", "python", "platform", "versions"}
    assert "numpy" in env["versions"]


def test_bootstrap_interval_covers_the_mean():
    v = np.full(10, 0.5)
    est = bootstrap_mean(v)
    assert est.mean == pytest.approx(0.5)
    assert est.lo == pytest.approx(0.5) and est.hi == pytest.approx(0.5)


def test_bootstrap_diff_sign():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 0.01, 20), rng.normal(0.0, 0.01, 20)
    d = bootstrap_diff(a, b)
    assert d.mean > 0.9 and d.lo > 0


def test_cliffs_delta_bounds():
    a, b = np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])
    assert cliffs_delta(a, b) == pytest.approx(1.0)
    assert cliffs_delta(b, a) == pytest.approx(-1.0)
    assert cliffs_delta(a, a) == pytest.approx(0.0)


def test_spearman_on_monotone_data():
    x = np.arange(10)
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_permutation_trend_detects_decrease_and_not_noise():
    rng = np.random.default_rng(0)
    decreasing = [rng.normal(m, 0.01, 10) for m in (0.99, 0.7, 0.4, 0.05)]
    assert permutation_trend_test(decreasing, n_perm=2000) < 0.01
    flat = [rng.normal(0.5, 0.01, 10) for _ in range(4)]
    assert permutation_trend_test(flat, n_perm=2000) > 0.01
