"""Split discipline: no MNIST image may appear in two splits."""

from __future__ import annotations

import numpy as np
import pytest

from neurogenesis.data import mnist

pytestmark = pytest.mark.needs_mnist


@pytest.fixture(scope="module")
def data():
    if not mnist.is_available():
        pytest.skip("MNIST cache absent; run scripts/download_mnist.py")
    return mnist.load(auto_download=False)


def test_shapes_and_counts(data):
    assert data.train_x.shape == (60000, 28, 28)
    assert data.test_x.shape == (10000, 28, 28)
    assert data.train_x.dtype == np.float32
    assert data.train_x.min() >= 0.0 and data.train_x.max() <= 1.0


def test_known_digit_histogram(data):
    """The standard MNIST train histogram -- catches a truncated or wrong download."""
    counts = [int((data.train_y == d).sum()) for d in range(10)]
    assert counts == [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]


def test_splits_are_disjoint_and_exhaustive(data):
    tr = set(range(*mnist.TRAIN_RANGE))
    va = set(range(*mnist.VAL_RANGE))
    assert tr.isdisjoint(va)
    assert len(tr) + len(va) == len(data.train_y)
    # the test split is a physically separate file, so disjointness is structural
    assert len(data.split("test")[1]) == 10000


def test_every_digit_present_in_every_split(data):
    for which in ("train", "val", "test"):
        idx = data.indices_by_digit(which)
        assert all(len(i) > 0 for i in idx), f"a digit is missing from {which}"
