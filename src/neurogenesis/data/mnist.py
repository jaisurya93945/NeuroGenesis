"""MNIST loading, with no ``torchvision`` dependency.

Two independent reasons not to use ``torchvision.datasets.MNIST`` here: this
research environment cannot reach ``download.pytorch.org``, and pinning the raw
idx bytes by checksum makes the data dependency explicit and verifiable rather
than delegated to a library that has silently changed mirrors several times.

Checksums below were computed from the files actually downloaded on 2026-09-04
(see ``REPRODUCIBILITY.md``); they are the standard MNIST distribution.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

SHA256 = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}

DEFAULT_ROOT = Path("data/raw")

#: Split boundaries over the 60k training images. The 10k test images are a
#: separate file and are reserved for test tuples only. No image ever appears in
#: two splits -- this is asserted in ``tests/test_data_splits.py``.
TRAIN_RANGE = (0, 50_000)
VAL_RANGE = (50_000, 60_000)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(root: Path = DEFAULT_ROOT, *, force: bool = False) -> Path:
    """Fetch the four idx files, trying each mirror, verifying every checksum."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for fname, want in SHA256.items():
        dest = root / fname
        if dest.exists() and not force and _sha256(dest) == want:
            continue
        last_err: Exception | None = None
        for mirror in MIRRORS:
            try:
                urllib.request.urlretrieve(mirror + fname, dest)  # noqa: S310
                got = _sha256(dest)
                if got != want:
                    raise OSError(f"{fname}: sha256 {got} != expected {want}")
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                last_err = exc
        if last_err is not None:
            raise RuntimeError(f"could not fetch {fname} from any mirror: {last_err}")
    return root


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, n = struct.unpack(">II", fh.read(8))
        if magic == 2051:  # images
            rows, cols = struct.unpack(">II", fh.read(8))
            buf = fh.read(n * rows * cols)
            return np.frombuffer(buf, np.uint8).reshape(n, rows, cols)
        if magic == 2049:  # labels
            return np.frombuffer(fh.read(n), np.uint8)
        raise ValueError(f"{path}: unrecognised idx magic {magic}")


@dataclass(frozen=True)
class MNIST:
    """In-memory MNIST, images as float32 in [0, 1]."""

    train_x: np.ndarray  # (60000, 28, 28) float32
    train_y: np.ndarray  # (60000,) int64
    test_x: np.ndarray  # (10000, 28, 28) float32
    test_y: np.ndarray  # (10000,) int64

    def split(self, which: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(images, labels)`` for ``'train'``, ``'val'`` or ``'test'``."""
        if which == "train":
            lo, hi = TRAIN_RANGE
            return self.train_x[lo:hi], self.train_y[lo:hi]
        if which == "val":
            lo, hi = VAL_RANGE
            return self.train_x[lo:hi], self.train_y[lo:hi]
        if which == "test":
            return self.test_x, self.test_y
        raise ValueError(f"unknown split {which!r}")

    def indices_by_digit(self, which: str, k: int = 10) -> list[np.ndarray]:
        """Per-digit index arrays within a split, for concept-conditioned sampling."""
        _, y = self.split(which)
        return [np.flatnonzero(y == d) for d in range(k)]


def load(root: Path = DEFAULT_ROOT, *, auto_download: bool = True) -> MNIST:
    """Load MNIST from the local cache, downloading it first if needed."""
    root = Path(root)
    missing = [f for f in SHA256 if not (root / f).exists()]
    if missing:
        if not auto_download:
            raise FileNotFoundError(f"missing MNIST files in {root}: {missing}")
        download(root)
    return MNIST(
        train_x=_read_idx(root / "train-images-idx3-ubyte.gz").astype(np.float32) / 255.0,
        train_y=_read_idx(root / "train-labels-idx1-ubyte.gz").astype(np.int64),
        test_x=_read_idx(root / "t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0,
        test_y=_read_idx(root / "t10k-labels-idx1-ubyte.gz").astype(np.int64),
    )


def is_available(root: Path = DEFAULT_ROOT) -> bool:
    """Whether the cache is present, so tests can skip rather than fail."""
    return all((Path(root) / f).exists() for f in SHA256)
