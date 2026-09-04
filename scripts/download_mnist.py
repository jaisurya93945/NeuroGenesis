#!/usr/bin/env python
"""Fetch and verify the MNIST idx files into ``data/raw/``."""

import sys

from neurogenesis.data.mnist import DEFAULT_ROOT, download, load

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    print(f"downloading MNIST into {root} ...")
    download(root)
    d = load(root)
    print(f"train {d.train_x.shape} labels {d.train_y.shape}")
    print(f"test  {d.test_x.shape} labels {d.test_y.shape}")
    print("per-digit train counts:", [int((d.train_y == i).sum()) for i in range(10)])
    print("OK -- all four files checksum-verified")
