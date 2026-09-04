# REPRODUCIBILITY.md

## Environment

Developed and verified on: Linux x86_64, Python 3.11.15, **4 CPU cores, 15 GB RAM, no GPU**.
Everything in this repo is designed to run to completion on that machine; GPU is a config change,
never a requirement.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install "torch>=2.9,<2.15"
.venv/bin/python scripts/download_mnist.py
.venv/bin/python -m pytest tests/ -q
```

### CPU-only note (a real trap)

The PyPI `torch` linux wheel pulls ~4 GB of NVIDIA CUDA packages and, from 2.14, **fails to import
without them** (`libcublasLt.so not found`) — so `--no-deps` does not work as a slimming trick.
Options: (a) install the full stack, as above (~4 GB, works fine on CPU, `cuda.is_available()` is
just `False`); (b) if you can reach `download.pytorch.org`, use
`pip install torch --index-url https://download.pytorch.org/whl/cpu` for a much smaller install.
This environment cannot reach that index, so (a) is what was verified here.

## Data

MNIST is fetched from two mirrors and every file is sha256-verified against hashes recorded in
`src/neurogenesis/data/mnist.py`:

| File | sha256 |
|---|---|
| `train-images-idx3-ubyte.gz` | `440fcabf…523609` |
| `train-labels-idx1-ubyte.gz` | `3552534a…10255c` |
| `t10k-images-idx3-ubyte.gz` | `8d422c7b…dbc4e6` |
| `t10k-labels-idx1-ubyte.gz` | `f7ae60f9…04aec6` |

Splits: train `0–49999`, val `50000–59999`, test = the separate 10k test file. No image appears in
two splits (asserted in tests).

## Determinism and provenance

- `data_seed` (pairing, subsampling, splits) is separate from `init_seed` (weights, batch order),
  so run-to-run variance can be decomposed into data vs initialisation.
- Every run record carries: commit SHA, config hash (sha256 of canonical JSON), seeds, package
  versions, hardware, wall-clock runtime.
- Results are **append-only JSONL** and are never hand-edited. All tables and figures are generated
  from them by `scripts/`.
- Re-running an existing config hash is a no-op, so sweeps resume safely after interruption.

## Known reproducibility limits

- Floating-point non-determinism across BLAS versions and thread counts means `Acc(C)` may differ
  in the last decimal. Conclusions are drawn from distributions over ≥10 seeds, not point values.
- The oracle is exact and fully deterministic; oracle results should reproduce bit-for-bit.
- `results/oracle_cache/` is committed so oracle-derived tables reproduce without clingo installed.
