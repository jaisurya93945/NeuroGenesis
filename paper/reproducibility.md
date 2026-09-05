# Reproducibility statement

## Everything ships

Raw run records (`results/runs/*.jsonl`) are committed, not just summaries. Every table and figure
re-derives from them **in seconds, without retraining anything**:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pip install "torch>=2.9,<2.15"
.venv/bin/python scripts/download_mnist.py
bash scripts/reproduce_all.sh
```

Each record carries its config hash, git commit, seeds, package versions, hardware and runtime.
Re-running an existing config hash is a no-op, so long sweeps resume safely.

## Verification the reader can run

- `pytest tests/ -q` — the full suite, including the oracle gates: the closed-form `gcd` grid, the
  DFS-vs-ASP differential test over 2000 comparisons, the naive-reference cross-check, and the
  oracle↔loss equivalence (`alpha` is a shortcut **iff** the corresponding encoder reaches zero loss).
- `bash scripts/smoke.sh` — end-to-end in minutes.
- CI runs the oracle gates **without torch installed**, so the tests that protect the science do not
  depend on a 4 GB CUDA wheel.

## Known limits, stated rather than smoothed

- **Floating-point non-determinism** across BLAS versions and thread counts moves `Acc(C)` in the
  last decimal. Conclusions come from distributions over >= 8 seeds, never point values.
- **The oracle is exact and deterministic**; oracle results reproduce bit-for-bit.
- **Superseded records are archived, not deleted.** Two runs were invalidated by the seeding bug
  described in `experiments.md` and re-run; the pre-fix records remain in
  `results/runs/archived_*.jsonl`, explicitly marked as the basis of no current claim.
- **No pretrained perception.** The environment cannot download model weights, so perceptual
  difficulty is manipulated synthetically rather than by using realistic encoders. This is a real
  gap relative to where the field is heading and is listed in `LIMITATIONS.md`.
- **Literature was gathered from search summaries, not primary PDFs** (arXiv is blocked). Every
  citation is marked `[S]` until re-read.
