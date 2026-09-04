# results/

- `runs/` — append-only JSONL run records. **Gitignored** (large, machine-generated); regenerate
  with `scripts/run_sweep.py`.
- `oracle_cache/` — **committed**. The RS oracle is exact and deterministic, so caching it lets
  every oracle-derived table reproduce without clingo installed and without re-running the search.
- `figures/` — gitignored; regenerate with `scripts/make_figures.py`.

Raw records are never edited by hand. Every table and figure in the paper is generated from them.
