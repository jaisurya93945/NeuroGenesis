# results/

## Policy

- `runs/runs.jsonl` — **committed.** Append-only raw run records, one JSON object per run, each
  carrying its config hash, git commit, seeds, package versions, hardware and full metrics.
  Committing them means anyone can re-derive every table and figure *without retraining anything*:
  `python scripts/analyze_e1.py` runs in seconds against this file.
  Kept committed while the file stays small (currently ~100 KB for 40 runs). If a sweep pushes it
  past a few MB it moves to a compressed archive and this note gets updated — the principle is that
  the evidence ships with the claims, not that the file is unconditionally tracked.
- `e1_summary.json` — **committed.** Derived; regenerate with `scripts/analyze_e1.py`.
- `figures/` — **committed.** Derived; regenerate with the same script.
- `oracle_cache/` — **committed.** The oracle is exact and deterministic, so caching lets
  oracle-derived tables reproduce without clingo installed.

## Rules

Raw records are **never edited by hand**. Every number in `RESULTS.md` and in the paper is produced
by a script reading this directory. If a number cannot be traced to a record here, it does not go
in the paper.
