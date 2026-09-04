# CLAUDE.md — NeuroGenesis operating memory

Concise and operational. Detail lives in the linked documents.

## Mission
Determine whether **provable identifiability of a neuro-symbolic task predicts empirical symbol
grounding under SGD**, and if so, turn that into a cheap alternative to per-concept supervision.

## Research question
> Does provable identifiability of a task's supervision support — decidable offline, before any
> training — actually translate into empirical symbol grounding? And can we *actively select* a
> minimal set of auxiliary tasks that provably collapses the reasoning-shortcut space?

## Current hypotheses
See `HYPOTHESES.md`. Short form: **H1** `Acc(C)` decreases in `|RS|`; **H2** `margin(T)` predicts
grounding better than binary identifiability; **H3** the association survives controlling for label
informativeness; **H4** greedy RS-cover ≥ information-greedy and ≥ concept supervision at matched cost.

## Status
| | |
|---|---|
| Phase | **M5 complete; E1 run and its convergence caveat diagnosed.** M6–M8 infrastructure built (margin oracle, generators, selection). |
| Experiments run | **E1, run twice** (before/after a seeding fix). Post-fix `Acc(C)` = 0.986 / 0.988 / 0.297 / 0.100 for `\|RS\|` = 1/2/5/10. P2 met (Δ=0.886, Cliff's δ +0.96); **P1 monotonicity NOT met**; `α̂ ∈ RS` 100% in both runs. See `RESULTS.md` §6. |
| Open caveat | Thin cells (`\|RS\|`=1,2 keep 5 and 3 of 10 runs) are **unstable across the re-run**; 10-run cells reproduced to 3 decimals. Nothing is concluded from the middle of the table. Exploratory H6 withdrawn — did not replicate. |
| Paper | Not started; skeleton at M10. |

## Architecture (one line each)
- `concepts.py` — `ConceptSpace(k, n_slots)`, the latent vocabulary.
- `tasks.py` — `Task`: total `label_table` over `[k]^n` + `support` where data lives. **The
  total/partial distinction is load-bearing** (see `RESEARCH.md`).
- `oracle/` — the RS set. `enumerate.py` = pruned DFS (shared maps); `asp.py` = clingo (per-slot,
  relational, margins). `base.py` makes `mode`/`closure` mandatory, un-defaulted arguments.
- `generators/algebraic.py` — modular-linear family with closed-form `|RS| = gcd(sum w, m)`.
- `data/mnist.py` — checksum-verified idx loader, two mirrors, no torchvision.
- `models/` — `losses.py` exact-marginalisation NeSy loss; `nesy.py` shared-encoder predictor;
  `TabularEncoder` realises a chosen `α` exactly, which is what makes the oracle↔loss test possible.
- `metrics.py` — `Acc(C)`, `F1(C)`, collapse, `α̂` recovery, `rs_membership`.
- `config.py` / `runner.py` — frozen dataclasses, config-hash run identity, append-only JSONL,
  resume-by-hash, full provenance. The confirmatory-seed leakage guard raises rather than warns.

## Key commands
```bash
.venv/bin/python -m pytest tests/ -q          # full suite (oracle gates included)
.venv/bin/python scripts/download_mnist.py    # fetch + verify data
```

## Key decisions
See `DECISIONS.md`. Most consequential: pivot from the original continual-learning framing to
reasoning shortcuts (the original hypothesis is already published as COOL, ICML 2023); `f` is
**total** with `support` restricting only the data distribution; configs are dataclasses+YAML, not Hydra.

## Constraints
CPU-only dev box (4 cores). No HuggingFace / arXiv / external GitHub / LLM API keys. MNIST + PyPI
reachable. **rsbench, NeSy-CL, BEARS cannot be cloned — all reimplemented.** Heavy runs happen on
the user's GPU box; this repo must stay runnable and verifiable on a laptop.

## Non-negotiables
No fabricated numbers. `RESULTS.md` carries only numbers traceable to a run manifest. Novelty is
labelled a *"potential research gap"* until E1 has run. Negative results are preserved, not buried.

## Next action
Preregister and run **E2**: the margin-vs-binary study (H2) over the generator families, with convergence rate as an additional readout to test the new H5.
