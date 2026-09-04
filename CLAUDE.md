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
| Phase | M2 complete (oracle triple-validated). M3 next (loss + oracle↔loss binding). |
| Experiments run | **None yet.** No training results exist. |
| Paper | Not started; skeleton at M10. |

## Architecture (one line each)
- `concepts.py` — `ConceptSpace(k, n_slots)`, the latent vocabulary.
- `tasks.py` — `Task`: total `label_table` over `[k]^n` + `support` where data lives. **The
  total/partial distinction is load-bearing** (see `RESEARCH.md`).
- `oracle/` — the RS set. `enumerate.py` = pruned DFS (shared maps); `asp.py` = clingo (per-slot,
  relational, margins). `base.py` makes `mode`/`closure` mandatory, un-defaulted arguments.
- `generators/algebraic.py` — modular-linear family with closed-form `|RS| = gcd(sum w, m)`.
- `data/mnist.py` — checksum-verified idx loader, two mirrors, no torchvision.

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
M3 — exact-marginalisation NeSy loss + `test_predicted_rs_achieves_zero_loss`, which binds the oracle's combinatorial claim to the objective the trainer actually optimises.
