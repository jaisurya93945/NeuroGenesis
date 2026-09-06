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
| Phase | **E1–E5 complete (~2,600 runs); paper drafted in full.** Six of eight hypotheses failed; H4 falsified twice. **E6 (Tier-M replication) is IN FLIGHT** — preregistered data-free in `529146e`. |
| Experiments run | **E1, run twice** (before/after a seeding fix). Post-fix `Acc(C)` = 0.986 / 0.988 / 0.297 / 0.100 for `\|RS\|` = 1/2/5/10. P2 met (Δ=0.886, Cliff's δ +0.96); **P1 monotonicity NOT met**; `α̂ ∈ RS` 100% in both runs. See `RESULTS.md` §6. |
| Open caveat | Thin cells (`\|RS\|`=1,2 keep 5 and 3 of 10 runs) are **unstable across the re-run**; 10-run cells reproduced to 3 decimals. Nothing is concluded from the middle of the table. Exploratory H6 withdrawn — did not replicate. |
| Paper | All sections written. `results.md` and `discussion.md` landed in `3973061`; every number regenerates byte-identically from `results/runs/*.jsonl` via `scripts/reproduce_all.sh`. |
| ⚠️ In flight | `results/runs/e6.jsonl` is **partial (264 runs expected)** and not analysable. The E2-subset arm (P3) has not started. |
| Literature | Re-sweep retried 2026-09-06 with web tooling: search works, **every primary source is egress-blocked** (arxiv, JAIR, ACM, OpenReview, CEUR, neurips.cc). Still `[S]`, still blocking for novelty language. Two summary findings now recorded because they cut against us — see `LITERATURE.md`. |

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
Wait for E6 to finish (264 runs), then `scripts/analyze_e6_tierm.py`, then run the E2-subset arm
(`experiments/e2_margin_vs_binary.py --tier M --seeds 3 --store results/runs/e6_e2subset.jsonl`) for
P3. Write E6 into `RESULTS.md` and the paper **whichever way it comes out** — a P4 failure means
greedy wins on cost under real perception and H4 is partially revived, and
`preregistration_e6.md` §6 commits to reporting that as a headline. The literature re-sweep stays
**blocking** for any novelty language — every primary source is egress-blocked here (2026-09-06).
