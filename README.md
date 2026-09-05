# NeuroGenesis

**Provable identifiability predicts symbol grounding — but it does not make grounding cheap.**

A neuro-symbolic predictor maps input `x` → latent concepts `c` → label `y` under fixed symbolic
knowledge `K`. A **reasoning shortcut** is when it gets `y` right by learning a *wrong* `c` that
still satisfies `K`. Label accuracy looks fine; interpretability and OOD behaviour are gone.

This repository is a preregistered study of what you can do about that, and it reaches a negative
answer to a question the field has posed.

## What was found

**The shortcut count predicts grounding.** `|RS|`, computed offline before any training, predicts
whether a model grounds its concepts — across 152 heterogeneous tasks, after controlling for label
informativeness, `|Y|`, `k` and support size (partial Spearman **−0.775**).

**The oracle predicts *which* shortcut you get.** Among models that actually fit the label,
**97.1%** adopt a relabelling the oracle named in advance.

**Selecting tasks to remove shortcuts works — and still loses.** Greedy shortcut-cover selection
reaches provable identifiability at the smallest possible budget, matches the exhaustive optimum,
beats mutual-information selection (which is actively misled) and beats random by +0.625. Then plain
concept supervision grounds the model more cheaply anyway. **The JAIR 2026 survey asks for
cost-efficient mitigation; on this instance class the answer is no.**

**Two proposed refinements failed**, and are reported because they were the intended contributions:
a graded *margin* adds essentially nothing over binary identifiability (ΔR² ≈ 0.0015), and the
apparent identifiability/optimisability trade-off does not generalise beyond one task family.

Full ledger in [`RESULTS.md`](RESULTS.md); five of seven hypotheses failed.

## Why you might trust it

- **The oracle has three independent implementations** — pruned DFS, ASP via clingo, and a
  deliberately naive enumerator — agreeing exactly on 2000+ cross-checks.
- **The oracle is bound to the objective.** A test asserts `α` is a shortcut **iff** the encoder
  realising it achieves zero loss, exhaustively over every map for small `k`.
- **Predictions were committed before the data existed.** Each preregistration is a commit with no
  results in it; the ordering is checkable in `git log`.
- **Raw run records ship with the code**, so every table re-derives in seconds without retraining.
- **Bugs found are documented, not buried** — including a seeding bug that shipped twice and
  invalidated a headline experiment, which was then re-run.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pip install "torch>=2.9,<2.15"     # see REPRODUCIBILITY.md for the CPU-only note
.venv/bin/python scripts/download_mnist.py
.venv/bin/python -m pytest tests/ -q          # includes the oracle correctness gates
bash scripts/reproduce_all.sh                 # every table + figure, no retraining
```

```python
from neurogenesis.generators.algebraic import addition_task, modular_task
from neurogenesis.oracle import enumerate as en

args = dict(mode="shared", closure="total", allow_noninjective=True)

en.rs_set(addition_task(k=10), **args).count       # 1  -- identifiable, out of 10^10 candidates
en.rs_set(modular_task([1, 9], 10), **args).count  # 10 -- every cyclic shift is a shortcut
```

No GPU is required for any result in this repository.

## Documents

| file | contents |
|---|---|
| [`RESULTS.md`](RESULTS.md) | every number, traceable to a run manifest |
| [`RESEARCH.md`](RESEARCH.md) | formal definitions, the closed-form derivation, methodology |
| [`HYPOTHESES.md`](HYPOTHESES.md) | H1–H7 and what happened to each |
| [`DECISIONS.md`](DECISIONS.md) | research decision records, including the pivot and why |
| [`LIMITATIONS.md`](LIMITATIONS.md) | scope, threats to validity, the load-bearing assumption |
| [`ROADMAP.md`](ROADMAP.md) | what is next, and what was abandoned |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | environment, seeds, commands |
| `paper/` | preregistrations and the paper draft |

## Licence

MIT.
