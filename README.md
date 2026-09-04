# NeuroGenesis

**Does provable identifiability predict empirical symbol grounding in neuro-symbolic learning?**

A neuro-symbolic predictor maps input `x` → latent concepts `c` → label `y` under fixed symbolic
knowledge `K`. A **reasoning shortcut** (RS) is when it gets `y` right by learning a *wrong* `c`
that still satisfies `K` — accuracy looks fine, interpretability and OOD generalisation are gone.

Recent work settled the *analysis* question: given a constraint set, one can decide whether it
uniquely determines the concept mapping. This project asks the *design* question that follows:

> If identifiability is decidable **before training**, does it actually predict what gradient
> descent finds? And can we *select* a cheap set of tasks that provably collapses the RS space,
> instead of paying for per-concept supervision?

Both outcomes are informative. If identifiability predicts grounding, we get a cheap, theoretically
grounded mitigation. If it does not, that is a substantive negative result: symmetry is not what
determines grounding — optimisation is.

## Status

**Early.** The reasoning-shortcut oracle is built and validated; **no training experiments have
been run yet**, and no empirical claims are made. See `ROADMAP.md`.

## What works today

```python
from neurogenesis.generators.algebraic import addition_task, modular_task
from neurogenesis.oracle import enumerate as en

args = dict(mode="shared", closure="total", allow_noninjective=True)

# MNIST-Addition is identifiable: only the identity survives, out of 10^10 candidates.
en.rs_set(addition_task(k=10), **args).count          # -> 1

# y = (c1 + 9*c2) mod 10 admits all ten cyclic shifts.
en.rs_set(modular_task([1, 9], 10), **args).count     # -> 10
```

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pip install "torch>=2.9,<2.15"      # see REPRODUCIBILITY.md for the CPU-only note
.venv/bin/python scripts/download_mnist.py
.venv/bin/python -m pytest tests/ -q
```

## Documents

| File | Contents |
|---|---|
| `RESEARCH.md` | Formal definitions, the closed-form RS derivation, evaluation methodology |
| `LITERATURE.md` | Structured matrix of prior work and where this sits |
| `HYPOTHESES.md` | H1–H4 and the nulls, with predicted directions |
| `DECISIONS.md` | Research decision records |
| `EXPERIMENTS.md` | E1–E4 protocols |
| `RESULTS.md` | Only numbers traceable to a run manifest |
| `LIMITATIONS.md` | Scope, threats to validity |
| `ROADMAP.md` | NOW / NEXT / LATER / NOT YET / ABANDONED |
| `REPRODUCIBILITY.md` | Environment, seeds, commands |

## License

MIT.
