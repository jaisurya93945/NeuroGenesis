# Architecture

```
                 symbolic side                          neural side
        ┌──────────────────────────┐          ┌──────────────────────────┐
        │  Task                    │          │  NeSyModel               │
        │   label_table  f:[k]ⁿ→Y  │          │   shared encoder φ_θ     │
        │   support      supp⊆[k]ⁿ │          │   x_j ↦ p(c_j | x_j)     │
        └────────────┬─────────────┘          └────────────┬─────────────┘
                     │                                     │
        ┌────────────▼─────────────┐          ┌────────────▼─────────────┐
        │  RS oracle               │          │  exact marginalisation   │
        │   enumerate.py  (DFS)    │          │   −log Σ_{f(c)=y} Πp(c_j)│
        │   asp.py        (clingo) │          └────────────┬─────────────┘
        │   → RS(T), |RS|, margin  │                       │
        └────────────┬─────────────┘                       │
                     │                                     │
                     └──────────────┬──────────────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │  metrics.py            │
                        │   Acc(C), F1(C), Cls(C)│
                        │   α̂  →  is α̂ ∈ RS(T)?  │   ← the bridge the project tests
                        └────────────────────────┘
```

## The one thing to understand

`RS(T)` is what the *data and knowledge* permit; `α̂` is what *optimisation* actually chose.
Everything in this repository exists to measure the gap between those two.

`tests/test_oracle_vs_loss.py` is what makes the comparison meaningful: it proves that
"`α ∈ RS(T)`" (a combinatorial statement the oracle computes) and "this encoder achieves zero
loss" (a numerical statement about the objective) are the *same* statement. Without it, the two
sides of the diagram would be measuring different things and every downstream number would be
uninterpretable.

## Module responsibilities

| Module | Owns |
|---|---|
| `concepts.py`, `tasks.py` | The formal objects. `Task` keeps `f` total and `support` separate — the definitional choice everything inherits |
| `oracle/enumerate.py` | Pruned DFS over `[k]^[k]`; fast for shared maps |
| `oracle/asp.py` | clingo encoding; per-slot maps, margins, relational `K` |
| `generators/algebraic.py` | Modular-linear family, closed-form `\|RS\| = gcd(Σw, m)` |
| `models/losses.py` | Exact marginalised NLL — no sampling, no relaxation |
| `models/nesy.py` | Shared-encoder predictor; `TabularEncoder` realises a chosen `α` |
| `metrics.py` | Concept quality, `α̂` recovery, `rs_membership` |
| `config.py` | Run identity (config hash), provenance, leakage guard |
| `runner.py` | Execute, record to append-only JSONL, resume by hash |
| `stats.py` | Bootstrap intervals, Cliff's delta, permutation trend test |

## Invariants the test suite enforces

1. Three independent oracle implementations agree (DFS, ASP, naive reference).
2. `α ∈ RS(T)` **iff** the corresponding encoder achieves zero loss and `Acc(Y) = 1`.
3. `RS` is a monoid; `RS(T₁∧T₂) = RS(T₁) ∩ RS(T₂)`; more support never grows `RS`.
4. A confirmatory task seed used in tuning mode raises, rather than warns.
5. No MNIST image appears in two splits.
