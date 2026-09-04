# Preregistration — E2

**Committed before any E2 run is executed.** As with E1, the commit adding this file contains no
E2 results; the commit adding results comes later, and the ordering is checkable in `git log`.

Code state: M6–M8 infrastructure complete (margin oracle validated against brute force; planted,
support-thinning, rarefying and random generators; selection module). Seeding bug fixed and pinned
by `tests/test_determinism.py`.

---

## 1. Question

E1 asked whether *binary* identifiability predicts grounding. E2 asks whether the **margin** — the
graded quantity — predicts it *better*, and whether either survives controlling for the obvious
confounds across a wide, heterogeneous set of tasks rather than one hand-picked family.

```
margin(T) = min over alpha != id of  Pr_{c ~ D} [ f(alpha(c)) != f(c) ]
```

`margin > 0` iff `T` is identifiable, so the margin strictly subsumes the binary property. The
question is whether the extra resolution buys predictive power.

## 2. Why this could fail, and why that is the interesting case

`rarefy_against` constructs tasks that are **provably identifiable with arbitrarily small margin**:
full support, unchanged data volume, `|RS| = 1`, but the evidence refuting the cheapest wrong
relabelling carries as little as 0.0005 of the probability mass. Binary identifiability says these
should ground perfectly. If they do, H2 is wrong and binary identifiability is the whole story. If
they do not, the binary property is insufficient and the margin is the operative quantity.

That is the decisive cell, and it is why the rarefied family is included by construction rather
than left to chance.

## 3. Design

**Tasks: 166 exactly** (39 algebraic, 42 rarefied, 36 planted, 31 addition/support-thinned, 18 random). Five generator families, each contributing a spread of `|RS|` and margin:

| Family | What it varies | Why included |
|---|---|---|
| `algebraic` | `(k, n, w, m)` — closed-form `\|RS\|` | Known answers; exact control of entropy |
| `planted` | A chosen symmetry monoid, incl. non-injective collapses | Shortcut types the modular family never produces |
| `support` | Support density 0.15–1.0 | Identifiability continuum; decouples constraint strength from data volume |
| `rarefied` | Margin at fixed `\|RS\| = 1` | **The decisive cell** — separates margin from the binary property |
| `random` | Nothing (null family) | Guards against conclusions that hold only on designed tasks |

**Perception.** Tier S (synthetic vectors, `dim=32`). 5 seeds per task, 166 tasks, **830 runs**.

**Calibration outcome (dev tasks only, recorded before any confirmatory run).** Frozen recipe:
`noise=0.1`, `epochs=30`, `n_train=8000`, MLP encoder, ~2.8 s/run. Verified on dev tasks that
identifiable tasks *do* ground under this recipe:

| dev task | `\|RS\|` | `Acc(Y)` | `Acc(C)` |
|---|---|---|---|
| addition k=10 | 1 | 1.000 | **1.000** |
| addition k=6 | 1 | 1.000 | **1.000** |
| random k=6, full support | 1 | 1.000 | **1.000** |
| planted cyclic k=6 | 3 | 1.000 | 1.000 |
| mod-10 `w=9` | 10 | 1.000 | 0.000 |
| **mod-10 `w=2`** | **1** | **0.490** | **0.000** |

**A known, declared limitation of the substrate.** The modular family at `k=10` does *not* converge
on Tier S — `Acc(Y)` peaks around 0.5 regardless of epochs (10→60), training-set size (8k→25k) or
noise (0.1→0.5). Modular arithmetic mod 10 has no ordinal structure for the encoder to exploit, and
it is the *identifiable* members that fail hardest, consistent with H5. Every other family grounds
correctly, so Tier S remains a valid substrate.

Consequence, stated in advance rather than discovered later: modular `k=10` runs will mostly be
removed by the convergence gate. Their exclusion rate is reported per family, and **no conclusion is
drawn from the modular arm on Tier S**. The modular family's role in this project is Tier M (E1),
where it does converge.

**Tier transfer check.** 20 of the 166 tasks are additionally run at Tier M (MNIST digits), 3 seeds
each, as a separate follow-up sweep. If Tier S and Tier M disagree in rank order on those 20, the
Tier S conclusions are demoted and reported as tier-specific rather than general.

## 4. Predictions (fixed now)

**P1 (H2, primary).** `margin(T)` predicts mean `Acc(C)` better than binary identifiability and
better than `log|RS|`. Tested by nested-model `ΔR²` with a bootstrap CI on the difference.
Direction: `ΔR²(margin over binary) > 0`, CI excluding 0.

**P2 (H2, the decisive cell).** Among **provably identifiable** tasks (`|RS| = 1`), `Acc(C)`
increases with margin. Specifically, low-margin rarefied tasks (`margin < 0.01`) ground materially
worse than high-margin identifiable tasks (`margin > 0.1`): predicted difference > 0.3.
*If this fails, H2 is falsified and the binary property is sufficient.*

**P3 (H3).** The `log|RS|` ↔ `Acc(C)` association survives partial correlation controlling for
`I(C;Y)`, `H(Y)`, `|Y|`, `k`, `n`, noise, support size and realised-vs-declared support gap.
Reported as partial Spearman with bootstrap CI **and** the residual scatter, not a coefficient alone.

**P4 (H5, the new hypothesis from E1's diagnostic).** Convergence rate — the fraction of runs
passing the gate — **increases** with `|RS|`. E1 saw 5/10, 4/10, 10/10, 10/10; E2 tests it across
166 tasks with `|RS|` spanning a wide range. Direction: positive Spearman between `log|RS|` and
per-task convergence rate.

**P5 (`rs_membership`).** Among converged non-grounding runs, `α̂ ∈ RS(T)` in ≥ 80% of cases,
replicating E1's 100% on a much more heterogeneous task set. A materially lower rate here would
mean the deterministic RS theory under-specifies what SGD finds on general tasks — publishable
in its own right.

## 5. Analysis plan (fixed now)

- **Unit of analysis: the task** (n = 166 before exclusions), response = mean `Acc(C)` over its seeds. Runs are not
  the unit; that would treat seeds as independent evidence about the `|RS|` relationship.
- **Convergence gate** as in E1: a run enters the `Acc(C)` analysis only if its `Acc(Y)` ≥ 0.95 ×
  the best `Acc(Y)` among that task's own seeds. Exclusion rates reported per family. Note the gate
  interacts with P4 by construction, so P4 is computed on *ungated* runs and says so.
- **Estimation, not significance.** Bootstrap 95% CIs throughout; effect sizes; scatter plots with
  every task shown. No significance stars.
- **Multiplicity.** Five confirmatory tests (P1–P5), Holm-corrected. Anything else is exploratory
  and labelled.
- **Degenerate tasks excluded, declared in advance:** tasks with `|Y| = 1` (a single label carries
  no signal at all) and tasks whose oracle result is `truncated`. Both are counted and reported.

## 6. Leakage control

Task-generator seeds `0–99` are dev — noise level and any recipe choice may only be tuned there.
Confirmatory tasks use seeds `≥ 1000` and are run once with the frozen recipe. `runner.run()`
raises if a confirmatory seed is used with `tuning_mode=True`.

## 7. Commitments

No prediction here is revised after seeing results. A negative outcome on P1 or P2 — that the
margin adds nothing over binary identifiability — is reported at the same prominence as a positive
one, and would mean the project's proposed refinement does not survive contact with evidence.
