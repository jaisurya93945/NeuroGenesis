# Preregistration — E5

**Committed before any E5 run.** As with E1–E3, this commit contains no E5 results.

---

## 1. Why E5 exists

E3 falsified H4: greedy RS-cover grounds at 2 authored rules, concept supervision grounds at 50
labels, so selection loses at every plausible `τ`. But that verdict rests on **one unmeasured
number** — the relative cost of authoring a rule versus labelling a concept — measured at a single
concept-vocabulary size, `k = 6`.

The asymmetry worth testing: **label cost must grow with `k`** (you cannot ground a vocabulary you
have never seen), while **rule cost need not**. If so, E3's verdict is an artefact of small `k`, and
the honest claim becomes conditional rather than flat.

## 2. What is already known, stated so it cannot be mistaken for a result

Two things were established **before** this preregistration and are *not* claims of E5:

**(a) The rule side is a deterministic combinatorial fact, not an experiment.** `c₀ mod q` is
invariant under the shift `u` exactly when `q | u`, so a set of divisor rules identifies the concept
map iff their moduli have lcm `k`. Computed with the oracle (no training):

| `k` | 6 | 8 | 10 | 12 | 15 | 16 | 20 | 24 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| rules to reach `\|RS\|=1` | 2 | **never** | 2 | 2 | 2 | **never** | 2 | 2 | 2 |

**Flat at 2 for every `k` with ≥2 distinct prime factors; impossible for prime powers**, where the
only identifying divisor rule is `mod k` itself, which is full concept supervision in disguise.

**(b) A dev screen (1 seed, generator seed 0) confirmed training grounds with those 2 rules** at
every composite `k ∈ {6,10,12,15,20,24,30}`: `Acc(Y) = 1.000`, `Acc(C) = 1.000`.

**The genuinely unknown quantity, and the only thing E5 measures, is `L(k)`: the minimum number of
concept labels needed to ground at each `k`.**

## 3. Design

- **Base:** `y = (c₁ + (k−1)·c₂) mod k`, giving `|RS| = k` (the maximal cyclic shortcut group).
- **`k` sweep:** composite `{6, 10, 12, 15, 20, 24, 30}` plus prime powers `{8, 16}` as the declared
  negative case.
- **Frozen recipe** (from the dev screen, dev seeds only): Tier S, `dim = 64`, `noise = 0.1`, MLP,
  40 epochs, 15000 train tuples, 8 seeds per cell.
- **Label arm:** concept-label budgets `{10, 20, 40, 80, 160, 320, 640, 1280}` — fixed now, so the
  grid cannot be chosen after seeing where the curve bends.
- **Rule arm:** greedy budgets `{0, 1, 2, 3}`.
- **Absolute convergence gate `Acc(Y) ≥ 0.99`**, per the E3 lesson.
- **Grounded** means mean `Acc(C) > 0.9` over the cell's seeds. `L(k)` and `R(k)` are the smallest
  budgets on the fixed grids meeting that bar.

## 4. Predictions (fixed now)

**P1 — labels scale with vocabulary.** `L(k)` is strictly increasing in `k` over the composite
sweep (Spearman `> 0.8`), and `L(30) ≥ 4 × L(6)`.

**P2 — rules do not.** `R(k) = 2` for every composite `k` in the sweep. (Combinatorially forced;
what is being tested is that *training* realises it at every `k`, which is not forced.)

**P3 — the crossover exists.** For a rule cost of `τ = 100` concept-labels-equivalent, there is some
`k ≤ 30` in the sweep with `L(k) > 2τ = 200`, i.e. where selection is strictly cheaper. Reported as
a curve over `τ ∈ {25, 50, 100, 200, 400}` with the crossover `k` for each.
*This is the prediction that would make H4 conditionally true rather than dead.*

**P4 — the declared limit.** For prime powers `k ∈ {8, 16}`, no rule budget reaches `Acc(C) > 0.9`,
while concept supervision still does. Selection has a structural blind spot, and it is reported with
the same prominence as any positive finding.

## 5. Analysis plan

- Unit: the `(k, arm, budget)` cell; response mean `Acc(C)` over 8 seeds.
- Bootstrap 95% CIs, per-seed scatter, effect sizes. No significance stars.
- Four confirmatory tests (P1–P4), Holm-corrected.
- Exclusions declared now: cells failing the absolute gate, reported as a rate per `k`.
- `L(k)` is right-censored if it exceeds 1280; reported as `> 1280`, never extrapolated.

## 6. What each outcome means

| outcome | reading |
|---|---|
| P1 + P3 hold | H4 is **conditionally revived**: selection wins for large concept spaces. The paper's claim becomes "cost-efficiency depends on `k`", which is a real, useful, and previously unstated result. |
| P1 holds, P3 fails within `k ≤ 30` | The trend is right but the crossover is out of reach at testable scale. Reported as suggestive, not established, with the extrapolation refused. |
| P1 fails (`L(k)` flat) | Labels do **not** scale with vocabulary, E3's verdict generalises, and **H4 is permanently dead**. This is the final nail and will be stated exactly that way. |
| P2 fails | Training does not realise the combinatorial promise at large `k` — the method is weaker than its own theory. |

## 7. Commitment

No prediction is revised after seeing results. A flat `L(k)` ends the design contribution
permanently, and `RESULTS.md` will say so with the same prominence a positive result would get.
