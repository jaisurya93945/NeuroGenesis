# Preregistration — E1

**Committed before any E1 run was executed.** The commit that adds this file contains no E1
results; the commit that adds E1 results comes later. That ordering is checkable in `git log` and
is the point of writing it down.

Date: 2026-09-04. Code state: M4 complete (oracle triple-validated, oracle↔loss binding proven,
training stack reproducing published MNIST-Addition numbers).

---

## 1. Question

Does **provable identifiability** of a neuro-symbolic task — a property of the symbolic knowledge
and support, decidable offline before any training — predict whether gradient descent actually
grounds the concepts?

## 2. Design

Task family: `y = (c₁ + w·c₂) mod 10`, MNIST digits as perception (Tier M).

| Condition | `w` | `\|RS\|` (oracle-verified) |
|---|---|---|
| A | 2 | 1 (identifiable) |
| B | 1 | 2 |
| C | 4 | 5 |
| D | 9 | 10 |

10 seeds per condition (`init_seed = data_seed = 0..9`), 40 runs total. Frozen recipe: SmallCNN,
Adam `lr=1e-3` cosine→`1e-4`, batch 128, 15 epochs, 25k train / 5k val / 5k test tuples.

**Why this family.** Across all four conditions the support (all 100 digit pairs), `|Y| = 10`, the
label marginal, `I(C;Y) = log 10`, perceptual difficulty and architecture are *identical*. Only
`|RS|` differs. The obvious confound — that `|RS|` merely tracks label informativeness — is
eliminated by construction, not by statistical adjustment. `Task.label_entropy()` being identical
across the family is asserted in `tests/test_tasks.py`.

Every shortcut in this family is a cyclic shift `α(x) = x + u mod 10`, so a model landing on a
shortcut scores `Acc(C)` **exactly 0**. The outcome is bimodal, not noisy.

## 3. Predictions (fixed now)

**P1 (primary, H1).** Mean test `Acc(C)` is monotone non-increasing in `|RS|` across A→B→C→D.

**P2 (primary, H1).** `mean Acc(C)[A] − mean Acc(C)[D] > 0.5`.

**P3.** Condition A reaches `Acc(C) > 0.95` in ≥ 9 of 10 seeds (it is identifiable, and M4 already
showed identifiable MNIST-Addition reaches ≈ 0.99 — so a failure here indicts the setup, not H1).

**P4.** `Acc(Y)` is high and roughly equal across **all** conditions. Shortcuts do not hurt label
accuracy — that is the entire premise of the field — so if `Acc(Y)` differs materially between
conditions, the conditions differ in task difficulty and the comparison is confounded.

**P5 (`rs_membership`).** For runs that fail to ground (`Acc(C) < 0.5`), the recovered `α̂` lies
inside the oracle-predicted `RS(T)` in ≥ 80% of cases.

## 4. Analysis plan (fixed now)

- **Primary readout:** mean test `Acc(C)` per condition, with bootstrap 95% CI (10,000 resamples)
  and **all 10 per-seed points overplotted**.
- **Effect size:** mean difference and Cliff's delta for A vs D.
- **Ordering test:** exact permutation test on condition labels for the monotone trend. One test.
- **No significance stars.** At 4 × 10 runs `p < 0.001` is trivially purchasable; the effect size
  and the raw scatter are the evidence.
- **Convergence gate:** a run enters the `Acc(C)` analysis only if its test `Acc(Y)` is ≥ 0.95 ×
  the best `Acc(Y)` achieved in its own condition. Exclusions are reported as a rate per condition,
  never silently dropped.
- **Secondary (exploratory, labelled as such):** `α̂` distribution, `F1(C)`, collapse, and whether
  non-identity `α̂` concentrate on particular shifts.

## 5. What would falsify H1 — and why each outcome is publishable

| Observation | Reading |
|---|---|
| P1 and P2 hold | Identifiability predicts grounding. Motivates the design/selection programme (E3). |
| **A grounds poorly too** | Identifiability is **not sufficient**. Optimisation, not symmetry, dominates. |
| **D grounds well** | Symmetry *permits* shortcuts but SGD *selects* the identity. The programme then needs a selection story, not a symmetry story (cf. arXiv 2608.10420). |
| Non-monotone in the middle (B, C) | The binary property is too coarse; supports H2, that the graded **margin** is the right predictor. |
| P5 fails — `α̂ ∉ RS` | Deterministic-RS theory under-specifies what SGD finds. A finding in its own right, invisible to `Acc(C)`. |

**Commitment.** No prediction above will be revised after seeing results. If the outcome is
negative it is reported at the same prominence as a positive one, in `RESULTS.md`. Any post-hoc
analysis is labelled **exploratory** and gets its own preregistration and its own runs before being
treated as confirmatory.

## 6. Known limits of E1 specifically

Single task family, single architecture, `k = 10`, CPU-scale. E1 cannot establish generality — that
is E2's job. A null result here means "not detected in this family at this scale", never
"does not exist".
