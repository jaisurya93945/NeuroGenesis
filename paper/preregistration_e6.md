# Preregistration — E6 (Tier-M replication)

**Committed before any confirmatory E6 run.** As with E1–E5, this commit contains no E6 results.
The `--tier` machinery in `experiments/e3_selection.py` ships in the same commit; it contains no data.

---

## 1. Why E6 exists

Everything after E1 is **synthetic perception**: concepts are rendered as vectors from a random
codebook plus Gaussian noise. That was a deliberate cost decision — it is what made ~2,600 runs
honest rather than aspirational on four CPU cores — but it means the paper's central negative
(`RESULTS.md` §8–9: shortcut-cover selection matches the exhaustive optimum and still loses on cost
to concept supervision) has only ever been measured where perception is a knob.

`preregistration_e2.md` §3 already committed to the consequence: **if the Tier-M rank order
disagrees with Tier S, the Tier-S conclusions are demoted to tier-specific.** E6 is that check.

## 2. What is already known, stated so it cannot be mistaken for a result

A dev screen ran on **dev seeds 900–903 only**, disjoint from the confirmatory range 0–7. Three
things it established, none of which is an E6 finding:

**(a) The phenomenon reproduces on MNIST.** Base-only reaches `Acc(Y)` ≈ 0.987 with `Acc(C)` ≈ 0.001
— a clean shortcut — and greedy at budget 2 grounds (`Acc(C)` 0.994–0.995). So E6 is not testing
whether anything happens at all; it is testing whether the *ordering and the cost verdict* survive.

**(b) The absolute `Acc(Y) ≥ 0.99` gate cannot be carried over, and the gate is therefore changed
here — before the confirmatory runs, with the reason recorded.** Across 21 dev runs `Acc(Y)` spans
0.983–0.991, maximum **0.9913**. Doubling epochs (30 → 60) did not raise it (base 0.9880 → 0.9867;
greedy@2 0.9880 → 0.9907). This is the **MNIST perceptual ceiling** — per-digit accuracy ≈99.4% over
two slots caps tuple label accuracy near 0.988 — not a convergence threshold. A 0.99 gate would
exclude most runs for a *perceptual* reason, which is the differential-exclusion trap E1 hit
(`RESULTS.md` §6.5). E1, the only prior Tier-M experiment, used a relative gate for this reason.

> **Tier-M gate: absolute `Acc(Y) ≥ 0.97`**, with the per-cell exclusion rate reported, and a
> **sensitivity check at 0.95 and 0.98 fixed now**. A genuinely non-converged run sits far below
> (E1's excluded runs were 0.4–0.7), so 0.97 separates convergence from the ceiling cleanly.
> This is still an **absolute** gate. The E3 lesson was about *relative* gates manufacturing a
> spurious theory gap; that is not being reintroduced.

**(c) Concept supervision grounds below the Tier-S budget grid, so the grid extends downward.** On
dev seeds it grounds at 25 labels, at 10 labels (2/2 seeds, 0.992–0.994), and at 5 labels in 1 of 2
seeds. `L` is therefore left-censored on Tier S's `{25, 50, 100, 200, 400}`.

> **Tier-M concept budgets: `{2, 5, 10, 15, 25, 50}`**, fixed now. Note the direction: extending the
> grid downward makes the cost comparison **harsher for the method being tested**, not kinder.

## 3. Design

- **Instance:** E3's primary instance only — `y = (c₁ + 5·c₂) mod 6`, `|RS| = 6`, rendered as MNIST
  digits 0–5. Task, auxiliary pool, oracle and selection methods are **identical** to Tier S; the
  only thing that changes is perception. That is what makes this a replication rather than a new
  experiment.
- **Arms:** every E3 method — `base_only`, `greedy_rs`, `information_greedy`, `exhaustive_optimal`,
  `random` (5 draws), `all_tasks`, `distractors_only`, `concept_supervision`.
- **Recipe, frozen by the dev screen:** `cnn` encoder (`SmallCNN`), 8000 train tuples, 1500 test,
  30 epochs, batch 128, Adam 1e-3 → 1e-4. Encoder seeded **inside** `build_encoder(seed=...)`.
- **Seeds:** 8 confirmatory seeds, 0–7. Dev seeds ≥900 are never pooled with them.
- **Second arm:** a `k ≤ 10` subset of E2's task set × 3 seeds, to check that `|RS|` still predicts
  `Acc(C)` under realistic perception.

## 4. Predictions (fixed now)

**P1 — the method still works.** `greedy_rs` reaches mean `Acc(C) > 0.9` at task budget 2, and
matches `exhaustive_optimal` on final `|RS|` at every budget.

**P2 — the ordering among selection methods is preserved.** `greedy_rs` beats `random` at matched
budget 2 (difference > 0, CI excluding zero) and beats `information_greedy` at every budget.

**P3 — `|RS|` still predicts grounding.** On the E2 subset, the partial Spearman between `log|RS|`
and `Acc(C)` is negative with a CI excluding zero, controlling for label entropy and `|Y|`.

**P4 — the cost verdict is preserved.** Letting `L_M` be the cheapest concept budget on
`{2, 5, 10, 15, 25, 50}` reaching mean `Acc(C) > 0.9`, and `R_M = 2` the greedy task budget that
grounds: `L_M ≤ 2τ` for every preregistered `τ ∈ {25, 50, 100, 200, 400}` — i.e. concept supervision
still wins or ties everywhere except possibly the smallest `τ`.

*P4 is the one that matters. The dev screen makes P1 near-certain and says nothing about the
confirmatory ordering or the cost margin.*

## 5. Analysis plan

- Unit: the `(method, budget, seed)` run; response test `Acc(C)`. Bootstrap 95% CIs, per-seed
  scatter, effect sizes. No significance stars.
- Four confirmatory tests (P1–P4), Holm-corrected.
- Exclusions declared now: runs failing the Tier-M gate, reported as a rate per cell, plus the
  0.95/0.98 sensitivity check.
- `rs_membership` computed only when the oracle result is **not truncated** (the trap from
  `RESULTS.md` §7.5). At `k = 6` no truncation is expected; the check is asserted anyway.
- `L_M` is left-censored if it grounds at the smallest budget (2) and reported as `≤ 2`, never
  extrapolated below the grid.

## 6. What each outcome means

| outcome | reading |
|---|---|
| P1–P4 all hold | The Tier-S conclusions **replicate under realistic perception**. The negative result stands as stated, and the synthetic-perception limitation shrinks to the E2 breadth claim. |
| **P4 fails** (greedy wins on cost at Tier M) | **H4 is partially revived at the realistic-perception tier.** This is a positive result for the method and will be reported at full prominence — headline, not footnote — alongside the two Tier-S falsifications, with the tier difference stated as the finding. |
| P1 or P2 fails | The *method* does not survive the tier change. The Tier-S design conclusions are demoted to tier-specific per `preregistration_e2.md` §3. |
| P3 fails | The paper's **positive** analysis finding is tier-specific. That is the most damaging possible outcome for the paper and will be reported as such. |

## 7. Commitment

No prediction is revised after seeing results. The gate and the budget grid are changed **here,
before the runs, with the dev-screen evidence recorded above** — not after seeing confirmatory data.
A P4 failure revives a hypothesis this project has twice declared dead, and it will be reported
exactly as loudly as the falsifications were.
