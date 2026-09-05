# Results

Every number below is produced by the `analyze_*.py` scripts from the committed run records in
`results/runs/`. Nothing is typed by hand. Approximately 2,600 training runs across five
experiments, each preregistered in a commit containing no data.

## E1 — Does the shortcut count predict grounding?

`y = (c₁ + w·c₂) mod 10` on MNIST digits, `w` chosen so `|RS| ∈ {1, 2, 5, 10}`. Across conditions the
support, `|Y|`, label marginal, `I(C;Y)` and perceptual difficulty are **identical**; only `|RS|`
varies. 40 runs.

**E1 was executed twice.** The first execution ran under a seeding bug that left `init_seed` without
control of weight initialisation, so those runs cannot be regenerated from their recorded seeds. The
bug was fixed and E1 re-run. **The table below is the post-fix run**; the pre-fix records are kept at
`results/runs/archived_e1_preseedfix.jsonl` and compared in `RESULTS.md` §6.3.

| `\|RS\|` | `w` | n (gated) | excluded | test `Acc(C)` [95% CI] | test `Acc(Y)` | grounded | `α̂ ∈ RS` |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 5 | 5 | **0.9858** [0.9842, 0.9869] | 0.9757 | 1.00 | 1.00 |
| 2 | 1 | 3 | 7 | 0.9881 [0.9865, 0.9900] | 0.9763 | 1.00 | 1.00 |
| 5 | 4 | 10 | 0 | 0.2965 [0.0015, 0.5917] | 0.9767 | 0.30 | 1.00 |
| 10 | 9 | 10 | 0 | **0.0998** [0.0011, 0.2963] | 0.9723 | 0.10 | 1.00 |

**P1 (monotone non-increasing) is NOT met.** The `|RS|=2` cell sits 0.002 *above* `|RS|=1`. Both are
≈0.986 and the `|RS|=2` cell retains 3 runs, so this is noise in a thin condition rather than a
reversal — but the prediction as written failed and is recorded as failed. Permutation trend
`p = 0.00010` (reported once).

**P2 is met.** The `|RS|=1` vs `|RS|=10` contrast is **0.8860** [0.6892, 0.9852], Cliff's delta
**+0.960**. `Acc(Y)` is flat (0.972–0.977) across all four conditions, confirming that shortcuts cost
nothing in label accuracy — they are invisible to the training signal.

The recovered shortcuts are exactly the predicted ones, in every gated run of every condition
(`α̂ ∈ RS` = 1.00 throughout; 57/57 across both executions). Theory says they are the shifts `u` with
`u(1+w) ≡ 0 (mod 10)`; observed shifts fall inside that set with nothing outside it — `{0}` for
`w=2`, `{0}` for `w=1` (the permitted `5` never appeared), all five of `{0,2,4,6,8}` for `w=4`, and
seven of the ten permitted shifts for `w=9`, the identity among them once, in the single seed that
grounded.

**Caveat, from the convergence diagnostic (`RESULTS.md` §6.5):** exclusion rates are uneven
(5, 7, 0, 0) and concentrated in the *low*-`|RS|` conditions. A 5-recipe, 50-run diagnostic on dev
tasks showed this survives every recipe tried (`|RS|=10` converged 25/25; `|RS|=1` never exceeded
3/5), so it is an effect of the manipulated variable, not a tuning artefact — identifiable modular
tasks are simply harder to optimise. It leaves the `|RS|`=1 and 2 means resting on 5 and 3 runs,
which is thin, and those two cells did **not** reproduce across the re-run while the 10-run cells
reproduced to three decimals. Nothing is concluded from the middle of the table; the headline
contrast does not depend on it.

## E2 — Does a graded margin predict better? No.

155 tasks from five generator families × 5 seeds = 830 runs, synthetic perception. After the declared
exclusions (2 degenerate `|Y|=1`, 1 truncated oracle), **152 analysable tasks**.

- **H2 falsified.** Adding `margin(T)` to a model that already knows binary identifiability yields
  `ΔR² = 0.0015` [0.0004, 0.0042]. The interval excludes zero, so the letter of P1 reads "met" — it
  should not be reported that way, and the preregistration committed to estimation over significance
  precisely so a detectable-but-tiny effect could not be dressed up as a finding. The substantive
  test, P2, predicted low-margin identifiable tasks would ground ≥0.3 worse; the observed difference
  is **0.037** [0.013, 0.060]. The `rarefied` family was built specifically to break the binary
  property — full support, `|RS| = 1`, margin driven down three orders of magnitude — and those tasks
  ground anyway (0.953, n=18, against 0.990, n=73, for high-margin identifiable tasks). **Binary
  identifiability is doing the work.** This was the project's proposed novel contribution.
- **H3 supported.** `log|RS|` ↔ `Acc(C)` raw Spearman **−0.801**; partial Spearman **−0.775**
  [−0.898, −0.516] after controlling for label entropy, `|Y|`, `k` and support size. The association
  is not an artefact of informativeness or task size, and it holds across planted symmetries, random
  label tables, support thinning and modular arithmetic alike.
- **H5 not supported outside one family.** The identifiability/optimisability trade-off seen in E1
  gives Spearman **+0.034** across heterogeneous tasks. The prediction said "positive" and +0.034 is
  positive, so P4's letter is met; it is indistinguishable from zero and is reported as not
  supported. H5 is demoted to a property of modular arithmetic at `k=10`, not of identifiability.

## E2b — The oracle predicts *which* wrong grounding is adopted

`α̂ ∈ RS(T)` held for 78.4% of converged non-grounding runs, below the 80% predicted. Triage
(`scripts/diagnose_h7.py`) shows this was a convergence artefact, not a gap in the theory:

| require `Acc(Y) ≥` | n | fraction outside `RS` |
|---|---|---|
| all gated runs | 210 | 19.0% |
| 0.99 | 188 | 9.6% |
| **0.999** | 175 | **2.9%** |

Out-of-set runs average `Acc(Y)` 0.908 against 1.000 for in-set runs. **Among models that actually
learned the task, 97.1% adopt a relabelling the oracle named in advance.** The deterministic theory
is *more* accurate than the preregistered test implied.

## E3 — Does selection beat the baselines on cost? No.

`y = (c₁ + 5·c₂) mod 6`, `|RS| = 6`, with an oracle-verified pool in which no single auxiliary task
suffices. 512 runs, absolute gate `Acc(Y) ≥ 0.99` (100% passed).

| method | budget | `\|RS\|` | `Acc(C)` [95% CI] | grounded |
|---|---|---|---|---|
| base only | 0 | 6 | 0.250 [0.000, 0.625] | 0.25 |
| **greedy RS-cover** | **2** | **1** | **1.000** [1.000, 1.000] | **1.00** |
| information-greedy | 3 | 2 | 0.250 [0.000, 0.625] | 0.25 |
| random (5 draws) | 3 | 1 | 0.675 [0.525, 0.825] | 0.68 |
| exhaustive optimum | 2 | 1 | 1.000 [1.000, 1.000] | 1.00 |
| **concept supervision** | **50 labels** | 6 | **1.000** [1.000, 1.000] | **1.00** |

The method works exactly as designed: it grounds at budget 2, **matches the exhaustive optimum** at
every budget, beats information-greedy (which never grounds — it spends its budget on
shift-invariant distractors carrying high label entropy and zero shortcut coverage) and beats random
by **+0.625** [0.475, 0.775].

**And it loses.** The cheapest concept supervision that grounds needs **50 labels**; greedy needs 2
authored rules. Over the preregistered `τ ∈ {25, 50, 100, 200, 400}`, greedy ties at `τ = 25` and
loses everywhere above. **H4 falsified.**

## E5 — Does the cost verdict flip at larger vocabularies? No — it worsens.

The last defence: labels must scale with vocabulary size while rules stay flat, so the comparison
flips at large `k`. 864 runs, exhaustive over all budgets (no bisection), gate passed by 100%.

| `k` | factors | `L(k)` labels | `R(k)` rules | base `Acc(C)` |
|---|---|---|---|---|
| 6 | 2·3 | **80** | 2 | 0.125 |
| 10 | 2·5 | 40 | 2 | 0.125 |
| 12 | 2²·3 | 40 | 2 | 0.000 |
| 15 | 3·5 | 40 | 2 | 0.000 |
| 20 | 2²·5 | 40 | 2 | 0.000 |
| 24 | 2³·3 | 20 | 2 | 0.000 |
| 30 | 2·3·5 | **10** | 2 | 0.000 |
| 8, 16 | 2³, 2⁴ | 40 | **never** | 0.125, 0.250 |

**The asymmetry exists and points the wrong way.** Spearman(`k`, `L`) = **−0.906** against a
predicted `> 0.8`; `L(30)/L(6) = 0.12` against a predicted `≥ 4`. Labels needed to ground *fall* 8×
as the vocabulary grows 5×, while rules stay flat at 2. Concept supervision becomes **more**
competitive as vocabularies grow.

The artifact check passed first: base `Acc(C)` with no supervision is **0.000–0.250 at every `k`**,
so nothing grounds spontaneously and the labels are doing the work.

**Structural boundary.** For prime powers the divisor-rule family contains no identifying rule at
all (`mod k` would be full supervision in disguise), so `R(k)` is infinite *within that family* — the
rule arm plateaus at `Acc(C)` 0.500 (`k=8`) and 0.625 (`k=16`) no matter the budget — while labels
still ground at 40. Preregistered as P4, reported separately, never mixed into the crossover
analysis. It is a blind spot of the rule family, not an implementation failure.

E5 measures **annotation quantity under this project's operational cost model**, not human
annotation time or effort; no human-annotation measurement was performed.

## E4 — Do models keep shortcuts their constraints now forbid? Essentially never.

120 sequential streams; 98 pass the final-phase gate.

- **RS lock-in occurs in 2 of 98 streams**, both inside a cell rendered unusable by exclusion
  (below). Models **escape** provably-forbidden shortcuts once the forbidding constraint arrives —
  preregistered in advance as a real finding either way.
- **Sequential arrival costs grounding:** joint 0.917 vs best sequential (replay) 0.673, difference
  **0.243** [0.018, 0.476], CI excluding zero. The clearest positive result in E4.
- **Rehearsal helps, weakly:** replay − naive = 0.241 [−0.004, 0.472], an interval essentially
  touching zero.
- **Faster shortcut collapse did not help.** The reverse order beat the greedy order, 0.731 vs
  0.596, difference −0.136 [−0.311, 0.038] — CI includes zero, so neither direction is established.

**Two design errors, ours, reported at full prominence.** (i) Only 1, 1 and 3 of 8 COOL streams pass
the gate against 7–8 elsewhere, because concept rehearsal on the model's own pseudo-labels breaks
current-task label accuracy; no COOL comparison is interpretable, and E4 was *not* re-run to rescue
it. (ii) The third arm varies task *identity*, not order, and is excluded from ordering conclusions.
Reported "forgetting" is negative throughout because phase 1 is shortcut-grounded near zero; in this
setting the quantity measures acquisition, not forgetting.

## Summary of hypotheses

| | outcome |
|---|---|
| H1 `\|RS\|` predicts grounding | **supported in magnitude** (E1 P2, E2 P3); E1's monotonicity prediction P1 **failed** |
| H3 survives confound controls | **supported** (partial Spearman −0.775) |
| Oracle predicts *which* shortcut | **supported** (97.1% among converged models) |
| Sequential arrival costs grounding | **supported** (0.243 [0.018, 0.476]) |
| H2 margin beats binary | falsified (ΔR² = 0.0015) |
| **H4 selection beats baselines on cost** | **falsified twice** (E3 at `k=6`; E5 across `k ≤ 30`) |
| H5 identifiability/optimisability trade-off | not supported outside one family |
| H6 uniform shortcut selection | withdrawn (did not replicate) |
| H7 theory under-specifies SGD | rejected (convergence artefact) |
| RS lock-in | essentially absent |

Six of eight numbered hypotheses failed, including both intended contributions.
