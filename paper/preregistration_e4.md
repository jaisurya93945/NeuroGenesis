# Preregistration — E4

**Committed before any E4 run.** As with E1–E3 and E5, this commit contains no E4 results.

---

## 1. The question only a sequential experiment can ask

`RS(T₁ ∧ T₂) ⊆ RS(T₁)`. So when a second task arrives, some shortcuts that were permitted become
**provably forbidden by the constraints the model has now been shown**.

A model trained jointly never occupies the intermediate state, so it can never exhibit the failure.
A model trained sequentially can:

> **RS lock-in:** after phase 2, `α̂ ∈ RS(T₁) \ RS(T₁ ∧ T₂)` — the encoder is still implementing a
> relabelling that the current constraint set rules out.

This is hysteresis in shortcut space. It is cheap to measure with the oracle already built, and
nothing in the literature measures it directly. It is the distinctive contribution of E4, and it is
independent of E3's negative outcome.

## 2. Design

**Stream.** `k = 6`. `T₁ = y = (c₁ + 5·c₂) mod 6` (`|RS| = 6`), then two auxiliary rules from
`divisor_modular_pool(6)`. Oracle-verified collapse:

| order | `\|RS\|` after phase 1 → 2 → 3 |
|---|---|
| greedy (`mod3_c0`, then `mod2_c0`) | 6 → 2 → 1 |
| reverse (`mod2_c0`, then `mod3_c0`) | 6 → 3 → 1 |
| random | 6 → · → 1 |

**Strategies.** `naive`, `replay` (input rehearsal), `ewc` (diagonal Fisher), `cool` (concept
rehearsal on the model's *own* pseudo-labels, reimplemented from the ICML 2023 description — the
paper's code is not reachable from this environment). Every strategy sees identical supervision for
the current phase; they differ only in what they carry forward.

**Reference arm.** Joint training on all three tasks — the upper bound the sequential arms are
measured against. E3 established joint reaches `Acc(C) = 1.000` on this instance.

**Recipe.** Tier S, `dim = 64`, `noise = 0.1`, MLP, 40 epochs per phase, 8000 train tuples,
8 seeds. Absolute gate `Acc(Y) ≥ 0.99` on the final phase.

3 orders × 4 strategies × 8 seeds = 96 streams, plus 24 joint runs.

## 3. What the dev smoke test already showed (disclosed, not a result)

At reduced settings (8 epochs, 4000 tuples, 1 seed) the stream collapsed `|RS|` as designed and the
strategies separated — `replay` grounded by phase 2 (`Acc(C) = 1.000`), `naive`/`ewc`/`cool` reached
only 0.44–0.66 by phase 3. **No lock-in was observed in that run.** Those settings are far below the
frozen recipe, so this is a smoke test of the instrument, not evidence about the phenomenon.

## 4. Predictions (fixed now)

**P1 — lock-in is detectable at all.** At least one (strategy, order) cell shows a lock-in rate
> 0 at some phase. *If lock-in is zero everywhere, that is itself a clean finding: models escape
forbidden shortcuts once the forbidding constraint arrives, and the worry motivating concept
rehearsal does not materialise in this setting.* Both outcomes are reported with equal prominence.

**P2 — naive is the worst.** Lock-in rate (or, if lock-in is zero, final `Acc(C)`) is worse for
`naive` than for `replay` and `cool`. Direction: `Acc(C)_final(naive) < Acc(C)_final(replay)`.

**P3 — sequential costs something.** Final `Acc(C)` under the best sequential strategy is strictly
below joint training on the same three tasks, by more than 0.05. If sequential matches joint, the
continual framing adds nothing here and that is said plainly.

**P4 — order matters.** The greedy order (which collapses `|RS|` fastest: 6 → 2 → 1) yields higher
final `Acc(C)` than the reverse order (6 → 3 → 1), difference > 0.05.

## 5. Analysis plan

- Unit: the (order, strategy, seed) stream. Primary response: final-phase `Acc(C)`.
  Secondary: per-phase lock-in indicator, concept forgetting (`Acc(C)` at phase 1 minus current).
- Bootstrap 95% CIs; per-seed scatter; effect sizes. No significance stars.
- Four confirmatory tests (P1–P4), Holm-corrected.
- Exclusions declared now: streams failing the final-phase absolute gate, reported as a rate.
- `rs_membership` and lock-in are only computed when the oracle result is **not truncated** — the
  trap that produced six impossible cases in the H7 triage (`RESULTS.md` §7.5). At `k = 6` no
  truncation is expected; the check is asserted anyway.

## 6. Commitment

No prediction is revised after seeing results. A zero lock-in rate is a real answer, not a failed
experiment, and will be reported as one.
