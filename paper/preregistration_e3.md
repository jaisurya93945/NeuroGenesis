# Preregistration — E3

**Committed before any E3 run.** As with E1 and E2, this commit contains no E3 results.

Code state: multi-task training implemented and verified (a base-task shortcut is loss-free on the
base alone but costs 19.4 once the selected auxiliaries are added — the set-cover mechanism visible
at the loss level). E1, E2 complete. H2, H5, H6, H7 all failed; see `RESULTS.md`.

---

## 1. Why E3 matters more than it did

E3 was originally one experiment among four. After E2 it is **the only remaining route to a design
contribution**. What survives so far — `|RS|` predicts grounding, and the oracle predicts which
wrong grounding a converged model adopts — is *analysis*, and a generalisation of known results.

E3 asks the design question: **given a budget, does selecting auxiliary tasks by shortcut coverage
beat the alternatives — including the expensive mitigation the JAIR 2026 survey says the field
relies on?**

If greedy RS-cover does not beat information-greedy selection and concept supervision at matched
cost, this project has no design contribution, and that will be stated plainly.

## 2. Setup

**Primary instance.** `y = (c₁ + 5·c₂) mod 6`, `|RS| = 6` (all six cyclic shifts).
Pool: `divisor_modular_pool(6)` — six candidates whose individual effects are oracle-verified:

| candidate | `\|RS\|` with base |
|---|---|
| `mod2_c0` | 3 |
| `mod3_c0` | 2 |
| `mod3_sum` | 2 |
| `mod2_sum` | 6 (no help) |
| `inv_diff` | 6 (shift-invariant distractor) |
| `inv_diff2` | 6 (shift-invariant distractor) |

No single candidate reaches identifiability; `mod2_c0 + mod3_c0` does. That is a genuine weighted
set-cover instance (see `DECISIONS.md` D7 for why the first, generic pool had to be replaced).

**Secondary instance.** `k = 8`, where greedy plateaus at `|RS| = 2` and cannot reach 1. Included
deliberately: a method that only looks good when it can reach a perfect answer is not much of a method.

**Perception.** Tier S, `noise=0.1`, `dim=32`, MLP, 30 epochs, 8000 train tuples — the frozen E2
recipe, which grounds `k=6` modular tasks reliably. 8 seeds per cell.

**Absolute convergence gate: `Acc(Y) ≥ 0.99`.** Not E2's relative gate. E2's
`Acc(Y) ≥ 0.95 × best-in-task` was too permissive and manufactured a spurious 19% "theory gap" that
was really unfinished training (`RESULTS.md` §7.5). Concept claims need models that actually learned
the task.

## 3. Methods compared

| arm | what it is |
|---|---|
| **greedy RS-cover** | the proposed method: damage-weighted greedy over shortcut coverage |
| **information-greedy** | pick by label entropy, ignoring shortcut structure — *the ablation that decides whether the RS machinery earns its place* |
| **random subset** | 5 independent draws per budget; guards against "more tasks is just better" |
| **exhaustive optimum** | separates search quality from objective quality |
| **concept supervision** | the standard expensive mitigation, at 25/50/100/200/400 labels |
| **all tasks** | unmatched ceiling, reference line only |
| **base only** | floor |

## 4. Cost model, and why the answer will hinge on it

`cost(auxiliary task) = τ` (authoring a rule); `cost(concept label) = 1`. `τ` is contestable and
depends entirely on who annotates, so **no single τ is reported**. Results are a curve over
`τ ∈ {25, 50, 100, 200, 400}`, and the headline is the **minimum cost at which each method reaches
`Acc(C) > 0.9`**, not accuracy at one arbitrary matched point.

**Stated in advance, from a single dev seed:** concept supervision with 200 labels already reaches
`Acc(C) = 1.000`, and greedy reaches it with 2 tasks. So at `τ = 100` the two arms *tie* on outcome
and the comparison reduces entirely to `τ`. This is disclosed now because discovering it afterwards
and then choosing a favourable `τ` would be exactly the manipulation the curve exists to prevent.

## 5. Predictions (fixed now)

**P1 (primary).** Greedy RS-cover reaches `Acc(C) > 0.9` at a strictly smaller *task budget* than
information-greedy. Direction: greedy at budget ≤ 2, information-greedy at budget ≥ 3 or never.

**P2.** Greedy matches the exhaustive optimum on final `|RS|` at every budget (a statement about the
search, not the objective).

**P3.** Random selection at matched budget grounds strictly less often than greedy, averaged over
its 5 draws.

**P4 (the one that decides whether there is a contribution).** There exists a non-empty range of `τ`
over which greedy RS-cover reaches `Acc(C) > 0.9` at strictly lower total cost than concept
supervision. Given the dev observation above, this range is expected to be `τ < 100`.
*If no such range exists, the method has no cost advantage and E3 is reported as a negative result.*

**P5.** Adding the two shift-invariant distractors changes `Acc(C)` by less than 0.05 — they cost
budget and eliminate no shortcuts, so a method that cannot tell them from useful tasks wastes its
budget. (Information-greedy is expected to pick them; that is the point of including them.)

## 6. Known risk, declared now

`|RS|` does not perfectly determine grounding at this scale. In the dev smoke test, `|RS| = 3`
(base + `mod2_c0`) grounded while `|RS| = 2` (base + `mod3_c0`) did not — one seed each, but it
shows the objective greedy optimises is not a perfect proxy for the outcome we care about. If this
persists across 8 seeds, then **minimising `|RS|` is the wrong objective**, and that is a finding
about the proposed method rather than a nuisance. It will be reported as such.

## 7. Analysis plan

- Unit: the (method, budget, seed) cell. Response: test `Acc(C)`.
- Bootstrap 95% CIs; per-seed scatter always shown; effect sizes. No significance stars.
- Five confirmatory tests (P1–P5), Holm-corrected.
- Exclusions declared in advance: runs failing the absolute `Acc(Y) ≥ 0.99` gate, reported as a rate
  per arm.

## 8. Commitment

No prediction here is revised after seeing results. A negative outcome on P4 means the project has
no design contribution, and `RESULTS.md` will say exactly that.
