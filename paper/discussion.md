# Discussion

## What we set out to do, and what happened

The project began from a question the JAIR 2026 survey leaves open: reasoning shortcuts are
well characterised, mitigations exist, and all of the effective ones cost concept supervision. Can
the *structure* of the shortcut set be used instead — decided offline, before any training — to buy
grounding more cheaply than annotation?

The answer is no, on this instance class, and the evidence is unusually clean about *why*: not
because the structural method fails, but because it works and is still beaten.

Two things did survive, and they are worth more than the failed design contribution.

## The analysis result: an offline quantity predicts an empirical one

`|RS(T)|` is computed by a deterministic search over relabellings, before a single gradient step, in
milliseconds. It predicts whether the trained model grounds its concepts: partial Spearman
**−0.775** [−0.898, −0.516] across 152 tasks from five generator families, after controlling for
label entropy, `|Y|`, `k` and support size. E1 isolates the same effect by construction — support,
`|Y|`, label marginal, `I(C;Y)` and perceptual difficulty held identical across conditions, only
`|RS|` varying — and separates the extremes by **0.886** [0.689, 0.985], Cliff's delta **+0.960**,
while label accuracy stays flat at ≈0.975. Shortcuts are free in the training objective and
catastrophic in the concepts.

The sharper form of the claim is E2b. The oracle does not merely predict *that* grounding fails; it
predicts *which* wrong grounding is adopted. Among models reaching `Acc(Y) ≥ 0.999`, **97.1%** land
on a relabelling enumerated in advance. The 19% apparent shortfall at a permissive gate was
unfinished training, not a gap in the formalism — a half-trained encoder has no well-defined
relabelling to test. We initially recorded that shortfall as a new hypothesis (H7: the deterministic
set under-specifies SGD) and then rejected it with our own triage.

That is a strong statement about a deterministic combinatorial object describing a stochastic
optimisation process, and it is the finding we would defend hardest.

**What it is good for.** `|RS|` is a cheap pre-training diagnostic. A practitioner can compute it on
a candidate task specification and know, before spending any compute, whether concept-level
correctness is achievable at all from label supervision — and if not, exactly which confusions to
expect. That is real, if modest, and it does not depend on any of the failed hypotheses.

## The design result: the method works and loses anyway

Greedy shortcut-cover selection behaves exactly as its theory says. It grounds at budget 2
(`Acc(C)` 1.000, 8/8 seeds), **matches the exhaustive optimum at every budget**, beats random
selection by **+0.625** [0.475, 0.775] at matched budget, and beats information-greedy selection —
which never grounds, because it spends its budget on shift-invariant distractors carrying high label
entropy and zero shortcut coverage. That contrast is itself informative: **label informativeness and
shortcut coverage are different quantities, and optimising the first does not deliver the second.**

And it loses. The cheapest concept supervision that grounds needs 50 labels; greedy needs 2 authored
rules. Under the preregistered exchange rates `τ ∈ {25, 50, 100, 200, 400}` the methods tie at
`τ = 25` and selection loses everywhere above.

E5 was built to attack that verdict at its weakest joint — the assumption that a rule and a label
are comparable across scales. The defence was that labels must scale with concept-vocabulary size
while rules stay flat, so the comparison flips at large `k`. **The asymmetry exists and points the
wrong way.** Rules do stay flat at 2 for every composite `k` (P2 met). But labels needed to ground
*fall*, from 80 at `k = 6` to 10 at `k = 30`: Spearman(`k`, `L`) = **−0.906** against a predicted
`> 0.8`, `L(30)/L(6) = 0.12` against a predicted `≥ 4`. Concept supervision becomes **more**
competitive as vocabularies grow, not less. The artifact check ran first and passed: base `Acc(C)`
without supervision is 0.000–0.250 at every `k`, so nothing grounds spontaneously and the labels are
doing the work.

H4 is falsified twice, in the two places it could have survived. This is the paper's centre of
gravity, and it is a negative result.

## Why the cost verdict reversed — a hypothesis, not a finding

We did not predict a *decreasing* `L(k)` and we did not test the following explanation. It is stated
as a post-hoc account of an observed effect, and the measurement that would settle it is named.

The base task is `y = (c₁ + (k−1)·c₂) mod k`, so `|Y| = k`. Increasing `k` therefore does two things
at once: it enlarges the concept vocabulary (which should make labelling harder) and it enlarges the
label alphabet (which makes each *label* observation more constraining, ≈`log k` bits instead of
`log 6`). The residual work left for concept labels is only to break a single degree of freedom —
for a cyclic shortcut group, identifying the shift `u ∈ Z_k` — and that residual does not grow with
`k` while the base signal does. On this account the reversal is a property of the experimental
family, not of annotation in general.

**The test:** hold `|Y|` fixed while varying `k` (for example by composing the modular sum with a
fixed surjection onto a smaller label set), and re-measure `L(k)`. If `L` stops falling, the
reversal was a label-alphabet effect and the vocabulary-scaling argument is untested rather than
refuted. That experiment is specified but not run, and no claim here depends on its outcome.

## What would overturn the negative

Three routes, in decreasing order of how much they threaten the conclusion.

**1. Non-cyclic shortcut groups.** Every base task studied has a cyclic shortcut group `Z_k`, which
has a special property: one correctly identified concept determines the entire shift. That is very
plausibly why labels are so cheap here. A shortcut group without that structure — a large
non-abelian symmetry, or many independent local swaps with no shared generator — would force labels
to pay per independent orbit while a single rule could still collapse many at once. This is the
**most important open direction and the largest threat to generality**, and the ASP backend already
supports the per-slot and relational encodings needed to construct such families.

**2. Annotation economics with real annotators.** E5 measures annotation **quantity** under this
project's operational cost model — how many concept labels and how many authored rules this training
protocol needs to reach `Acc(C) > 0.9`. It does **not** measure human annotation time, effort, or
money, and **no human-annotation measurement was performed**. The exchange rate `cost(rule) = τ`
versus `cost(label) = 1` is the load-bearing assumption of the whole comparison, and it remains
unmeasured. If authoring a correct auxiliary rule for a real domain turns out to cost less than 25
concept labels, selection wins at `k = 6` and the verdict is domain-dependent rather than negative.
Settling this needs an annotation study, not a machine-learning experiment.

**3. Asymmetric domains.** Where rules are near-free because a specification already exists
(regulatory constraints, physical invariants, an existing knowledge base) and labelling requires
expert time, the comparison inverts by assumption rather than by evidence. Our result says nothing
about such settings except that the structural machinery is available and correct if you are in one.

There is also a declared boundary that no amount of evidence removes: for prime-power `k` the
divisor-rule family contains **no** identifying rule, so the rule arm plateaus (`Acc(C)` 0.500 at
`k = 8`, 0.625 at `k = 16`) at every budget while labels still ground at 40. That is a structural
blind spot of the rule family, preregistered as P4 and reported as such.

## The continual result, and what it corrects

`RS(T₁ ∧ T₂) ⊆ RS(T₁)`, so a sequentially trained model can occupy a state a jointly trained one
never can: still implementing a shortcut that the constraints it has now been shown provably forbid.
We measured that directly. **It essentially does not happen** — 2 of 98 gated streams, both inside a
single cell that differential exclusion renders uninterpretable.

This matters because the fear of exactly that hysteresis is part of the motivation for concept
rehearsal. In this setting the fear is unfounded: models escape forbidden shortcuts once the
forbidding constraint arrives.

What *does* cost something is sequential arrival itself — joint 0.917 against the best sequential
strategy 0.673, difference **0.243** [0.018, 0.476]. But the mechanism is not retention. Measured
"forgetting" is negative throughout, because phase 1 is shortcut-grounded near zero and later phases
improve on it; in this setting the quantity measures **acquisition, not forgetting**. The sequential
model is not losing grounding it once had — it is failing to acquire grounding the joint model
reaches. That is a different problem from the one continual-learning machinery is built to solve,
and it is the more useful reading of the result.

Two design errors of ours limit E4 and are reported rather than smoothed. Concept rehearsal on the
model's own pseudo-labels breaks current-task label accuracy badly enough that only 1, 1 and 3 of 8
COOL streams survive the gate against 7–8 elsewhere, so **no comparison involving COOL is
interpretable**; E4 was not re-run to rescue it, because re-running after seeing the outcome is the
post-hoc rescue this project has avoided for five experiments. And the third arm varies task
*identity* rather than order, so it is excluded from ordering conclusions. The order prediction P4
came out **reversed** — reverse order 0.731 against greedy 0.596, difference −0.136 [−0.311, 0.038],
CI including zero — so the intuition that faster shortcut collapse is better is not supported, and
neither is its opposite.

## What we are not claiming

- Not that shortcut-cover selection is useless. It is optimal within its own budget and beats every
  baseline tried. It is beaten by a *different* method that costs more of a *different* resource.
- Not a general impossibility result. The conclusion is bounded to cyclic shortcut groups, two
  concept slots, `k ≤ 30`, and — after E1 — synthetic perception. `LIMITATIONS.md` states each bound.
- Not novelty. `LITERATURE.md` is entirely `[S]`-marked search-summary material; arXiv and OpenReview
  are unreachable from the build environment (re-verified 2026-09-05), so no primary source has been
  read end to end. Every relation-to-prior-work statement stays provisional until that re-sweep, and
  the blocking prerequisite is recorded in `ROADMAP.md`.
- Not tier-general, yet. Everything after E1 is synthetic perception; the Tier-M replication is the
  outstanding check, and `preregistration_e2.md` §3 already commits to demoting the Tier-S
  conclusions to tier-specific if the rank order disagrees.

## What a practitioner should take from this

Compute `|RS|` first — it is nearly free, it tells you whether your task specification permits
grounding at all, and it tells you which confusions to expect if it does not. Then, if it does not,
**buy concept labels**, not auxiliary tasks: on everything we measured, a few dozen labels beat the
best possible selection of rules, and they get *relatively* cheaper as the vocabulary grows. Reach
for structural selection when your shortcut group is not cyclic, or when a specification you already
own makes rules genuinely free — and in the first case, measure it, because we have not.
