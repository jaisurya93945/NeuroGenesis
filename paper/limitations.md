# Limitations and threats to validity

## What the negative result does and does not establish

The central claim is that **shortcut-cover selection loses on cost to concept supervision** on the
instance class studied. That is bounded in specific ways:

- **One shortcut-group family.** Every base task has a *cyclic* shortcut group `Z_k`. Cyclic groups
  have a special property — one correctly-labelled concept determines the whole shift — which is
  very plausibly *why* labels turned out so cheap. A shortcut group without that structure (a large
  non-abelian symmetry, or many independent local swaps) could invert the comparison. **This is the
  most important open direction and the biggest threat to the generality of the conclusion.**
- **Two concept slots, `n = 2`.** Arity was not varied.
- **`k ≤ 30`.** Beyond that the oracle stays fast but training cost grows.
- **Cost model.** `cost(rule) = τ`, `cost(label) = 1`. E5 removed the dependence on `τ` for the
  *scaling* argument, but the exchange rate for real annotators on real domains is still unmeasured
  — and it is now the only route by which the conclusion could reverse. That is an
  annotation-economics question rather than a machine-learning one.
- **The concept-supervision baseline is the weak form, which understates the negative.** Labels are
  selected **at random**. Search summaries of the RS literature report active-learning selection
  reaching >90% concept accuracy in ~50 queries against ~75% for random sampling
  (`LITERATURE.md`, re-sweep 2026-09-06, still `[S]`). A stronger baseline would widen the gap
  against shortcut-cover selection, not close it, so this limitation cannot rescue H4.
- **A published claim we could not check.** Search summaries state that required concept supervision
  "grows linearly with the number of possible concept combinations", which is the assumption E5 tested
  and found reversed in sign. The two may address different quantities (`k^n` asymptotics versus
  empirical labels-to-ground at `n = 2`), but we cannot adjudicate it without the primary source,
  which is egress-blocked here. It is the top item in the blocking re-sweep.

## Scope of the positive results

- **`|RS|` predicts grounding** across 152 tasks and five generator families, but all perception
  after E1 is synthetic. The environment cannot download pretrained vision models, so perceptual
  difficulty is manipulated by adding noise to a random codebook rather than by using realistic
  encoders. This is a real gap relative to where the field is heading.
- **97.1% membership** is measured on models that reach `Acc(Y) ≥ 0.999`. Below that threshold the
  figure degrades smoothly (19% out-of-set at the permissive gate), because a half-trained encoder
  has no well-defined relabelling — not because the theory fails.
- **No high-stakes task.** rsbench's BDD-OIA-style setting is unreachable from this environment, so
  whether any of this matters in safety-relevant deployments is untested.

## Scope of the formal object

- **Deterministic shortcuts only.** `RS(T)` counts deterministic maps. Stochastic optima — mixtures
  whose induced `p(y|x)` still matches — form a strict superset and are not counted. `rs_membership`
  detects when a model lands outside the deterministic set; an LP feasibility check for mixtures is
  named follow-up work, not done.
- **Shared encoder by default.** Per-slot encoders enlarge the shortcut object; supported by the ASP
  backend, but not the headline setting.
- **Functional knowledge only.** Genuinely relational knowledge (one concept tuple admitting several
  labels) is an ASP-backend extension that is not implemented.

## Threats to validity, and how each was handled

| threat | handling |
|---|---|
| `\|RS\|` co-varies with label informativeness | Controlled *by construction* in E1 (identical support, `\|Y\|`, `I(C;Y)`); statistically in E2 (partial Spearman) |
| Oracle semantics wrong | Three independent implementations agreeing on 2000+ comparisons; the oracle↔loss equivalence test |
| Tuning leakage across many tasks | Dev/confirmatory seed ranges enforced by a runner that raises, not warns |
| Seed variance | ≥8 seeds, bootstrap CIs, per-seed scatter, effect sizes for bimodal outcomes |
| Post-hoc analysis choices | Preregistration commits with no data, verifiable by `git log`; analysis code committed alongside |
| Results not reproducible | Raw records ship; a seeding bug that broke this twice is now structurally impossible and pinned by tests |
| Convergence confounding concept claims | Absolute gate from E3 onward, after the relative gate produced a spurious 19% "theory gap" |

## Honest summary

Six of eight hypotheses failed, including both intended contributions. The paper reports a **negative
result with two positive analysis findings**, and the negative is bounded to cyclic shortcut groups,
synthetic perception, and `k ≤ 30`. It is a real answer to a question the field posed, not a general
impossibility claim, and it is stated that way throughout.
