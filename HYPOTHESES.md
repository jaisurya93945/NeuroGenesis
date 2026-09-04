# HYPOTHESES.md

Predicted directions are fixed **before** confirmatory runs; `paper/preregistration.md` freezes
them at M5. Everything not listed here is exploratory and will be labelled as such.

## Primary

**H1 — identifiability predicts grounding.**
Mean `Acc(C)` decreases monotonically in `|RS|`. On the mod-10 family, `Acc(C)(|RS|=1) − Acc(C)(|RS|=10) > 0.5`.
*Null H1₀:* `Acc(C)` is independent of `|RS|`; grounding is set by optimisation bias, not symmetry.

**H2 — FALSIFIED by E2.** *(had been flagged as "most likely the real contribution")*
Adding the margin to a model that already knows whether `|RS| = 1` explains a further **0.15%** of
variance (`ΔR²` = 0.0015 [0.0004, 0.0042]). The decisive cell failed outright: identifiable tasks
with margin < 0.01 ground at 0.953 versus 0.990 for margin > 0.1 — a gap of 0.04 against >0.3
predicted. Binary identifiability is doing the work. See `RESULTS.md` §7.2. Original statement:
`margin(T)` predicts `Acc(C)` better than binary identifiability and better than `log|RS|`.
Tested by nested-model `ΔR²` with a bootstrap CI. Motivation: a task can be identifiable only via a
handful of very low-probability support tuples (`generators/threadbare.py` constructs exactly these).

## Secondary

**H3 — not a confound.** The `log|RS|` ↔ `Acc(C)` association survives controlling for `I(C;Y)`,
`H(Y)`, `|Y|`, `k`, `n`, render noise, and the realised-vs-declared support gap. Partial Spearman
with bootstrap CI; residual scatter shown, not just a coefficient.

**H4 — selection beats the standard mitigation.** Greedy RS-cover selection ≥ information-greedy
selection and ≥ per-concept supervision at matched annotation cost, on `Acc(C)`, across the whole
budget curve (not at one cherry-picked point).

## Falsifiers, and why each is interesting

| Outcome | Reading |
|---|---|
| `\|RS\|=1` tasks also ground poorly | Identifiability is **not sufficient**; optimisation dominates. Directly informs the field's current framing. |
| `\|RS\|=10` tasks ground **well** | Symmetry permits but SGD *selects* the identity. The programme then needs the selection story, not the symmetry story. |
| `alpha_hat ∉ RS(T)` frequently | Deterministic-RS theory under-specifies what SGD finds (mixtures, finite data). Publishable alone; invisible to `Acc(C)`. |
| information-greedy matches greedy-RS | The RS machinery adds nothing over "pick informative tasks". **This is the headline negative and gets reported as such, not buried.** |

## Emerged from evidence (exploratory — needs its own preregistration)

**H5 — NOT SUPPORTED outside the modular family (E2).** Convergence rate vs `log|RS|` across
152 heterogeneous tasks: Spearman **+0.034**, indistinguishable from zero. The E1 pattern appears to
be specific to modular arithmetic at `k=10`. Demoted to a family-specific observation. Original: A task with `|RS|` shortcuts has `|RS|`
global optima, so convergence rate should *increase* with `|RS|` even as grounding *decreases*.

Support so far: E1 gate-passing rates 5/10, 4/10, 10/10, 10/10 for `|RS|` = 1, 2, 5, 10; and a
5-recipe diagnostic in which `|RS|`=10 converged 25/25 while `|RS|`=1 never exceeded 3/5
(`RESULTS.md` §6). Not yet a claim — a finer `|RS|` sweep with convergence rate as the response
variable is the test, planned as an E2 readout.

**H6 — WITHDRAWN.** "SGD selects near-uniformly among permitted shortcuts": fraction grounded
tracked `1/|RS|` in E1's first execution (1.00, 0.50, 0.30, 0.00). The re-run after the seeding fix
gave 1.00, **1.00**, 0.30, 0.10 — the `|RS|`=2 cell breaks it. Withdrawn after one replication.
It was labelled exploratory and never claimed; recorded here because withdrawn hypotheses are part
of the record, not something to quietly drop.

**H7 — REJECTED, same session it was proposed.** "The deterministic RS set under-specifies what
SGD finds." Triage (`scripts/diagnose_h7.py`) shows the shortfall is incomplete convergence, not a
gap in the formalism: out-of-set runs average `Acc(Y)` 0.908 versus 1.000 for in-set runs, and among
runs reaching `Acc(Y) ≥ 0.999` only **2.9%** land outside RS (vs 19.0% unrestricted). **97.1%
membership among models that actually learned the task** — the theory is *more* accurate than the
preregistered test implied. See `RESULTS.md` §7.5.

## Standing commitments

- Novelty is described as a **"potential research gap"**, never as "novel", until E1 has run.
- No hypothesis is revised after seeing confirmatory results; revisions create a *new* hypothesis
  with its own pre-registration and its own runs.
- Negative results are preserved in `RESULTS.md` with the same prominence as positive ones.
