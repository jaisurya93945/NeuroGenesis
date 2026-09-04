# LIMITATIONS.md

Written before results exist, so it cannot be shaped to flatter them.

## Scope of the formal object

- **Deterministic shortcuts only.** `RS(T)` counts deterministic maps `alpha: [k] -> [k]`. Stochastic
  optima — mixtures of deterministic maps whose induced `p(y|x)` still matches — form a strict
  superset and are **not** counted. The `rs_membership` metric detects when a trained model lands
  outside the deterministic set; an LP feasibility check for mixture optima is scoped out of v1 and
  named as follow-up work.
- **Shared encoder by default.** Per-slot encoders enlarge the shortcut object to an `n`-tuple of
  maps. Supported by the ASP backend, but not the default and not the headline setting.
- **Total-`f` reading.** Counts differ under the `partial` reading. Both are implemented; the choice
  is explicit, never defaulted.
- **Functional knowledge only in v1.** `label_table` encodes a function `[k]^n -> Y`. Genuinely
  relational knowledge (one concept tuple admitting several labels) is a planned ASP-backend
  extension, not present yet.

## Scope of the empirical claims

- **Small concept spaces.** `k <= 10`, `n <= 3`. Whether conclusions transfer to large vocabularies
  is untested and will be stated as untested.
- **MNIST-grade perception only.** No pretrained vision models: this environment cannot download
  any. So "perceptual difficulty" is manipulated synthetically (Tier S noise) rather than by using
  realistic encoders — a real gap versus the foundation-model direction the field is moving toward.
- **No high-stakes task.** rsbench's BDD-OIA-style tasks are unavailable here (data not reachable),
  so the "does this matter in safety-relevant settings" question is out of reach.
- **CPU-scale compute.** ~17 h total. Any effect requiring larger models or longer training is
  outside what this artifact can detect, and absence of an effect will be reported as
  "not detected at this scale", never as "does not exist".

## What the evidence has already cost this project

Two proposed contributions did not survive contact with data, and are recorded here rather than
quietly dropped:

- **H2 (margin over binary identifiability) — falsified by E2.** It had been the project's headline
  candidate. `ΔR²` = 0.0015.
- **H6 (uniform selection among shortcuts) — withdrawn** after failing to replicate across E1's re-run.
- **H5 (identifiability/optimisability trade-off) — demoted** to a modular-family observation after
  E2 found Spearman +0.034 on heterogeneous tasks.
- **H7 (deterministic theory under-specifies SGD) — rejected in the same session it was proposed**,
  once out-of-set runs were shown to be the unconverged ones (2.9% out-of-set among runs reaching
  `Acc(Y) ≥ 0.999`, vs 19.0% unrestricted).

What remains is H1/H3 — real, robust, and a generalisation of known results rather than a new
mechanism — plus the corrected membership result (97.1%), which strengthens the *existing* theory
rather than replacing it. That is a smaller claim than the project set out to make, and it is stated
as such. The remaining route to a design contribution is E3.

## Threats to validity

| Threat | Status |
|---|---|
| `\|RS\|` co-varies with label informativeness | Controlled *by construction* in the mod-10 family; controlled statistically (H3) in the wide sweep |
| Oracle semantics wrong | Three independent implementations + known-answer gates; `test_predicted_rs_achieves_zero_loss` binds the oracle to the actual loss |
| Tuning leakage across many tasks | Dev/confirmatory generator-seed ranges enforced *mechanically* by the runner, not by intention |
| Seed variance | ≥10 seeds, bootstrap CIs, per-seed scatter always shown; data-seed × init-seed variance decomposition |
| Tier S conclusions not transferring to Tier M | Explicit agreement check before committing to the Tier-S sweep |
| Literature moving underneath us | Re-sweep at M5 and again before submission; `LITERATURE.md` versioned |

## Literature-access limitation (important)

arXiv, OpenReview, ACM and Semantic Scholar are **blocked in this environment**. `LITERATURE.md`
was assembled from search-result summaries, not primary PDFs, and every entry is marked `[S]` or
`[V]` accordingly. Re-reading primary sources is a **blocking prerequisite** for any paper claim,
and no novelty statement should survive without it.
