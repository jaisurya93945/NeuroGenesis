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
