# EXPERIMENTS.md

Every experiment records: ID, hypothesis, config hash, commit SHA, dataset version, seed, hardware,
runtime, raw output path, and interpretation. `paper/preregistration.md` freezes E1–E4's predicted
directions before confirmatory runs.

## E1 — minimal falsifiable experiment (M5)

- **Hypothesis:** H1.
- **Design:** `y = (c1 + w·c2) mod 10`, `w ∈ {2, 1, 4, 9}` giving `|RS| ∈ {1, 2, 5, 10}`.
  Tier M (real MNIST digits), 10 seeds per condition. 40 runs, ~1.5 h on 4 cores.
- **Why this design:** support, `|Y|`, label marginal, `I(C;Y)` and perceptual difficulty are
  identical across conditions; only `|RS|` moves. Shortcuts are cyclic shifts, so a shortcut scores
  `Acc(C) = 0` exactly — bimodal outcome, large effect size.
- **Primary readout:** mean `Acc(C)` per condition, with per-seed scatter.
- **Secondary readout:** `rs_membership` — is the recovered `alpha_hat` inside the predicted `RS(T)`?
- **Gate:** runs failing the convergence gate (`Acc(Y) ≥ 0.95 × achievable`) are excluded and the
  exclusion rate reported.
- **Falsifiers:** see `HYPOTHESES.md`. Both directions are informative.

## E2 — correlational study (M7)

- **Hypotheses:** H2, H3.
- **Design:** ~200 generated tasks (algebraic, planted, support-density, threadbare) × 5 seeds,
  Tier S (synthetic renderer) with a Tier-S/Tier-M agreement check on ~20 overlapping tasks.
- **Unit of analysis:** the *task* (n ≈ 200), response = mean `Acc(C)` over seeds.
- **Analysis:** nested-model `ΔR²` (margin vs binary vs `log|RS|`) with bootstrap CI; partial
  Spearman controlling for `I(C;Y)`, `H(Y)`, `|Y|`, `k`, `n`, noise, realised-vs-declared support gap.

## E3 — active selection (M8)

- **Hypothesis:** H4.
- **Methods:** greedy RS-cover (damage-weighted), greedy RS-cover (count-weighted), random subset
  (20 draws), **information-greedy** (the ablation that decides whether the RS machinery earns its
  place — run early), concept supervision at matched annotation cost, all-tasks ceiling,
  exhaustive-optimal subset (separates search quality from objective quality).
- **Reporting:** full performance-vs-budget curve with a sensitivity band over the authoring-cost
  parameter `τ ∈ {0, 100, 1000}`. Never a single matched-budget point.

## E4 — continual (M9)

- **Hypothesis:** H3 (continual form) — identifiability of the *union* of tasks is not realised by
  *sequential* training.
- **Design:** 3 tasks arriving in sequence; orders = greedy-RS, reverse-greedy, random,
  fastest-collapse. Strategies = naive fine-tuning, input replay, COOL-style concept rehearsal,
  EWC. 5 seeds.
- **Novel measurement — RS lock-in:** after training on `T₂`, test whether `alpha_hat` remains in
  `RS(T₁) \ RS(T₁∧T₂)` — i.e. the model retains a shortcut the *current* constraint set provably
  forbids. Hysteresis in RS space, measured directly.

## Compute ledger (4 CPU cores)

E1 ≈ 1.5 h · E2 ≈ 4 h · E3 ≈ 2 h · E4 ≈ 3 h · oracle sweeps ≈ 1 h · dev/tuning ≈ 5 h ≈ **17 h**,
fully resumable via config-hash dedup. Hard rule: any config exceeding 5 min/run is cut.
