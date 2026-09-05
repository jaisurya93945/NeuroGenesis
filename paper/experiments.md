# Experimental protocol

## Preregistration

Every experiment's predictions, analysis plan and exclusion rules were committed to version control
**before its data existed**. Each preregistration commit contains no results; the results commit
comes later. The ordering is verifiable in `git log`:

| experiment | preregistration commit | question |
|---|---|---|
| E1 | `M5a: preregister E1 before running it` | does `\|RS\|` predict grounding? |
| E2 | `Preregister E2 ...` | does the graded margin predict it better? |
| E3 | `Preregister E3 ...` | does selection beat the baselines on cost? |
| E4 | `Preregister E4 ...` | do models retain provably forbidden shortcuts? |
| E5 | `Preregister E5 ...` | does the cost verdict flip with vocabulary size? |

Analysis code was committed alongside each preregistration, before the data, so it could not be
shaped to fit the outcome.

## Design principles, and the mistakes that produced them

**Confounds controlled by construction, not by adjustment.** E1's family
`y = (c_1 + w c_2) mod 10` holds support, `|Y|`, the label marginal, `I(C;Y)` and perceptual
difficulty *identical* across conditions while `|RS|` varies over `{1,2,5,10}`. Every shortcut in
that family is a cyclic shift, so a shortcut scores `Acc(C)` exactly 0 — the outcome is bimodal
rather than noisy.

**Convergence gates.** A claim about which shortcut a model chose is meaningless if the model never
learned the task. E1–E2 used a relative gate (`Acc(Y) >= 0.95 x best-in-condition`). That proved
**too permissive for membership questions**: a task whose seeds all top out at 0.92 passes its own
gate, and a half-trained encoder has no well-defined relabelling. It manufactured an apparent 19%
"gap" in the theory that dissolved under stratification (§Results). E3–E5 use an **absolute** gate,
`Acc(Y) >= 0.99`.

**Leakage control, enforced mechanically.** Generator seeds `0–99` are development — the only tasks
any hyperparameter may see. Seeds `>= 1000` are confirmatory, run once with the frozen recipe. The
runner *raises* if a confirmatory seed is used in tuning mode.

**Determinism.** `data_seed` (sampling, splits) is separate from `init_seed` (weights, batch order),
so variance decomposes. This was broken twice — the encoder was constructed *before* `torch`
was seeded, in two different places — and both times the bug was invisible until two supposedly
identical configurations disagreed. It is now structurally impossible: `build_encoder` takes the
seed as an argument, and three regression tests pin it.

**Statistics.** Estimation, not significance: bootstrap 95% CIs, effect sizes (Cliff's delta for
bimodal outcomes), and every per-seed point plotted. **No significance stars anywhere** — at four
conditions and ten seeds, `p < 0.001` is purchasable by adding seeds and would be evidence of
nothing. Confirmatory tests are Holm-corrected within each experiment; everything else is labelled
exploratory.

## Compute

All experiments run on 4 CPU cores. No GPU is required for any result in this paper. Total across
E1–E5: roughly 2,600 training runs.
