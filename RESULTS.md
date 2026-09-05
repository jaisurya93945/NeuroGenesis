# RESULTS.md

**Rule: every number here traces to a run manifest or a test in `tests/`. Nothing is typed by hand
from memory, and nothing appears before the code that produced it.**

## 1. Oracle validation (M1) — PASSING

Not a research result; a correctness result. It establishes that the RS oracle computes what the
theory says it should, which every later number depends on.

Reproduce: `.venv/bin/python -m pytest tests/test_oracle_analytic.py -q` → **30 passed**.

| Check | Expected | Oracle | Source |
|---|---|---|---|
| MNIST-Addition, `k=10`, full support | `RS = {id}` | `count = 1` | published; re-derived in `RESEARCH.md` §4 |
| `y=(c1+w·c2) mod 10`, `w=1..9` | `gcd(1+w, 10)` | exact match, all 9 | derived in `RESEARCH.md` §4 |
| `y=(Σ wᵢcᵢ) mod m`, `m=5..12`, `n∈{2,3}`, all coprime `w` | `gcd(Σw, m)` | exact match across the grid | derived in `RESEARCH.md` §4 |
| All shortcuts in the mod-10 family are cyclic shifts | yes | yes | — |
| `RS` is a monoid (identity + composition-closed) | yes | yes | `RESEARCH.md` §2 |
| `RS(T₁∧T₂) = RS(T₁) ∩ RS(T₂)` | yes | yes; `w=9` ∩ `w=4` → 5 | basis of set-cover selection |
| Monotonicity: more support never grows `RS` | yes | yes | — |

Resulting `|RS|` for the E1 family:

| `w` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| `\|RS\|` | 2 | **1** | 2 | **5** | 2 | 1 | 2 | 1 | **10** |

Performance note: the `k=10` search space is `10¹⁰` shared maps; pruned DFS settles
MNIST-Addition in ~1 ms, so the oracle is not a bottleneck at this scale.

## 2. Oracle cross-validation (M2) — PASSING

Three implementations of one definition, sharing no code and no algorithm:
pruned-DFS (`oracle/enumerate.py`), declarative ASP via clingo (`oracle/asp.py`), and a literal
no-cleverness enumerator (`tests/test_oracle_reference.py`).

Reproduce: `.venv/bin/pytest tests/test_oracle_differential.py tests/test_oracle_reference.py tests/test_oracle_per_slot.py -q` → **17 passed**.

| Check | Scope | Result |
|---|---|---|
| DFS ≡ ASP, identical map sets | 500 random tasks × {total, partial} × {injective, any} = **2000 comparisons** | exact agreement |
| Naive reference ≡ DFS ≡ ASP | 120 random tasks (`k ≤ 5`) + structured tasks | exact agreement |
| `partial` closure ⊆ `total` closure | 150 tasks | holds |
| permutation-only count = `n_permutations` of the unrestricted set | 150 tasks | holds |
| ASP per-slot ≡ naive per-slot enumeration | structured tasks | exact agreement |

**A real bug this caught.** `RSResult.n_permutations` assumed 2-D maps and silently misreported for
per-slot results (`(R, n_slots, k)` stacks), where it counted a permutation pair as a collapse.
Found only because a per-slot test printed an implausible `perms=0, collapsing=1`. Fixed, with a
regression test. This is the class of quiet wrongness the three-implementation policy exists for.

**A corrected expectation.** Full-support *integer* addition is identifiable under per-slot maps
too: the natural opposing-shift shortcut (`α₀(x)=x+t`, `α₁(x)=x−t`) is blocked because `α` must map
`[k]` into `[k]` and a non-zero shift pushes an endpoint out of range. *Modular* addition removes
that obstruction and does admit per-slot shortcuts. The oracle was right and the initial test
expectation was wrong; both cases are now asserted.

## 3. Oracle ↔ loss binding (M3) — PASSING

The oracle makes a *combinatorial* claim; the trainer optimises a *numerical* objective. This is
the test that they are the same statement. For a given `α`, it hand-builds the encoder emitting
`α(ground truth)`, pushes it through the **real** `semantic_nll`, and asserts

> `loss == 0` and `Acc(Y) == 1`  **iff**  `α ∈ RS(T)`

Reproduce: `.venv/bin/pytest tests/test_oracle_vs_loss.py -q` → **12 passed**.

| Check | Scope | Result |
|---|---|---|
| every oracle shortcut achieves zero loss | 5 structured tasks, all shortcuts | holds |
| every non-shortcut incurs loss (guards against a vacuous oracle) | 40 sampled non-shortcuts per task | holds |
| **exhaustive iff over every map in `[k]^[k]`** | `k=4` addition + 2 modular tasks | holds |
| iff on sparse random supports (where total-vs-partial could diverge) | 25 tasks, all maps | holds |

## 4. Training stack validation (M4) — PASSING

MNIST-Addition, `k=10`, full support (`|RS| = 1`), SmallCNN (61,706 params), 15 epochs,
1 CPU thread, seed 0:

| | |
|---|---|
| test `Acc(Y)` | **0.9764** |
| test `Acc(C)` | **0.9882** |
| test `F1(C)` | 0.9881 |
| concept collapse `Cls(C)` | 0.000 |
| recovered `α̂` | identity — and `rs_membership = True` |
| runtime | 66.9 s (4.5 s/epoch) |

`Acc(C) ≈ 0.99` on identifiable MNIST-Addition is consistent with published DeepProbLog-family
results, which is the point of the check: the stack reproduces a known number before being used to
produce unknown ones. Runtime is 4× inside the 5-min/run budget.

*This is a stack-validation result, not a research finding: that an identifiable task grounds
correctly is already known. The research question is what happens as `|RS|` grows — E1, not yet run.*

## 5. Data integrity (M0) — PASSING

MNIST fetched from `ossci-datasets.s3.amazonaws.com`, all four idx files sha256-verified
(hashes recorded in `src/neurogenesis/data/mnist.py`). Shapes `(60000,28,28)` / `(10000,28,28)`;
per-digit train counts `[5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]`.

## 6. E1 — does provable identifiability predict symbol grounding? (M5)

Preregistered in `paper/preregistration.md` **before** any run. 40 runs (4 conditions × 10 seeds),
`y = (c₁ + w·c₂) mod 10` on MNIST digits. Reproduce: `bash scripts/reproduce_e1.sh`.

**E1 was run twice.** The first execution happened while a seeding bug meant `init_seed` did not
control weight initialisation, so those runs could not be regenerated from their recorded seeds.
The bug was fixed and E1 re-run. **The numbers below are the post-fix run**; the pre-fix records are
archived at `results/runs/archived_e1_preseedfix.jsonl` and compared in §6.3.

### 6.1 Primary result (post-fix)

| `\|RS\|` | `w` | n (gated) | excluded | test `Acc(C)` mean [95% CI] | test `Acc(Y)` | fraction grounded | `α̂ ∈ RS` |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 5 | 5 | **0.9858** [0.9842, 0.9869] | 0.9757 | 1.00 | 1.00 |
| 2 | 1 | 3 | 7 | 0.9881 [0.9865, 0.9900] | 0.9763 | 1.00 | 1.00 |
| 5 | 4 | 10 | 0 | 0.2965 [0.0015, 0.5917] | 0.9767 | 0.30 | 1.00 |
| 10 | 9 | 10 | 0 | **0.0998** [0.0011, 0.2963] | 0.9723 | 0.10 | 1.00 |

- **P1 (monotone non-increasing in `|RS|`): NOT MET.** `|RS|`=2 sits 0.002 *above* `|RS|`=1.
  Both are ≈0.986 and the `|RS|`=2 cell has n=3, so this is noise in a thin condition rather than a
  reversal — but the prediction as written failed, and it is recorded as failed.
  Permutation trend test `p = 0.00010`.
- **P2 (`Acc(C)[|RS|=1] − Acc(C)[|RS|=10] > 0.5`): MET.** Difference **0.8860** [0.6892, 0.9852],
  Cliff's delta **+0.960**.
- **P3:** all 5 converged seeds in the identifiable condition grounded (5/5); 5 failed the gate.
- **P4:** gated `Acc(Y)` is flat (0.972–0.977) across all four conditions; exclusion rates are not.
- **P5 (`α̂ ∈ RS(T)`): MET at 100%**, in every condition, in **both** executions — 57/57 gated runs
  across the two runs combined. The single most robust finding in the experiment.

### 6.2 What is and is not established

**Established.** Tasks with many provable shortcuts ground far worse than identifiable ones
(`|RS|`=1 → 0.986 versus `|RS|`=10 → 0.100, complete-ish separation, Cliff's delta +0.96), while
label accuracy is unaffected. Every model that converged landed on a relabelling the oracle had
predicted in advance — never outside it.

**Not established.** The *fine-grained ordering* across intermediate `|RS|`. The `|RS|`=2 condition
retained only 3 of 10 runs after the convergence gate, and its mean moved by 0.49 between the two
executions. Nothing should be concluded from the middle of this table.

### 6.3 Before/after the seeding fix — the thin cells were never evidence

`scripts/compare_e1_reruns.py`:

| `\|RS\|` | gated old→new | `Acc(C)` old | `Acc(C)` new | Δ |
|---|---|---|---|---|
| 1 | 5/10 → 5/10 | 0.9866 | 0.9858 | **−0.0008** |
| 2 | 4/10 → 3/10 | 0.4946 | 0.9881 | **+0.4935** |
| 5 | 10/10 → 10/10 | 0.2977 | 0.2965 | **−0.0012** |
| 10 | 10/10 → 10/10 | 0.0016 | 0.0998 | +0.0982 |

The pattern is clean and worth stating: **the two conditions with 10 gated runs reproduced to three
decimal places; the conditions with ≤5 did not.** The well-powered cells are stable under a change
that re-randomised every weight initialisation; the thin cells are not. That is the strongest
available evidence that the thin cells carried no weight — and they were flagged as thin *before*
this comparison existed.

### 6.4 An exploratory pattern that did NOT replicate — withdrawn

The first execution showed fraction-grounded tracking `1/|RS|` (1.00, 0.50, 0.30, 0.00 vs 1.00,
0.50, 0.20, 0.10), and it was recorded as exploratory. The re-run gives **1.00, 1.00, 0.30, 0.10**.
The `|RS|`=2 cell breaks the pattern outright.

**H6 is withdrawn.** It was labelled exploratory, never claimed, and did not survive one replication.
This is what that label is for.

### 6.5 Convergence caveat — diagnosed, and part of the phenomenon

Exclusion rates were uneven (5, 7, 0, 0 post-fix; 5, 6, 0, 0 pre-fix): low-`|RS|` conditions had
many runs that never fit the label. `experiments/diag_convergence.py` (50 runs, **dev tasks only**)
tried five frozen recipes:

| recipe | `w=2` (`\|RS\|`=1) | `w=9` (`\|RS\|`=10) |
|---|---|---|
| baseline (E1's) | 1/5 | **5/5** |
| lower lr (3e-4) | 1/5 | **5/5** |
| smaller batch (64) | 3/5 | **5/5** |
| longer (30 epochs) | 2/5 | **5/5** |
| lower lr + longer | 2/5 | **5/5** |

**Robust across every recipe.** `|RS|`=10 converged 25/25; `|RS|`=1 never exceeded 3/5. Not a
hyperparameter artifact.

*Mechanism hypothesis, not demonstrated:* a task with `|RS|` shortcuts has `|RS|` global optima, so
more shortcuts give SGD more targets. Optimisation gets easier exactly as identification gets
harder. Recorded as exploratory **H5**; a finer `|RS|` sweep with convergence rate as the response
is the test, planned as an E2 readout.

This means the differential exclusion is *not* a nuisance confound better tuning would remove — it
is an effect of the manipulated variable. It also explains why the low-`|RS|` cells are thin: those
are exactly the conditions that converge least often.

## 7. E2 — does the margin beat binary identifiability? (M7)

Preregistered in `paper/preregistration_e2.md`; analysis code committed before the data existed.
830 runs, 166 tasks, 5 generator families, Tier S. After declared exclusions (2 degenerate `|Y|=1`,
1 truncated oracle): **152 analysable tasks**. Reproduce: `python scripts/analyze_e2.py`.

### 7.1 Headline: the core survives, the proposed refinement does not

| | Prediction | Result | Verdict |
|---|---|---|---|
| **P1** | margin beats binary (`ΔR² > 0`) | `ΔR²` = **0.0015** [0.0004, 0.0042] | technically met, **scientifically negligible** |
| **P2** | identifiable low-margin tasks ground ≥0.3 worse | difference **0.037** [0.013, 0.060] | **NOT MET — H2 falsified** |
| **P3** | `log\|RS\|`↔`Acc(C)` survives controls | partial Spearman **−0.775** [−0.898, −0.516] | **MET, strongly** |
| **P4** | convergence rate rises with `\|RS\|` | Spearman **+0.034** | technically met, **indistinguishable from zero** |
| **P5** | `α̂ ∈ RS` ≥ 80% of non-grounding runs | **0.784** (n=185) | NOT MET — near miss, and informative |

### 7.2 H2 is falsified, and that is the main result of E2

The `rarefied` family was built precisely to break the binary property: full support, unchanged data
volume, `|RS| = 1`, margin driven down three orders of magnitude. If binary identifiability were
insufficient, these should have failed to ground.

**They ground anyway.** Among the 112 provably identifiable tasks:

| margin band | n | mean `Acc(C)` |
|---|---|---|
| `> 0.1` | 73 | 0.9900 |
| `< 0.01` | 18 | **0.9527** |

Collapsing the refuting evidence to a thousandth of the probability mass costs about **0.04** of
concept accuracy — not the >0.3 predicted. Adding the margin to a model that already knows whether
`|RS| = 1` explains a further **0.15% of variance**.

P1's interval technically excludes zero, and under the preregistered analysis that reads "MET".
It should not be reported that way. My own preregistration committed to estimation over
significance precisely so that a detectable-but-tiny effect would not be dressed up as a finding:
**`ΔR² = 0.0015` means the margin adds essentially nothing over the binary property.**

This was the project's proposed novel contribution — HYPOTHESES.md called H2 "most likely the real
contribution". The evidence says no. **Binary identifiability is doing the work.**

### 7.3 What did survive, and it is the stronger claim

`log|RS|` predicts grounding across 152 heterogeneous tasks spanning five generator families:

- raw Spearman **−0.801**
- partial Spearman **−0.775** [−0.898, −0.516], controlling for label entropy `H(Y)`, `|Y|`, `k`,
  and support size

So the relationship is not a re-description of label informativeness or of how much data the task
carries. E1 established this on one hand-picked family; E2 shows it holds on tasks generated by
planted symmetries, random label tables, support thinning and modular arithmetic alike.

### 7.4 H5 is not supported here

Convergence rate versus `log|RS|`: Spearman **+0.034**. The prediction said "positive", and +0.034
is positive, so the letter of P4 is met — but it is indistinguishable from zero and is reported as
**not supported**.

H5 came from E1's Tier-M diagnostic, where identifiable modular tasks converged 5/10 and
high-`|RS|` ones 10/10. On a heterogeneous task set that pattern does not generalise. The most
likely reading: it was a property of *modular arithmetic at k=10*, not of identifiability. H5 is
demoted to a family-specific observation.

### 7.5 The P5 near-miss was a convergence artefact — and the theory comes out *stronger*

`α̂ ∈ RS(T)` held for 78.4% of converged non-grounding runs, against 80% predicted and 100% in E1.
That looked like a gap in the formalism, and `HYPOTHESES.md` recorded it as a new H7:
"the deterministic RS set under-specifies what SGD finds". Before claiming it, the cheap
explanations were tested (`scripts/diagnose_h7.py`).

**It does not survive.** Out-of-set runs are systematically *less converged*:

| | out-of-set runs | in-set runs |
|---|---|---|
| mean `Acc(Y)` | **0.9084** | **1.0000** |
| `α̂` coverage | 1.000 | 1.000 |

Coverage is perfect in both, so the mode estimator is not at fault. Stratifying by how well each run
fitted its own objective is decisive:

| require `Acc(Y) ≥` | n | fraction outside RS | median violation mass |
|---|---|---|---|
| 0.0 (all gated runs) | 210 | **19.0%** | 0.0099 |
| 0.95 | 195 | 12.8% | 0.0030 |
| 0.99 | 188 | 9.6% | 0.0030 |
| **0.999** | 175 | **2.9%** | 0.0010 |

**Among models that actually learned the task, 97.1% land on a relabelling the oracle predicted in
advance.** The 19% figure is dominated by runs that had not finished learning — they are not
choosing an unpredicted shortcut, they are not yet choosing anything.

So the correct reading is the opposite of the one E2's raw P5 suggested: **the deterministic RS
theory is more accurate than the preregistered test implied, not less.** H7 is rejected.

**Methodological lesson, recorded because it will recur.** The preregistered convergence gate
(`Acc(Y) ≥ 0.95 × best-in-task`) is appropriate for comparing *accuracy* across conditions but is
too permissive for *membership* questions: a task whose seeds all top out at 0.92 passes its own
gate at 0.92, and a half-learned encoder has no well-defined relabelling to test. Membership
questions need an absolute convergence criterion, and E3/E4 will use one.

**A bug this triage caught in the triage itself.** The first version of the diagnostic did not
exclude tasks whose oracle enumeration hit the 200,000-map limit. On a truncated result
`RSResult.contains` can return `False` for a genuine shortcut that simply fell outside the
enumerated prefix — which produced six impossible cases with *exactly zero* violation mass reported
as "outside RS". Noticing that a shortcut cannot both violate nothing and be outside the set is what
exposed it. `analyze_e2.py` had excluded truncated tasks all along, so **P5 = 0.784 was never
affected**; only the diagnostic was.

### 7.6 Status of the hypotheses after E2

| | |
|---|---|
| **H1** (identifiability predicts grounding) | supported, and generalised beyond E1's family |
| **H2** (margin beats binary) | **falsified** |
| **H3** (survives confound controls) | supported |
| **H5** (identifiability/optimisability trade-off) | not supported outside the modular family |
| **H6** (uniform shortcut selection) | withdrawn after E1's re-run |
| **H7** (deterministic theory under-specifies SGD) | **rejected** — the shortfall was incomplete convergence; 97.1% membership among converged models |

## 8. E3 — does selecting tasks by shortcut coverage beat the alternatives? (M8)

Preregistered in `paper/preregistration_e3.md`; analysis code committed before the data existed.
512 runs (2 instances × 32 cells × 8 seeds), absolute gate `Acc(Y) ≥ 0.99` — **100% of runs passed**.
Reproduce: `python scripts/analyze_e3.py`.

**E3 was run twice.** The first execution reintroduced the seeding bug (the E3 driver built the
encoder before seeding torch), so its runs were not reproducible. Caught when an exploratory sweep
disagreed with the main run at the same setting. Archived at
`results/runs/archived_e3_preseedfix.jsonl`; numbers below are the deterministic re-run.

### 8.1 Primary instance, `y = (c₁ + 5·c₂) mod 6`, `|RS| = 6`

| method | budget | `\|RS\|` | test `Acc(C)` [95% CI] | grounded | chosen |
|---|---|---|---|---|---|
| base only | 0 | 6 | 0.250 [0.000, 0.625] | 0.25 | — |
| greedy RS-cover | 1 | 2 | 0.375 [0.125, 0.750] | 0.38 | mod3_c0 |
| **greedy RS-cover** | **2** | **1** | **1.000 [1.000, 1.000]** | **1.00** | mod3_c0, mod2_c0 |
| information-greedy | 1 | 6 | 0.250 [0.000, 0.625] | 0.25 | **inv_diff** (a distractor) |
| information-greedy | 2 | 6 | 0.125 [0.000, 0.375] | 0.12 | both distractors |
| information-greedy | 3 | 2 | 0.250 [0.000, 0.625] | 0.25 | 2 distractors + mod3_sum |
| exhaustive optimum | 2 | 1 | 1.000 [1.000, 1.000] | 1.00 | mod2_c0, mod3_c0 |
| random (5 draws) | 2 | 2 | 0.375 [0.225, 0.525] | 0.38 | — |
| random (5 draws) | 3 | 1 | 0.675 [0.525, 0.825] | 0.68 | — |
| **concept supervision** | **50 labels** | 6 | **1.000 [1.000, 1.000]** | **1.00** | — |

### 8.2 The method works exactly as designed — and still loses

- **P1 MET.** Greedy grounds at task budget **2**; information-greedy **never** grounds at any
  budget tried. It spends its budget on the shift-invariant distractors, which carry high label
  entropy and eliminate no shortcuts.
- **P2 MET.** Greedy matches the exhaustive optimum on final `|RS|` at every budget. The *search* is
  not the weak point.
- **P3 MET.** Greedy beats random at matched budget: +0.625 [0.475, 0.775] at budget 2, +0.325
  [0.175, 0.475] at budget 3. (Budget 1's interval includes zero.)
- **P4 NOT MET.** **This is the one that mattered.** The cheapest concept supervision that grounds is
  **50 labels**. Greedy needs 2 authored auxiliary tasks. Over the preregistered range
  `τ ∈ {25, 50, 100, 200, 400}`, greedy *ties* at `τ = 25` and loses everywhere above it. It would
  win only if authoring a symbolic rule cost **less than 25 concept labels** — and labelling 25 digits
  is plainly cheaper than writing and validating a rule.

**So: no design contribution.** The preregistration committed to saying exactly that if P4 failed.
The selection machinery is correct, beats its ablations, and matches the optimum — and none of that
matters, because the baseline it needed to beat is simply cheaper.

The secondary `k = 8` instance is worse for the method: greedy plateaus at `|RS| = 2` and reaches
only 0.375, while concept supervision grounds at 25 labels.

- **P5 NOT MET, but on a threshold I set badly.** Distractors moved `Acc(C)` from 0.250 to 0.125,
  `|Δ| = 0.125` against a 0.05 threshold. With 8 seeds and a near-binary outcome, the mean is
  quantised in steps of 0.125 — *one seed* — so a 0.05 threshold could not have been met by anything
  except an exact tie. The mechanism is verified independently: the distractors leave `|RS| = 6`
  unchanged by the oracle. The prediction was untestable as written; that is my error, not evidence
  that distractors help.

### 8.3 Where the project stands

| hypothesis | outcome |
|---|---|
| H1 identifiability predicts grounding | **supported** (E1, E2) |
| H3 survives confound controls | **supported** (E2, partial Spearman −0.775) |
| H2 margin beats binary | **falsified** (E2) |
| H4 selection beats the baselines on cost | **falsified** (E3, P4) |
| H5 identifiability/optimisability trade-off | not supported outside one family |
| H6 uniform shortcut selection | withdrawn |
| H7 theory under-specifies SGD | rejected (97.1% membership among converged models) |

**Five of seven hypotheses failed.** What survives is real and well-measured, but it is *analysis*:
the shortcut count predicts grounding across heterogeneous task families, and the oracle predicts
which wrong grounding a converged model adopts. Both generalise existing results rather than
establishing a new mechanism, and the design/intervention question the project was built around has
a negative answer.

That is the honest state. It is a publishable negative — the cost comparison against concept
supervision is exactly the question the JAIR 2026 survey poses, and answering it "no, on this
instance class" is informative — but it is not the contribution the project set out to make.

## 9. E5 — does the cost verdict flip at larger vocabularies? No. It gets worse. (M11)

Preregistered in `paper/preregistration_e5.md`, committed with no data. 864 runs, absolute gate
`Acc(Y) ≥ 0.99` — **100% passed**. Reproduce: `python scripts/analyze_e5.py`.

E3 killed H4 at `k = 6`. E5 was the one remaining route to reviving it: the hypothesis that concept
labels must scale with vocabulary size while rules need not, so the comparison flips at large `k`.

**The asymmetry exists and points the wrong way.**

| `k` | factors | `L(k)` labels to ground | `R(k)` rules to ground | base `Acc(C)` |
|---|---|---|---|---|
| 6 | 2·3 | **80** | 2 | 0.125 |
| 8 | 2³ | 40 | **never** | 0.125 |
| 10 | 2·5 | 40 | 2 | 0.125 |
| 12 | 2²·3 | 40 | 2 | 0.000 |
| 15 | 3·5 | 40 | 2 | 0.000 |
| 16 | 2⁴ | 40 | **never** | 0.250 |
| 20 | 2²·5 | 40 | 2 | 0.000 |
| 24 | 2³·3 | 20 | 2 | 0.000 |
| 30 | 2·3·5 | **10** | 2 | 0.000 |

- **P1 NOT MET — and reversed.** Predicted Spearman(`k`, `L`) > 0.8; observed **−0.906**.
  Predicted `L(30)/L(6) ≥ 4`; observed **0.12**. Labels needed to ground *fall* by 8× as the
  vocabulary grows 5×.
- **P2 MET.** `R(k) = 2` for every composite `k`, as the combinatorics require.
- **P3 NOT MET.** A crossover exists only at `τ = 25` and only at `k = 6`. At `τ ≥ 50` there is no
  `k ≤ 30` where selection is cheaper.
- **P4 MET.** Prime powers are a structural blind spot: at `k = 8, 16` no rule budget grounds
  (best `Acc(C)` 0.500 and 0.625), while 40 concept labels do.

### The artifact check that had to pass first

A −0.906 correlation invites the obvious suspicion that large-`k` tasks ground on their own, making
`L(k)` meaningless. They do not: base `Acc(C)` with zero rules and zero labels is **0.000–0.250 at
every `k`** (column above). Nothing grounds spontaneously; the labels are doing the work.

### What this means

**H4 is permanently dead.** The cost-model attack was the last route to a design contribution, and
it failed in the strongest possible way — not ambiguously, but with the key quantity moving opposite
to the hypothesis. Concept supervision does not merely win at `k = 6`; it becomes **more**
competitive as vocabularies grow, while rule cost stays flat and rules stop working entirely on
prime powers.

*Exploratory, not preregistered:* the mechanism is plausibly that breaking a cyclic shortcut group
needs `O(1)` label information at any `k` — one correctly-labelled concept determines the shift —
so labels never had to scale. What does scale is how hard it is for SGD to maintain a *consistent*
30-way rotation while a handful of supervised examples contradict it. That is a hypothesis about
optimisation dynamics, on 9 vocabulary sizes in one shortcut-group family, and would need its own
preregistration to be more than a story.

### Ledger after E5

| hypothesis | outcome |
|---|---|
| H1 `\|RS\|` predicts grounding | **supported** |
| H3 survives confound controls | **supported** |
| H2 margin beats binary | falsified |
| **H4 selection beats baselines on cost** | **falsified twice** — E3 at `k=6`, E5 across `k ≤ 30` |
| H5, H6, H7 | demoted / withdrawn / rejected |

Six of eight hypotheses failed. What survives is analysis, not design, and the design question now
has a decisive negative answer rather than an open one — which is a better outcome than ambiguity.

## 10. E4 — do models keep shortcuts their constraints now forbid? Essentially never. (M12)

Preregistered in `paper/preregistration_e4.md`, committed data-free. 120 streams;
**98 pass the final-phase gate** `Acc(Y) ≥ 0.99`. Reproduce: `python scripts/analyze_e4.py`.

| order | strategy | n | final `Acc(C)` [95% CI] | lock-in |
|---|---|---|---|---|
| greedy | naive | 8 | 0.456 [0.231, 0.669] | 0.00 |
| greedy | replay | 7 | 0.877 [0.631, 1.000] | 0.00 |
| greedy | ewc | 8 | 0.438 [0.252, 0.644] | 0.00 |
| greedy | joint *(ref)* | 8 | 1.000 [1.000, 1.000] | 0.00 |
| reverse | naive | 8 | 0.665 [0.559, 0.791] | 0.00 |
| reverse | replay | 7 | **1.000** [1.000, 1.000] | 0.00 |
| reverse | ewc | 8 | 0.528 [0.418, 0.675] | 0.00 |
| alt-taskset | naive | 8 | 0.176 [0.021, 0.426] | 0.00 |
| alt-taskset | joint *(ref)* | 8 | 0.750 [0.375, 1.000] | 0.00 |
| *(all `cool` cells)* | | **1, 1, 3** | *unusable — see caveat 1* | |

### The headline: RS lock-in is essentially absent

**2 of 98 streams** show a model retaining a shortcut its current constraint set provably forbids —
and both sit inside the one cell that is unusable (caveat 1). The analysis script reports P1 as
"MET" on that basis; **that verdict should not be believed**. The honest reading is the preregistered
alternative: *models escape forbidden shortcuts once the forbidding constraint arrives.*

`preregistration_e4.md` P1 committed in advance to treating this as a real finding rather than a
failed experiment, so it is reported as one. It also undercuts a motivation the field has for
concept rehearsal: in this setting the worry that models get *stuck* on stale groundings does not
materialise. What actually limits sequential grounding is that later constraints are only partly
exploited — not that earlier shortcuts persist.

### Other preregistered outcomes

- **P2 MET, but weakly.** `replay − naive = 0.241`, CI **[−0.004, 0.472]** — the interval essentially
  touches zero. Rehearsal helps, but this does not establish by how much.
- **P3 MET.** Joint 0.917 vs best sequential (replay) 0.673, difference **0.243 [0.018, 0.476]**,
  CI excluding zero. **Sequential arrival genuinely costs grounding** even when the final constraint
  set is identical — the clearest positive result in E4.
- **P4 NOT MET, and directionally reversed.** The *reverse* order (`|RS|` 6→3→1) beat the greedy
  order (6→2→1): 0.731 vs 0.596, difference **−0.136 [−0.311, 0.038]**. The CI includes zero, so the
  reversal is not established either — but the prediction that faster shortcut collapse helps
  is not supported.

### Caveat 1 — differential exclusion destroys the COOL arm (my design error)

Only **1, 1 and 3 of 8** COOL streams pass the gate, against 7–8 for every other arm. Concept
rehearsal on the model's *own* pseudo-labels pulls the encoder toward its earlier grounding hard
enough to break current-task label accuracy. That is a real and interesting effect — and it means
**no comparison involving COOL is interpretable here.** Its apparent 1.000 scores are one or three
surviving seeds, and the only non-zero lock-in cell is `alt-taskset|cool` at n=3.

This is the same differential-exclusion trap E1 hit, and I did not anticipate it when designing E4.
**E4 was not re-run to rescue it**: re-running after seeing the outcome is precisely the post-hoc
rescue this project has refused for five experiments. The exclusion rate *is* the result.

### Caveat 2 — the third arm is a different task set, not a different order (my design error)

The arm labelled `random` uses `mod3_sum` where greedy/reverse use `mod3_c0`. It is therefore a
**different task set**, not a permutation of the same one, and its joint reference reaches only
0.750 against 1.000 — the task set itself is harder. It is renamed **`alt-taskset`** above and must
not be read as evidence about ordering. P4's greedy-vs-reverse comparison is unaffected: those two
*are* permutations of the same pair.

### A metric that means the opposite of its name here

Reported "forgetting" (`Acc(C)` at phase 1 minus current) is **negative throughout** (−0.24 to
−0.84) — grounding *improves* across the stream. That is expected: phase 1 has `|RS| = 6` and is
shortcut-grounded near zero, so later constraints can only help. The quantity is retained for
comparability with the continual-learning literature, but in this setting it measures *acquisition*,
not forgetting, and should be read that way.

### Ledger after E4

| hypothesis | outcome |
|---|---|
| H1, H3 | supported |
| H2, H4 (×2), H5, H6, H7 | falsified / demoted / withdrawn / rejected |
| **RS lock-in** (E4's distinctive measurement) | **essentially absent** — a clean negative |
| **Sequential arrival costs grounding** (P3) | **supported**, CI excluding zero |
