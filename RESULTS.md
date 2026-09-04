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

## 7. Training experiments still outstanding

E2, E3 and E4 are specified in `EXPERIMENTS.md` and have not been run.
