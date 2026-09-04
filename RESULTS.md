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

Preregistered in `paper/preregistration.md`, committed at `f367b1c` **before** any run.
40 runs (4 conditions × 10 seeds), `y = (c₁ + w·c₂) mod 10` on MNIST digits.
Reproduce: `bash scripts/reproduce_e1.sh`. Raw records: `results/runs/runs.jsonl`.

### Primary result

| `\|RS\|` | `w` | n (gated) | excluded | test `Acc(C)` mean [95% CI] | test `Acc(Y)` | fraction grounded | `α̂ ∈ RS` |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 5 | 5 | **0.9866** [0.9853, 0.9876] | 0.9775 | **1.00** | 1.00 |
| 2 | 1 | 4 | 6 | 0.4946 [0.0032, 0.9860] | 0.9718 | 0.50 | 1.00 |
| 5 | 4 | 10 | 0 | 0.2977 [0.1002, 0.5935] | 0.9762 | 0.30 | 1.00 |
| 10 | 9 | 10 | 0 | **0.0016** [0.0011, 0.0020] | 0.9737 | **0.00** | 1.00 |

- **P1 (monotone non-increasing in `|RS|`): MET.** Means 0.9866 → 0.4946 → 0.2977 → 0.0016.
  Permutation trend test `p = 0.00005` (reported once, as preregistered; no stars).
- **P2 (`Acc(C)[|RS|=1] − Acc(C)[|RS|=10] > 0.5`): MET.** Difference **0.9850** [0.9836, 0.9861],
  Cliff's delta **+1.000** (complete separation).
- **P3 (identifiable condition grounds in ≥9/10 seeds): partially met.** All 5 *converged* seeds
  grounded (5/5); the other 5 failed the convergence gate — see the caveat below.
- **P4 (`Acc(Y)` comparable across conditions): MET among gated runs** (0.972–0.978, right panel of
  the figure) **but FAILED on exclusion rates** — see caveat.
- **P5 (`α̂ ∈ RS(T)` in ≥80% of non-grounding runs): MET, at 100%.** Every gated run in every
  condition recovered an `α̂` inside the oracle-predicted set — 29/29 including all 19 shortcut runs.

**H1 is supported in this family.** `Acc(C)` is bimodal exactly as the design predicted: every run
lands at ≈0.99 or ≈0.00, never between. Label accuracy is flat at ≈0.975 throughout, confirming the
field's premise that shortcuts cost nothing in `Acc(Y)`.

### The recovered shortcuts are exactly the predicted ones

Theory says the shortcuts of `y=(c₁+w·c₂) mod 10` are the shifts `u` with `u(1+w) ≡ 0 (mod 10)`.
The empirically recovered `α̂` match that set exactly, with nothing outside it:

| `w` | predicted shift set | shifts actually observed |
|---|---|---|
| 2 | {0} | {0} |
| 1 | {0, 5} | {0, 5} |
| 4 | {0, 2, 4, 6, 8} | {0, 2, 4, 6, 8} — all five |
| 9 | {0,…,9} | {2,4,5,6,7,8,9} — seven of ten, **never 0** |

### Exploratory observation (NOT preregistered — labelled as such)

The fraction of runs that ground tracks `1/|RS|` closely:

| `\|RS\|` | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| fraction grounded | 1.00 | 0.50 | 0.30 | 0.00 |
| `1/\|RS\|` | 1.00 | 0.50 | 0.20 | 0.10 |

This is consistent with SGD selecting **approximately uniformly at random** among the permitted
shortcuts, showing no intrinsic preference for the ground-truth grounding. If it holds up it is a
sharper statement than H1 — symmetry would not merely *permit* shortcuts, it would *set the
probability* of correct grounding. **It is a post-hoc observation on 29 runs in one task family and
is not evidence yet.** It requires its own preregistration and its own runs (planned for E2) before
being treated as anything more than a hypothesis.

### Caveat that materially qualifies this result

**Differential convergence failure.** Exclusion rates were 5, 6, 0, 0 across the four conditions:
the *low*-`|RS|` conditions had many runs that never fit the label (`Acc(Y)` 0.42–0.87 versus ≈0.975
for converged runs), with non-injective `α̂ ∉ RS` — degenerate concept collapse, not shortcuts.

This violates P4's spirit even though gated `Acc(Y)` is flat, and it is the obvious line of attack
on this result: conditions that are supposed to be matched are not equally easy to *optimise*.
The direction of the bias is not obviously favourable or unfavourable — the excluded runs failed
rather than shortcutted — but the asymmetry is real and unexplained.

`experiments/diag_convergence.py` tests candidate recipes on **dev tasks only** to find one that
converges reliably. **If a better frozen recipe exists, E1 will be re-preregistered and re-run
before any of the above is treated as settled.** Until then this result is reported as *provisional*.

## 7. Training experiments still outstanding

E2, E3 and E4 are specified in `EXPERIMENTS.md` and have not been run.
