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

## 3. Data integrity (M0) — PASSING

MNIST fetched from `ossci-datasets.s3.amazonaws.com`, all four idx files sha256-verified
(hashes recorded in `src/neurogenesis/data/mnist.py`). Shapes `(60000,28,28)` / `(10000,28,28)`;
per-digit train counts `[5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]`.

## 4. Training experiments

**None run yet.** E1–E4 are specified in `EXPERIMENTS.md` and are not yet executed. No claim about
the relationship between identifiability and empirical grounding is made anywhere in this repo.
