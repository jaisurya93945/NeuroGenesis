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

## 2. Data integrity (M0) — PASSING

MNIST fetched from `ossci-datasets.s3.amazonaws.com`, all four idx files sha256-verified
(hashes recorded in `src/neurogenesis/data/mnist.py`). Shapes `(60000,28,28)` / `(10000,28,28)`;
per-digit train counts `[5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]`.

## 3. Training experiments

**None run yet.** E1–E4 are specified in `EXPERIMENTS.md` and are not yet executed. No claim about
the relationship between identifiability and empirical grounding is made anywhere in this repo.
