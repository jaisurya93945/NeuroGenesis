# RESEARCH.md — formal definitions and methodology

## 1. Setting

A neuro-symbolic (NeSy) predictor factors through a discrete bottleneck:

```
x = (x_1, ..., x_n)  --phi_theta-->  c = (c_1, ..., c_n) in [k]^n  --f-->  y in Y
```

`phi_theta` is a learned concept encoder, **shared across slots**; `f` is fixed symbolic knowledge,
given upfront and never learned. Only `y` is supervised. The scientific question is whether the
learned `c` carries its intended semantics.

## 2. Reasoning shortcuts

For a task `T = (f, supp)` where `supp` is the support of the data distribution over concept tuples:

```
RS(T) = { alpha : [k] -> [k]  |  for all c in supp,  f(alpha(c_1), ..., alpha(c_n)) = f(c_1, ..., c_n) }
```

`T` is **identifiable** iff `RS(T) = {id}`.

Two structural facts, both asserted as tests in `tests/test_oracle_analytic.py`:

1. **`RS(T)` is a monoid** — contains `id`, closed under composition.
2. **`RS(T_1 and T_2) = RS(T_1) ∩ RS(T_2)`.**

Fact 2 is what makes multitask constraint *selection* a weighted set-cover problem over "which
non-identity maps does task `j` eliminate", and it is the formal basis of `neurogenesis.selection`.

`alpha` is **not** required to be a bijection. Non-injective `alpha` are concept *collapses* — a
distinct and separately interesting failure mode — so `allow_noninjective` is an explicit switch.

### 2.1 The total/partial distinction (load-bearing)

`f` is **total** on `[k]^n`; `supp` restricts only where the data lives. A neural encoder can emit
any concept tuple, so `f(alpha(c))` always evaluates and "does this shortcut encoder still achieve
zero loss?" has a definite answer. The alternative `partial` reading — disqualify `alpha` if it maps
a support tuple off-support — is a different, defensible question with different counts.

Because getting this wrong would silently answer the wrong question with plausible-looking numbers,
`mode` and `closure` are **mandatory, un-defaulted keyword arguments** on every oracle entry point.

## 3. The graded refinement: margin

Binary identifiability is coarse. A task can be identifiable only because of a handful of very
low-probability support tuples. Define

```
margin(T) = min over alpha != id of  Pr_{c ~ D} [ f(alpha(c)) != f(c) ]
```

— the evidence mass ruling out the *cheapest-to-satisfy* wrong map. This is the natural
finite-data quantity, generalises to approximate shortcuts `RS_eps(T) = { alpha : Pr[violation] <= eps }`,
and is computed exactly by clingo `#minimize`. **H2 is that `margin` predicts grounding better than
the binary property**, and this is the most likely actual contribution.

## 4. Closed form for the modular-linear family

For `y = (sum_i w_i c_i) mod m` on full support, write `d(x) = alpha(x) - x (mod m)`. Label
preservation gives, for all `c`:

```
sum_i w_i d(c_i) = 0   (mod m)
```

Varying one coordinate with the others fixed forces `w_i d(.)` constant; when `gcd(w_i, m) = 1` that
makes `d` a constant `u`. Substituting back: `u * sum_i w_i = 0 (mod m)`, which has exactly
`gcd(sum_i w_i, m)` solutions in `Z_m`. Therefore

```
|RS| = gcd(sum_i w_i, m)      (full support, every w_i coprime to m)
```

and **every shortcut is a cyclic shift** `alpha(x) = x + u`. A non-zero shift maps every concept to
a wrong one, so a model landing on a shortcut has concept accuracy exactly **0**. The outcome is
therefore bimodal rather than noisy — which is what makes the design statistically sharp.

*Status: derived here, and verified against the oracle across the full `(m, n, w)` grid for
`m = 5..12`, `n in {2,3}` — see `RESULTS.md` §1.*

## 5. Why the mod-10 family is the headline design

For `y = (c_1 + w c_2) mod 10`, `w = 1..9`: `|RS| = gcd(1 + w, 10)` takes values `{1, 2, 5, 10}`.
Across the family, **support, `|Y|`, the label marginal, `I(C;Y) = log 10`, perceptual difficulty
and architecture are all identical.** Only `|RS|` varies.

This matters because the obvious confound — that `|RS|` merely co-varies with how informative the
label is — is eliminated *by construction* rather than by statistical adjustment. The wide
correlational sweep (E2) is corroborative; this family is the primary evidence.

## 6. Metrics

| Metric | Definition |
|---|---|
| `Acc(Y)` | label accuracy on held-out tuples |
| `Acc(C)` | per-slot concept accuracy against ground-truth concepts |
| `F1(C)` | macro-F1 over concepts, robust to collapse onto frequent concepts |
| `Cls(C)` | concept collapse: `1 - |{predicted concepts}| / k` |
| `alpha_hat` | recovered map, `alpha_hat(c) = mode over test x of argmax p_theta(. \| x)` |
| `rs_membership` | whether `alpha_hat in RS(T)` — tests whether the *theory* covers what SGD finds |

`rs_membership` is not a standard metric and is one of this project's cheap novel measurements. If
trained models routinely land outside the deterministic RS set, the deterministic theory
under-specifies reality (finite data, mixture optima). That is invisible to `Acc(C)` alone and is a
publishable finding in its own right.

## 7. Evaluation discipline

- **Convergence gate.** A run enters RS analysis only if `Acc(Y)_test >= 0.95 * best achievable`.
  An RS claim about a model that never fit the label is meaningless. Exclusion rates are reported.
- **Leakage.** Task-generator seeds `0–99` are *dev* — the only tasks on which any hyperparameter
  may be tuned. Seeds `1000+` are *confirmatory*, run once with the frozen recipe. The runner
  raises if a confirmatory seed is used in tuning mode.
- **Splits.** MNIST images `0–49999` train, `50000–59999` val, `60000–69999` (the test file) test.
  No image appears in two splits.
- **Statistics.** Estimation, not significance: bootstrap 95% CIs, effect sizes, per-seed scatter
  always overplotted. No significance stars. At 4 conditions × 10 seeds, `p < 0.001` is trivially
  purchasable and the paper will say so. Only the four pre-registered tests are Holm-corrected.
