# Method

## 1. Setting

A neuro-symbolic predictor factors through a discrete bottleneck:

```
x = (x_1, ..., x_n)  --phi_theta-->  c in [k]^n  --f-->  y in Y
```

`phi_theta` is a learned concept encoder **shared across slots**; `f` is fixed symbolic knowledge,
given upfront and never learned. Only `y` is supervised, so the concepts are latent and their
semantics is not directly constrained by the training signal.

## 2. Reasoning shortcuts

For a task `T = (f, supp)`, where `supp` is the support of the data distribution over concept
tuples:

```
RS(T) = { alpha : [k] -> [k]  |  for all c in supp,  f(alpha(c)) = f(c) }
```

`T` is **identifiable** iff `RS(T) = {id}`. Two structural facts, both asserted as tests:

- `RS(T)` is a **monoid**: it contains the identity and is closed under composition.
- `RS(T_1 and T_2) = RS(T_1) intersect RS(T_2)`.

The second makes multitask constraint *selection* a weighted set-cover problem, which is the basis
of the design experiments (E3, E5).

`alpha` is **not** required to be a bijection: non-injective `alpha` are concept *collapses*, a
distinct failure mode, so injectivity is an explicit switch rather than an assumption.

### 2.1 The total/partial distinction is load-bearing

`f` is **total** on `[k]^n`; `supp` restricts only where data lives. A neural encoder can emit any
concept tuple, so `f(alpha(c))` always evaluates and "does this shortcut still achieve zero loss?"
has a definite answer. The alternative reading — disqualify `alpha` if it maps a support tuple
off-support — is a different question with different counts.

Because a silent default here would answer the wrong question with plausible-looking numbers,
`mode` and `closure` are **mandatory, un-defaulted arguments** on every oracle entry point.

## 3. The oracle, and why it can be trusted

Three independent implementations, cross-checked:

1. **Pruned DFS** over `[k]^[k]`. A relabelling is refuted as soon as `alpha(0..d)` are fixed, using
   support tuples bucketed by maximum entry. Searches `10^10` for MNIST-Addition in ~1 ms and
   `10^26` at `k = 20` in 36 ms.
2. **ASP via clingo.** Handles what DFS cannot: per-slot relabellings, relational knowledge, and
   margin optimisation (a constrained minimisation, not an enumeration).
3. **A deliberately naive enumerator** (`itertools.product`, no pruning, no solver) for `k <= 5`,
   guarding against the two clever implementations being identically wrong.

Validation, all in CI:

| check | scope | result |
|---|---|---|
| closed form `\|RS\| = gcd(sum w, m)` | `m = 5..12`, `n in {2,3}`, all coprime weights | exact |
| MNIST-Addition full support | — | `RS = {id}` |
| DFS vs ASP, identical map sets | 500 random tasks x {total, partial} x {injective, any} | 2000/2000 agree |
| naive vs both backends | 120 random + structured tasks | exact |

**The test that matters most** binds the oracle to the objective. For a given `alpha`, a tabular
encoder that emits `alpha(ground truth)` exactly is pushed through the *real* loss, and we assert

```
loss == 0 and Acc(Y) == 1   if and only if   alpha in RS(T)
```

including an exhaustive check over **every** map in `[k]^[k]` for small `k`, and on sparse random
supports where the total/partial choice would diverge if it were wrong. Without this, the oracle's
combinatorial claim and the trainer's numerical objective would be different statements.

## 4. Learning

Only `y` is supervised, so the likelihood marginalises over concept tuples consistent with the label:

```
p(y | x) = sum over {c : f(c) = y} of prod_j p(c_j | x_j),      L = -log p(y | x)
```

At `k <= 30`, `n = 2` this sum is computed **exactly** by a masked `logsumexp` — no sampling, no
relaxation, no clamping. Any failure to ground therefore cannot be blamed on approximate inference.
For multiple tasks the per-task losses are averaged, so gradient scale does not grow with the number
of selected tasks (which would confound "more tasks" with "larger learning rate").

## 5. Metrics

`Acc(Y)` is blind to shortcuts by construction — that is the premise of the field — so the metrics
that matter concern concepts: `Acc(C)`, macro-`F1(C)`, collapse `Cls(C)`, the recovered relabelling
`alpha_hat(c) = mode over test examples of argmax p(.|x)`, and **`rs_membership`**: whether
`alpha_hat` lies in the oracle-predicted set. The last is non-standard and is what lets us ask
whether the theory describes what SGD actually finds.
