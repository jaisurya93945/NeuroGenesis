# DECISIONS.md — research decision records

## D1 — Pivot from continual learning to reasoning shortcuts

**Decision.** Target reasoning shortcuts / symbol grounding, not the originally-proposed
"separate neural representations from symbolic knowledge to improve stability-plasticity".

**Alternatives.** (a) The original continual NeSy hypothesis. (b) LLM agent memory conflict
resolution. (c) CoT verification (the repo's stale `VermiMind` README). (d) Classic CL on CIFAR.

**Evidence.** (a) is essentially Marconato et al., ICML 2023 (COOL) — already published.
(b) is crowded with 2026 work (MemConflict, TOKI, Supersede, STALE, Memora, MemoryArena) *and*
needs LLM API keys, which this environment lacks. (c) is crowded (Safe, Typed-CoT, CRV, VeryTrace)
and equally infeasible without API access. (d) is crowded, marginal, and CIFAR is network-blocked.

**Chosen.** Reasoning shortcuts, design/intervention framing.

**Why.** It is the only direction where (i) the literature names an explicit open problem, (ii) the
benchmarks are small enough to run honestly on a CPU box, and (iii) the formal core is clean enough
to give exact known-answer tests. The environment and the literature select the same answer.

**Trade-offs.** Narrower audience than LLM work. Requires reimplementing rsbench/NeSy-CL/BEARS
because no external repo is cloneable.

**Validation.** E1 falsifies or supports H1 at ~1.5 h of compute.

## D2 — `f` is total; `support` restricts only the data distribution

**Decision.** `label_table` is defined on all of `[k]^n`; `support` marks where data lives.
A `partial` closure exists but is not the default.

**Why.** A neural encoder can emit any concept tuple, so `f(alpha(c))` must always evaluate for
"does this shortcut still achieve zero loss?" to have an answer.

**Trade-off.** The two readings give different counts. Mitigated by making `closure` a
**mandatory, un-defaulted** argument, so no caller can be ambiguous by accident.

**Validation.** `test_predicted_rs_achieves_zero_loss` (M3) binds the oracle's combinatorial claim
to the loss the trainer actually optimises. If this choice is wrong, that test fails.

## D3 — Three independent RS implementations

**Decision.** Pruned-DFS enumeration, clingo ASP, and closed-form analytic — cross-checked.

**Why.** The oracle is the single point of failure: every downstream number inherits its semantics,
and a definitional error produces plausible-looking numbers that answer the wrong question.
Differential testing between independent implementations is the only cheap defence.

**Validation.** Analytic `gcd` grid + DFS agreement is **already passing** (`RESULTS.md` §1);
DFS↔ASP differential test lands at M2.

## D4 — Dataclasses + YAML, not Hydra

**Decision.** Frozen dataclasses with a small YAML loader and a `config_hash`.

**Why.** The sweep axis is *generated tasks* (programmatic), so Hydra's override grammar buys
nothing; its per-run CWD changes fight an append-only results store, and `DictConfig` does not hash
stably into a run ID. We need `from_yaml`, `config_hash`, and resume-by-hash — about 80 lines.
Also satisfies the master prompt's "avoid framework lock-in".

## D5 — No torchvision; checksum-pinned raw MNIST

**Decision.** Fetch the four idx files directly from two mirrors, verify sha256, parse in-repo.

**Why.** `download.pytorch.org` is blocked here, and pinning raw bytes makes the data dependency
explicit rather than delegated to a library whose mirrors have changed repeatedly.

## D6 — The mod-10 family is the primary evidence

**Decision.** `y = (c1 + w*c2) mod 10` for `w = 1..9` is E1, not a restricted-support variant.

**Why.** It varies `|RS|` over `{1,2,5,10}` while holding support, `|Y|`, label marginal, `I(C;Y)`
and perceptual difficulty *exactly* constant — the main confound is eliminated by construction.
Every shortcut is a cyclic shift, so a shortcut has concept accuracy exactly 0: the outcome is
bimodal, giving a large effect size and high power at 10 seeds.

**Trade-off.** Modular arithmetic is less "natural" than plain addition. Mitigated by including
plain MNIST-Addition and a restricted-support variant as anchors to the published literature.

## D7 — the auxiliary-task pool had to be redesigned to make E3 non-vacuous

**Decision.** E3 uses `divisor_modular_pool`, whose candidates are individually
insufficient, rather than the generic predicate pool first written.

**Evidence.** With the generic pool, *every* method — greedy RS-cover, information-greedy, and
**uniform random** — made `y = (c₁+9c₂) mod 10` identifiable with a **single** auxiliary task.
A predicate such as `c₀ mod 3` is invariant under no cyclic shift of a 10-element space, so it
annihilates all nine non-identity shortcuts at once. Every method looks equally good and the
experiment measures nothing. This is the "everything works, so nothing is learned" failure mode
flagged as Risk #2 in the plan.

**Why the replacement is principled, not rigged.** `c₀ mod q` is invariant under shift `u` exactly
when `q | u`. Restricting to moduli that **divide** `k` gives every candidate a genuine partial
invariance — each kills some shifts and spares others — so reaching identifiability requires a
*combination*. That is a real weighted set-cover instance, which is the structure E3 is supposed to
be about. The pool also contains shift-invariant distractors (`(c₀−c₁) mod k`), which are perfectly
natural auxiliary predicates that happen to eliminate nothing.

**Disclosure, because it matters.** This pool was designed **after** observing that the generic one
was trivial. That is legitimate experimental *design* — no E3 outcome had been measured, and the
change makes the comparison possible rather than favourable — but it is design-after-observation
and is recorded as such. E3's confirmatory runs are preregistered separately, against this pool,
before they are executed.

**Observed on the design pool (not an E3 result — no training involved, oracle only):** greedy
reaches identifiability at budget 2 and matches the exhaustive optimum; information-greedy fails at
budget 3, spending its budget on the shift-invariant distractors, which carry high label entropy and
zero shortcut coverage. That separation is what E3 will test with training in the loop.

**Trade-off.** A hand-designed pool is less "natural" than a scraped one. Mitigated by
`individually_insufficient()`, a generic filter that turns *any* pool into one where selection
matters, so the finding does not depend on this particular hand-built set.
