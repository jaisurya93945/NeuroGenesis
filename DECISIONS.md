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
