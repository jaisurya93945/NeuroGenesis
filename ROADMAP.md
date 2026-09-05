# ROADMAP.md

## NOW
**M13 — Tier-M replication.** Everything after E1 is synthetic-perception only. Replicate E3's
primary instance and a `k ≤ 10` subset of E2 on MNIST digits. If the rank order disagrees with
Tier S, every Tier-S conclusion is demoted to tier-specific, per `preregistration_e2.md` §3.

## DONE
- E1–E5 complete (~2,600 runs), each preregistered in a data-free commit.
- All paper sections written, including `results.md` and `discussion.md`. Every number comes from
  the `analyze_*.py` scripts.

## DO NOT BUILD YET
Distributed execution; a web dashboard; a general probabilistic-logic engine (exact marginalisation
suffices at `k ≤ 10`, `n ≤ 3`); VLM/LLM concept extractors (no model downloads or API keys here);
mixture-optimum LP feasibility (named v1 scope limit); relational non-functional knowledge.

## ABANDONED
| Direction | Why |
|---|---|
| Original NeuroGenesis continual hypothesis | Already published as COOL (Marconato et al., ICML 2023) |
| `VermiMind` LLM reasoning verification | Crowded (Safe, Typed-CoT, CRV, VeryTrace); no LLM API keys here |
| LLM agent memory conflict resolution | Very crowded in 2026; needs API access |
| Classic CL on CIFAR | Crowded, marginal, and CIFAR is network-blocked |

## POTENTIAL FUTURE PAPERS
Only if the evidence arrives — listed as questions, not plans.
1. ~~Does the *margin* govern symbol grounding?~~ **Answered: no** (E2, H2 falsified).
2. RS lock-in: hysteresis of shortcuts under sequential constraint arrival. (E4)
3. Minimum identifying supervision support: complexity and approximation guarantees. (theory)
4. ~~Do trained NeSy models leave the deterministic RS set?~~ **Answered: essentially no.** 97.1%
   membership among converged models (H7 rejected).

## BLOCKING PREREQUISITE FOR SUBMISSION
Re-read all `[S]`-marked primary sources in `LITERATURE.md` from a network that can reach arXiv and
OpenReview. No novelty claim survives without it.
