# ROADMAP.md

## NOW
**E3 — active selection.** With H2, H5, H6 and H7 all gone, the surviving claim is H1/H3: binary
identifiability predicts grounding across heterogeneous task families (partial Spearman −0.775), and
the oracle predicts *which* wrong grounding a converged model adopts (97.1%). Both are solid, and
both are about *analysis*. E3 is the remaining piece that would make this a **design** contribution:
choosing tasks to collapse the shortcut space at minimum annotation cost.

Infrastructure is built and validated (`selection.py`, the D7 divisor-modular pool, exhaustive
optimum for small pools). What it needs is a preregistration and a run with training in the loop.
The honest framing is that E3 now carries the project: if greedy RS-cover does not beat
information-greedy and concept supervision at matched budget, there is no design contribution here.

**Use an absolute convergence criterion, not the relative gate.** E2's `Acc(Y) ≥ 0.95 × best-in-task`
was too permissive and produced a spurious 19% "theory gap" that was really unfinished training
(`RESULTS.md` §7.5). E3/E4 gate on an absolute threshold.

## NEXT
- Preregister E3 (predictions, budget curve, `τ` sensitivity band) before running it.
- Tier-M transfer check for a subset of E2's tasks.
- E4 (continual / RS lock-in).

## LATER
- Implement `RS_ε` (approximate shortcuts) in the oracle. No longer motivated by H7, but still the
  natural generalisation of `measures.margin`, and cheap.
- Paper scaffold, reviewer simulation, literature re-sweep.

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
