# ROADMAP.md

## NOW
**M5 — preregistration, then E1.** Write `paper/preregistration.md` fixing the predicted directions
*before* the confirmatory runs, then execute E1: `w ∈ {2,1,4,9}` (`|RS| ∈ {1,2,5,10}`) × 10 seeds.
**First falsifiable result.**

## NEXT

## LATER
- **M6** — generators: planted-monoid, support-density, and `threadbare` (identifiable but tiny
  margin — the decisive cell for H2). Tier-S renderer + tier-agreement check.
- **M7** — margin oracle via clingo `#minimize`; **E2**.
- **M8** — `selection.py` (greedy cover, exhaustive) + **E3**.
- **M9** — continual strategies + **E4** (RS lock-in).
- **M10** — figures, `reproduce.sh`, paper scaffold, reviewer simulation, literature re-sweep.

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
1. Does the *margin* (not binary identifiability) govern symbol grounding? (H2, if it holds)
2. RS lock-in: hysteresis of shortcuts under sequential constraint arrival. (E4)
3. Minimum identifying supervision support: complexity and approximation guarantees. (theory)
4. Do trained NeSy models leave the deterministic RS set? (if `rs_membership` is routinely low)

## BLOCKING PREREQUISITE FOR SUBMISSION
Re-read all `[S]`-marked primary sources in `LITERATURE.md` from a network that can reach arXiv and
OpenReview. No novelty claim survives without it.
