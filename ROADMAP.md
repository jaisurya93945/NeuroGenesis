# ROADMAP.md

## NOW
**Decide the direction after H2's falsification.** E2 killed the project's proposed refinement:
the margin adds 0.15% of variance over binary identifiability, and the decisive cell failed outright
(`RESULTS.md` §7.2). What survives is stronger but less novel — `log|RS|` predicts grounding across
152 heterogeneous tasks (partial Spearman −0.775), which is a *generalisation* of known results
rather than a new mechanism.

The most promising remaining thread is **H7**, which E2 produced by accident: only 78.4% of
converged non-grounding runs land inside the deterministic RS set, against 100% in E1's narrow
family. One failure in five is a shortcut the theory does not predict. That is a gap in the
*formalism*, not just in the empirics, and it is the kind of thing the field would want to know.

## NEXT
- **H7 preregistration and experiment.** Characterise the out-of-set relabellings: are they
  approximate shortcuts (`RS_ε` for small ε), stochastic/mixture optima, or artefacts of the `α̂`
  mode-recovery estimator? The third possibility must be ruled out first — it is the cheapest
  explanation and would be a measurement bug rather than a finding.
- **Implement `RS_ε`** (approximate shortcuts) in the oracle. `measures.margin` already computes the
  minimum violation mass; the ε-relaxed *set* is the natural next object.

## LATER
- **E3** — active selection. Infrastructure is built and validated (`selection.py`, D7 pool), but its
  motivation weakened: if the binary property is what matters, selecting for it is still sensible,
  though the "margin-aware selection" angle is gone with H2.
- **E4** — continual / RS lock-in.
- Tier-M transfer check for E2's 20 overlap tasks.
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
4. **Do trained NeSy models leave the deterministic RS set?** E2 says yes, ~22% of the time on
   heterogeneous tasks. Now the leading candidate (H7).

## BLOCKING PREREQUISITE FOR SUBMISSION
Re-read all `[S]`-marked primary sources in `LITERATURE.md` from a network that can reach arXiv and
OpenReview. No novelty claim survives without it.
