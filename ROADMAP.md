# ROADMAP.md

## NOW
**A direction decision, and it is the user's to make.** E3 answered the design question and the
answer is no (`RESULTS.md` §8). Five of seven hypotheses have failed, including both proposed
contributions. The machinery is correct and well-tested; the science did not go where it was aimed.

Three defensible options, in the order I would rank them:

**A. Write up the negative result as it stands.** The cost comparison against concept supervision is
precisely the question the JAIR 2026 survey poses, and "no, not on this instance class" is a real
answer supported by 1382 preregistered runs. Add: a Tier-M replication, a wider instance class than
`mod k`, and an honest scope statement. This is the shortest path to something submittable, most
likely as a workshop paper or a short empirical note. Its weakness is that a negative result on one
instance family is thin.

**B. Attack the cost model instead of the method.** E3's verdict rests entirely on "50 concept
labels < 2 authored rules". That is an *empirical* claim about annotation effort that nobody has
measured, and it decides the whole question. Measuring it — or finding a regime where rules are
genuinely cheap (rules reused across many tasks, rules already existing as domain knowledge,
rules covering large concept spaces where labelling scales but rule-writing does not) — could
overturn P4 legitimately. The `k`-scaling angle is the strongest: concept-label cost grows with
`k`, rule cost may not.

**C. Change the question to one the evidence supports.** The most robust result in the project is
that the oracle predicts *which* wrong grounding a converged model adopts (97.1%). That is a
prediction about failure modes, not a mitigation. "Given a task, which shortcut will your model
take?" is answerable, useful for diagnosis, and already largely evidenced.

**I recommend B**, because it is the only option that could restore a design contribution and its
central claim is genuinely untested rather than already refuted. But this is a research-direction
call with no purely technical answer, so it goes to the user.

## NEXT (independent of the choice above)
- Tier-M replication of E3's primary instance — everything after E1 is synthetic-perception only.
- E4 (continual / RS lock-in) remains unrun and is unaffected by E3's outcome.

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
