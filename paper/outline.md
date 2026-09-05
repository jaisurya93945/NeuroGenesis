# Paper outline

**Working title:** *Provable Identifiability Predicts Symbol Grounding, but Does Not Make It Cheap*

**Claim structure — what the evidence supports, and nothing more:**

1. **Positive, well-evidenced (analysis).** The size of the reasoning-shortcut set, `|RS|`, computed
   offline before any training, predicts whether a neuro-symbolic model grounds its concepts —
   across 152 heterogeneous tasks from five generator families, surviving controls for label
   informativeness, `|Y|`, `k` and support size (partial Spearman −0.775).
2. **Positive, and stronger than expected.** Among models that actually fit the label, **97.1%**
   adopt a relabelling the oracle predicted in advance. The deterministic theory is accurate;
   an apparent 19% shortfall was unfinished training.
3. **Negative, and the paper's centre of gravity (design).** Selecting auxiliary tasks by shortcut
   coverage *works* — it beats information-greedy and random selection and matches the exhaustive
   optimum — and still **loses on cost** to plain concept supervision. The JAIR 2026 survey names
   cost-efficient mitigation as open; on this instance class the answer is no.
4. **Scope of that negative (E5).** Whether it survives at larger concept vocabularies.
5. **Negative refinements.** The graded *margin* adds nothing over binary identifiability (H2);
   identifiability/optimisability trade-off does not generalise (H5).

**Sections**

| file | contents | status |
|---|---|---|
| `introduction.md` | problem, the survey's open question, contributions stated as they are | settled |
| `related_work.md` | RS literature, the two 2026 analysis papers, what is new here | settled |
| `method.md` | formal setting, the oracle and its triple validation, the loss, metrics | settled |
| `experiments.md` | protocol, preregistration discipline, gates, seeds | settled |
| `results.md` | E1–E5, generated from run records | written |
| `discussion.md` | why the negative matters; what would change it | written |
| `limitations.md` | from `LIMITATIONS.md`, incl. the cost model as load-bearing | settled |
| `reproducibility.md` | environment, seeds, commands, the two seeding bugs found | settled |

**Rules for this paper**
- No number is typed by hand. `scripts/make_all_figures.py` and the `analyze_*.py` scripts are the
  only path from run records to reported values.
- Every hypothesis that failed is reported, including the two that were the proposed contributions.
- Novelty language stays out until the `[S]`-marked literature in `LITERATURE.md` is re-read from
  primary sources (arXiv is blocked in the build environment).

**Venue thinking (no claims about acceptance).** The shape — a careful negative on a question a
survey posed, with a validated artifact — fits a workshop or an empirical/reproducibility track
better than a main conference. Decide after E5.
