# Related work

**Provenance warning, and it is not boilerplate.** The build environment blocks arXiv, OpenReview,
ACM and Semantic Scholar. Every entry below was assembled from search-result summaries, not primary
PDFs, and is marked `[S]` in `LITERATURE.md`. **Re-reading the primary sources is a blocking
prerequisite before any novelty language enters this paper.** Titles, venues and headline claims
should be treated as unverified until then.

## Reasoning shortcuts

The failure mode — a neuro-symbolic predictor satisfying its symbolic constraints while assigning
concepts the wrong semantics — was characterised by Marconato et al. (*Not All Neuro-Symbolic
Concepts Are Created Equal*, NeurIPS 2023) `[S]`, and given a benchmark suite by Bortolotti et al.
(*rsbench*, NeurIPS 2024 D&B) `[S]`, which also defines the concept-quality metrics reused here
(`Acc(C)`, `F1(C)`, collapse). BEARS `[S]` makes models aware of their own shortcuts through
calibration — detection rather than prevention.

Two 2026 papers settle the **analysis** side and are the direct antecedents of this work.
*Constraint-Based Analysis of Reasoning Shortcuts* `[S]` formalises shortcuts as a constraint
satisfaction problem and gives a sound-and-complete ASP procedure deciding whether a constraint set
uniquely determines the concept map — the decision procedure our oracle reimplements and validates
independently. *Reasoning Shortcuts and Value Symmetries* `[S]` separates what symmetry *permits*
from what optimisation *selects*, which is the same distinction our `rs_membership` metric measures
empirically.

**We therefore do not claim the theory→practice measurement as novel.** What appears to remain open
is the *design* question built on top of it.

## The open problem this paper answers

The JAIR 2026 survey *Symbol Grounding in Neuro-Symbolic AI* `[S]` names as open: mitigation that is
"theoretically grounded **and** cost-efficient, requiring minimal human annotation effort while
still providing provable guarantees", and states that "it remains unclear how to construct multitask
learning approaches effectively to remove reasoning shortcuts".

This paper takes that literally: it constructs the multitask approach, verifies it collapses the
shortcut space optimally, and then measures its cost against the annotation baseline the survey says
the field relies on. **The answer is negative on the instance class studied**, which is a direct
response to the stated question rather than a sidestep of it.

## Continual neuro-symbolic learning

Marconato et al. (*Neuro-Symbolic Continual Learning*, ICML 2023) `[S]` showed prior knowledge alone
does not prevent shortcuts in a task stream, and proposed COOL, a concept-level rehearsal strategy.
That work pre-empts the hypothesis this project originally set out to test, which is recorded in
`DECISIONS.md` D1 as the reason for the pivot. We reimplement a COOL-style concept-rehearsal arm
from the paper description (its code is unreachable from this environment) as one baseline in E4,
and add a measurement that strategy work has not made: whether a model *retains a shortcut its
current constraint set provably forbids*.

## Positioning, stated conservatively

Adjacent work (NeurIPS 2025, *Shortcuts and Identifiability in Concept-based Models*) `[S]` already
reports that existing methods often fail to **meet** identifiability conditions in practice. Our
distinct angle is to **construct** task sets that provably meet them and then ask whether meeting
them is worth what it costs. That distinction is thin enough that it must be re-checked against
primary sources before any claim of novelty is made — hence the "potential research gap" language
maintained throughout `HYPOTHESES.md`.
