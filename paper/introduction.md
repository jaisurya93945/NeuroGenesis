# Introduction

Neuro-symbolic predictors promise the best of two worlds: a neural encoder that perceives, and
symbolic knowledge that constrains what the perception may mean. The promise fails in a specific,
well-documented way. A model can satisfy every symbolic constraint while assigning its latent
concepts the *wrong* semantics — predicting the right label for the wrong reasons. These
**reasoning shortcuts** leave label accuracy untouched and destroy interpretability and
out-of-distribution behaviour, which is precisely what the symbolic component was supposed to buy.

Recent work has largely settled how to *analyse* this. Given a constraint set, one can now decide
whether it uniquely determines the concept map, and one can separate the shortcuts a symmetry
permits from the ones optimisation actually selects. What remains open — and what a 2026 survey of
the area names explicitly — is what to *do* about it cheaply: mitigation that is theoretically
grounded and cost-efficient, "requiring minimal human annotation effort while still providing
provable guarantees".

There is an obvious candidate the survey itself gestures at. Because the shortcut set of a
conjunction of tasks is the intersection of their shortcut sets, choosing auxiliary tasks that
collapse that intersection is a **weighted set-cover problem**. It is solvable, it comes with the
usual greedy guarantee, and it requires no per-concept annotation at all. This paper builds that
method, verifies it works, and then asks whether it is worth using.

**It is not, on the instance class we study.** Greedy shortcut-cover selection does everything it
should: it reaches provable identifiability at the smallest possible task budget, matches the
exhaustive optimum, beats selection by mutual information (which is actively misled by
constraint-irrelevant tasks) and beats random selection by a wide margin. And it still loses,
because the baseline it must beat — simply labelling some concepts — grounds the model more cheaply.
The verdict turns on a single quantity nobody has measured: the cost of authoring a symbolic rule
relative to the cost of one concept label.

Along the way the same apparatus yields two positive results. First, the shortcut count `|RS|`,
computable offline before any training, **predicts** whether a model will ground its concepts, and
does so across a heterogeneous population of tasks after controlling for label informativeness and
task size. Second, and more strongly than we expected: among models that actually fit their training
objective, **97.1% adopt a relabelling the oracle named in advance**. The deterministic theory is a
good description of what gradient descent finds; an apparent shortfall turned out to be unfinished
training rather than a gap in the formalism.

## Contributions

1. **A validated shortcut oracle.** Three independent implementations (pruned DFS, ASP, and a naive
   reference) agreeing exactly, plus a test binding the oracle's combinatorial claim to the trainer's
   numerical objective — `alpha` is a shortcut **iff** the corresponding encoder achieves zero loss.
2. **`|RS|` predicts grounding** across 152 tasks from five generator families, robust to controls
   (partial Spearman −0.775), and the oracle predicts *which* wrong grounding is adopted (97.1%).
3. **A negative answer to the survey's open question.** Shortcut-cover selection is correct, optimal
   and beaten on cost by concept supervision. We isolate exactly what would have to be true —
   about annotation economics, or about vocabulary size — for the conclusion to reverse.
4. **Two refinements that do not work**, reported because they were our proposed contributions: a
   graded *margin* adds essentially nothing over binary identifiability, and the apparent
   identifiability/optimisability trade-off does not generalise beyond one task family.
5. **A preregistered artifact.** Every prediction was committed to version control before the data
   existed; the ordering is checkable in the git history. All raw run records ship with the code.
