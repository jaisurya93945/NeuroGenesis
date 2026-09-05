# LITERATURE.md

**Provenance caveat, stated once and meant literally.** This environment blocks arXiv, OpenReview,
ACM, Semantic Scholar and HuggingFace. Everything below was gathered through web *search summaries*,
not by reading the PDFs. Entries are therefore recorded at the fidelity actually obtained, and
marked accordingly:

- **[S]** — known from search-result summaries only. Titles, venues, authors and headline claims
  should be re-verified against the primary source before any of it enters a paper.
- **[V]** — independently verified in this repo (re-derived or reproduced by our own code).

A full re-read of primary sources is a **blocking prerequisite for paper submission** and is
tracked in `ROADMAP.md`.

## Status of this review: INCOMPLETE

Re-verified on 2026-09-05: `arxiv.org` and `openreview.net` are **still unreachable** from this
environment (both return no response through the egress proxy). Every entry below therefore remains
`[S]` — assembled from search-result summaries, never read as a primary source.

**This literature review is not complete, and no claim in this project may rest on it.** It is
sufficient to establish that the area is active and to locate the open problem being answered; it is
**not** sufficient to support any statement of novelty. Re-reading the primary sources from a network
that can reach arXiv is a blocking prerequisite before submission language is written, and that
requirement is recorded in `ROADMAP.md`.

## Core matrix

| Work | Venue / year | Problem | Contribution | Relation to us |
|---|---|---|---|---|
| Marconato et al., *Neuro-Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal* **[S]** | ICML 2023 | Continual NeSy | Shows prior knowledge alone does **not** prevent RSs; proposes COOL (concept-level rehearsal) | **Pre-empts the original NeuroGenesis hypothesis.** Becomes our E4 baseline, not our contribution |
| Marconato et al., *Not All Neuro-Symbolic Concepts Are Created Equal* **[S]** | NeurIPS 2023 | RS analysis | Characterises RSs; mitigation strategies | Defines the problem we build on |
| Marconato et al., *BEARS Make NeSy Models Aware of their Reasoning Shortcuts* **[S]** | 2024 | RS awareness | Calibration/ensembling so models know when they may be shortcutting | Complementary: detection, not prevention |
| Bortolotti et al., *rsbench: A NeSy Benchmark Suite for Concept Quality and Reasoning Shortcuts* **[S]** | NeurIPS 2024 D&B | Benchmarking | Tasks + metrics `Acc(C)`, `F1(C)`, `Cls(C)`; `countrss` RS counter | Metric definitions we reimplement. **Code not cloneable here** |
| Bortolotti et al., *Shortcuts and Identifiability in Concept-based Models from a NeSy Lens* **[S]** | NeurIPS 2025 | CBM identifiability | Conditions for identifying concepts + inference layer; finds existing methods "often fail to meet these conditions in practice" | **Closest adjacent work.** Must be cited and carefully distinguished — see below |
| *Constraint-Based Analysis of Reasoning Shortcuts in NeSy Learning* (arXiv 2604.23377) **[S]** | 2026 | RS decidability | RSs as a CSP; a *discrimination property* (necessary, not sufficient); **sound and complete ASP algorithm** deciding whether a constraint set uniquely determines the concept map | **Solves the analysis question we build the design question on top of** |
| *Reasoning Shortcuts and Value Symmetries* (arXiv 2608.10420) **[S]** | 2026 | Theory vs practice | Hierarchy of symmetry readings; separates what symmetry *permits* from what optimisation *selects* | **Partial collision with our theory→practice measurement.** Constrains our claims |
| Marconato et al., *Symbol Grounding in NeSy AI: A Gentle Introduction to Reasoning Shortcuts* **[S]** | JAIR 2026 (arXiv 2510.14538) | Survey | Taxonomy of mitigations; identifiability + statistical-learning views | **Source of our stated gap** (below) |
| *Right for the Right Reasons: Prototypical NeSy AI* (arXiv 2510.25497) **[S]** | 2025/26 | RS mitigation | Prototype-augmented architecture | Alternative mitigation; a possible extra baseline |
| van Krieken et al., *On the Independence Assumption in Neurosymbolic Learning* **[S]** | ICML 2024 | NeSy modelling | Independence assumption hinders learning and uncertainty modelling | Relevant to our exact-marginalisation loss |

## The stated gap

The JAIR 2026 survey names as open:

1. mitigation that is **theoretically grounded *and* cost-efficient**, "requiring minimal human
   annotation effort while still providing provable guarantees";
2. "it remains unclear **how to construct multitask learning approaches** effectively to remove
   reasoning shortcuts."

Our design question targets (2) directly, and (1) is the cost story.

## Honest positioning

The *analysis* side is solved: 2604.23377 decides identifiability; 2608.10420 measures the
permits-vs-selects gap. So **we do not claim the theory→practice measurement as novel** — it is
partly covered. What appears to remain open is the **design/intervention** question: given a budget,
*which* tasks should you acquire, and in what order, to collapse the RS space at minimum cost.

NeurIPS 2025 (Bortolotti et al.) already reports that existing methods often fail to *meet*
identifiability conditions. Our distinct angle is to **construct** supports that provably meet them,
and then test whether meeting them suffices. That distinction is thin enough that it must be
re-checked against the primary sources before any novelty claim is made — hence the
"potential research gap" label in `HYPOTHESES.md`.

## Not yet investigated
Model editing; test-time adaptation; the concept-bottleneck-model literature beyond the NeSy lens;
identifiability results from nonlinear ICA / disentanglement, which likely bear on H2.
