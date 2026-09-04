"""Concept-quality metrics.

``Acc(Y)`` alone is blind to reasoning shortcuts -- that is the entire premise of
the field -- so the metrics that matter here are about the *concepts*:

- ``acc_c`` / ``f1_c``  -- did the model learn the intended semantics?
- ``collapse``          -- did it merge concepts together?
- ``alpha_hat``         -- *which* relabelling did it actually learn?
- ``rs_membership``     -- is that relabelling one the oracle predicted?

``rs_membership`` is the non-standard one and it is deliberate. The deterministic
RS set is a claim about which solutions the data permits. If trained models
routinely land *outside* it, the deterministic theory under-specifies what SGD
finds (finite data, mixture optima) -- a finding invisible to ``acc_c`` alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .oracle.base import RSResult


@dataclass
class ConceptMetrics:
    """Everything measured about one trained model's concepts."""

    acc_y: float
    acc_c: float
    f1_c: float
    collapse: float
    n_distinct_concepts: int
    alpha_hat: list[int]
    alpha_hat_is_identity: bool
    rs_membership: bool | None
    alpha_hat_coverage: float

    def to_dict(self) -> dict:
        return asdict(self)


def label_accuracy(pred_y: np.ndarray, true_y: np.ndarray) -> float:
    return float((np.asarray(pred_y) == np.asarray(true_y)).mean())


def concept_accuracy(pred_c: np.ndarray, true_c: np.ndarray) -> float:
    """Per-slot concept accuracy, flattened over slots."""
    return float((np.asarray(pred_c).ravel() == np.asarray(true_c).ravel()).mean())


def concept_macro_f1(pred_c: np.ndarray, true_c: np.ndarray, k: int) -> float:
    """Macro-F1 over concepts.

    Reported alongside accuracy because accuracy can look respectable while the
    model collapses rare concepts onto frequent ones; macro-F1 punishes that.
    """
    pred = np.asarray(pred_c).ravel()
    true = np.asarray(true_c).ravel()
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        denom = 2 * tp + fp + fn
        if denom == 0:
            continue
        f1s.append(2 * tp / denom)
    return float(np.mean(f1s)) if f1s else 0.0


def concept_collapse(pred_c: np.ndarray, k: int) -> tuple[float, int]:
    """``Cls(C) = 1 - |distinct predicted concepts| / k``, plus the raw count.

    ``0`` means every concept is used somewhere; higher means the encoder has
    merged concepts. Note a low value is not automatically good: a shortcut can
    activate all concepts while assigning all of them wrongly.
    """
    distinct = int(len(np.unique(np.asarray(pred_c))))
    return 1.0 - distinct / k, distinct


def recover_alpha(pred_c: np.ndarray, true_c: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Recover the relabelling the model implements: ``alpha_hat(c) = mode of predictions``.

    Returns ``(alpha_hat, coverage)`` where ``coverage`` is the fraction of true
    concepts that were actually observed. Unobserved concepts map to themselves,
    which keeps ``alpha_hat`` total; ``coverage < 1`` means ``rs_membership``
    should be read with care, so it is reported rather than hidden.
    """
    pred = np.asarray(pred_c).ravel()
    true = np.asarray(true_c).ravel()
    alpha = np.arange(k, dtype=np.int8)
    seen = 0
    for c in range(k):
        sel = true == c
        if not sel.any():
            continue
        seen += 1
        counts = np.bincount(pred[sel], minlength=k)
        alpha[c] = int(counts.argmax())
    return alpha, seen / k


def compute_metrics(
    pred_y: np.ndarray,
    true_y: np.ndarray,
    pred_c: np.ndarray,
    true_c: np.ndarray,
    k: int,
    rs: RSResult | None = None,
) -> ConceptMetrics:
    """Bundle every concept metric for one evaluation pass."""
    alpha_hat, coverage = recover_alpha(pred_c, true_c, k)
    collapse, distinct = concept_collapse(pred_c, k)
    return ConceptMetrics(
        acc_y=label_accuracy(pred_y, true_y),
        acc_c=concept_accuracy(pred_c, true_c),
        f1_c=concept_macro_f1(pred_c, true_c, k),
        collapse=collapse,
        n_distinct_concepts=distinct,
        alpha_hat=[int(v) for v in alpha_hat],
        alpha_hat_is_identity=bool((alpha_hat == np.arange(k)).all()),
        rs_membership=None if rs is None else rs.contains(alpha_hat),
        alpha_hat_coverage=coverage,
    )


def passes_convergence_gate(acc_y: float, reference: float, ratio: float = 0.95) -> bool:
    """Whether a run fitted its own objective well enough to be interpretable.

    An RS claim about a model that never learned to predict the label is
    meaningless: it has not chosen a shortcut, it has simply failed. ``reference``
    is the best label accuracy achieved in the same condition, so the gate asks
    "did this seed converge like its siblings?" rather than imposing an absolute
    threshold that would differ in meaning across tasks.

    Excluded runs are always reported as an exclusion *rate*, never silently dropped.
    """
    return acc_y >= ratio * reference
