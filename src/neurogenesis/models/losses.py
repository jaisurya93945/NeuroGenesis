"""The neuro-symbolic loss: exact marginalisation over the concept bottleneck.

Only the label ``y`` is supervised. The concepts are latent, so the likelihood of
an example marginalises over every concept tuple consistent with the observed
label under the fixed knowledge ``f``::

    p(y | x) = sum over { c : f(c) = y } of prod_j p(c_j | x_j)

    L = -log p(y | x)

At ``k <= 10`` and ``n <= 3`` the concept grid has at most 1000 cells, so this sum
is computed *exactly* by a masked ``logsumexp``. No probabilistic-logic engine,
no sampling, no relaxation -- which removes a whole class of confound from the
experiments, since any failure to ground concepts cannot be blamed on an
approximate inference scheme.

Everything is in log space; the mask contributes ``0`` or ``-inf`` additively
rather than multiplying probabilities by ``0``, so there is no ``clamp(1e-12)``
anywhere and gradients stay exact.
"""

from __future__ import annotations

import numpy as np
import torch

from ..tasks import Task

NEG_INF = float("-inf")


def build_label_mask(task: Task, device: torch.device | str = "cpu") -> torch.Tensor:
    """``(n_labels, k**n)`` additive log-mask: ``0`` where ``f(c) = y``, ``-inf`` elsewhere.

    Precomputed once per task and reused for every batch.
    """
    dense = task.dense_mask()  # (k,)*n + (n_labels,)
    flat = dense.reshape(-1, task.n_labels).T  # (n_labels, k**n)
    mask = np.where(flat, 0.0, NEG_INF).astype(np.float32)
    return torch.from_numpy(mask).to(device)


def joint_log_probs(slot_log_probs: torch.Tensor) -> torch.Tensor:
    """Combine per-slot log-probs into the joint over concept tuples.

    Args:
        slot_log_probs: ``(B, n_slots, k)`` log-probabilities, already normalised.

    Returns:
        ``(B, k**n_slots)`` joint log-probabilities, in row-major tuple order --
        the same order as ``Task.all_tuples()`` and ``Task.dense_mask()``.

    The independence across slots is the standard NeSy factorisation. It is an
    assumption, and a known-contested one (see LITERATURE.md, van Krieken et al.),
    so it is stated here rather than buried.
    """
    b, n, k = slot_log_probs.shape
    out = slot_log_probs[:, 0, :]
    for j in range(1, n):
        out = out.unsqueeze(-1) + slot_log_probs[:, j, :].view(b, *([1] * j), k)
        out = out.reshape(b, -1)
    return out.reshape(b, k**n)


def semantic_nll(
    slot_log_probs: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    """``-log p(y | x)`` marginalising over concept tuples. Returns ``(B,)``."""
    joint = joint_log_probs(slot_log_probs)  # (B, k**n)
    scores = joint + label_mask[labels]  # (B, k**n), -inf off-label
    return -torch.logsumexp(scores, dim=1)


def label_log_probs(
    slot_log_probs: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    """``log p(y | x)`` for every label. Returns ``(B, n_labels)``."""
    joint = joint_log_probs(slot_log_probs)  # (B, K)
    return torch.logsumexp(joint.unsqueeze(1) + label_mask.unsqueeze(0), dim=2)


def concept_supervision_loss(
    slot_log_probs: torch.Tensor,
    concepts: torch.Tensor,
) -> torch.Tensor:
    """Plain cross-entropy on ground-truth concepts -- the expensive baseline.

    Args:
        slot_log_probs: ``(B, n_slots, k)``.
        concepts: ``(B, n_slots)`` ground-truth concept indices.
    """
    return -slot_log_probs.gather(2, concepts.unsqueeze(-1)).squeeze(-1).mean(dim=1)
