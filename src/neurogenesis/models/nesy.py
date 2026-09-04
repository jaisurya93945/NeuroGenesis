"""The neuro-symbolic predictor: a shared concept encoder plus fixed knowledge."""

from __future__ import annotations

import torch
from torch import nn

from ..tasks import Task
from .losses import build_label_mask, label_log_probs, semantic_nll


class NeSyModel(nn.Module):
    """Shared encoder applied to every slot, then exact marginalisation over ``f``.

    The encoder is *shared* across slots, which is what makes a single relabelling
    ``alpha: [k] -> [k]`` the right object for the oracle to reason about. Per-slot
    encoders correspond to ``mode="per_slot"`` and a much larger shortcut space.
    """

    def __init__(self, encoder: nn.Module, task: Task) -> None:
        super().__init__()
        self.encoder = encoder
        self.task = task
        self.n_slots = task.space.n_slots
        self.k = task.space.k
        self.register_buffer("label_mask", build_label_mask(task), persistent=False)

    def slot_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, n_slots, ...) -> (B, n_slots, k)`` normalised log-probabilities.

        The slot axis is folded into the batch so the shared encoder sees one
        stream of inputs -- this is what "shared" means operationally.
        """
        b = x.shape[0]
        flat = x.reshape(b * self.n_slots, *x.shape[2:])
        logits = self.encoder(flat)
        return torch.log_softmax(logits, dim=-1).reshape(b, self.n_slots, self.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``log p(y | x)`` for every label, shape ``(B, n_labels)``."""
        return label_log_probs(self.slot_log_probs(x), self.label_mask)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean semantic negative log-likelihood over the batch."""
        return semantic_nll(self.slot_log_probs(x), y, self.label_mask).mean()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(predicted_label, predicted_concepts)``."""
        slot_lp = self.slot_log_probs(x)
        return self(x).argmax(dim=1), slot_lp.argmax(dim=2)


class TabularEncoder(nn.Module):
    """A hand-set encoder that emits ``alpha(c)`` from a concept index.

    Not for training -- this exists so a *specific* relabelling can be realised
    exactly and pushed through the real loss. It is the instrument that turns the
    oracle's combinatorial claim into a testable statement about the objective
    (see ``tests/test_oracle_vs_loss.py``).
    """

    def __init__(self, alpha: torch.Tensor, k: int, logit_scale: float = 30.0) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha.long())
        self.k = k
        self.logit_scale = logit_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` holds ground-truth concept indices; emit near-one-hot ``alpha(x)``."""
        idx = x.reshape(-1).long()
        out = torch.zeros(idx.shape[0], self.k, device=x.device)
        out[torch.arange(idx.shape[0], device=x.device), self.alpha[idx]] = self.logit_scale
        return out
