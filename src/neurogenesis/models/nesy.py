"""The neuro-symbolic predictor: a shared concept encoder plus fixed knowledge."""

from __future__ import annotations

import torch
from torch import nn

from ..tasks import Task
from .losses import build_label_mask, label_log_probs, semantic_nll  # noqa: F401


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


class MultiTaskNeSyModel(nn.Module):
    """One shared encoder supervised by several tasks at once.

    This is the object the selection question is actually about. ``RS(T_1 and ...
    and T_m) = intersection of RS(T_i)`` is a statement about a *single* concept
    map constrained by several pieces of knowledge simultaneously -- so testing
    whether shrinking that intersection improves grounding requires training one
    encoder against all the selected tasks jointly, not training separately and
    combining afterwards.

    Every task must share the concept space. Each contributes its own marginalised
    likelihood over the same latent concepts; the losses are averaged so that the
    total gradient scale does not grow with the number of selected tasks (which
    would otherwise confound "more tasks" with "larger learning rate").
    """

    def __init__(self, encoder: nn.Module, tasks: list[Task]) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("need at least one task")
        k, n = tasks[0].space.k, tasks[0].space.n_slots
        for t in tasks[1:]:
            if t.space.k != k or t.space.n_slots != n:
                raise ValueError("all tasks must share the concept space")
        self.encoder = encoder
        self.tasks = tasks
        self.n_slots = n
        self.k = k
        for i, t in enumerate(tasks):
            self.register_buffer(f"mask_{i}", build_label_mask(t), persistent=False)

    def _mask(self, i: int) -> torch.Tensor:
        return getattr(self, f"mask_{i}")

    def slot_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, n_slots, ...) -> (B, n_slots, k)`` normalised log-probabilities."""
        b = x.shape[0]
        flat = x.reshape(b * self.n_slots, *x.shape[2:])
        logits = self.encoder(flat)
        return torch.log_softmax(logits, dim=-1).reshape(b, self.n_slots, self.k)

    def loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean semantic NLL across tasks. ``labels`` is ``(B, n_tasks)``."""
        slot_lp = self.slot_log_probs(x)
        total = None
        for i in range(len(self.tasks)):
            li = semantic_nll(slot_lp, labels[:, i], self._mask(i)).mean()
            total = li if total is None else total + li
        return total / len(self.tasks)

    def loss_subset(
        self, x: torch.Tensor, labels: torch.Tensor, task_indices: list[int]
    ) -> torch.Tensor:
        """Loss from only the named tasks -- the continual setting's supervision.

        In a sequential stream the model is shown one task at a time, so it must be
        possible to compute a loss over a subset while the masks for every task
        remain registered (rehearsal and evaluation still need them). ``labels`` is
        ``(B, n_tasks)``; only the listed columns are used.
        """
        slot_lp = self.slot_log_probs(x)
        total = None
        for i in task_indices:
            li = semantic_nll(slot_lp, labels[:, i], self._mask(i)).mean()
            total = li if total is None else total + li
        return total / max(1, len(task_indices))

    def masks(self) -> list[torch.Tensor]:
        """All registered label masks, in task order."""
        return [self._mask(i) for i in range(len(self.tasks))]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted label for the **primary** task (index 0), plus concepts."""
        slot_lp = self.slot_log_probs(x)
        primary = label_log_probs(slot_lp, self._mask(0))
        return primary.argmax(dim=1), slot_lp.argmax(dim=2)
