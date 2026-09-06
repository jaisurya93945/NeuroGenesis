"""Strategies for learning a shared concept encoder from tasks arriving in sequence.

The question E4 asks is not "how much accuracy is forgotten" -- that is well
studied -- but something the RS oracle makes newly measurable:

    when a later task **provably forbids** the shortcut a model already adopted,
    does the model actually give it up?

Formally, ``RS(T1 and T2)`` is a subset of ``RS(T1)``. A model trained on ``T1``
alone may settle on some ``alpha`` in ``RS(T1)``. If ``alpha`` is not in
``RS(T1 and T2)``, then after ``T2`` arrives that relabelling is no longer
consistent with the constraints the model has been shown. A model still
implementing it is exhibiting **RS lock-in**: hysteresis in shortcut space.

Joint training cannot show this -- it never passes through the intermediate state.
That is why it needs a sequential experiment, and it is the distinctive measurement
in E4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..models.losses import concept_supervision_loss, semantic_nll


@dataclass
class ReplayBuffer:
    """Stored inputs plus the task labels they carried, for rehearsal.

    Holds raw inputs rather than gradients or activations, which is the honest
    baseline: it assumes the old *data* is still available. Concept rehearsal
    (below) is the interesting variant because it stores the model's own beliefs.
    """

    capacity: int = 2000
    x: np.ndarray | None = None
    labels: np.ndarray | None = None  # (N, n_tasks_so_far)
    concepts: np.ndarray | None = None
    pseudo_concepts: np.ndarray | None = None

    def add(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        concepts: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        n = min(self.capacity, len(x))
        idx = rng.choice(len(x), size=n, replace=False)
        self.x = x[idx]
        self.labels = labels[idx]
        self.concepts = concepts[idx]

    def set_pseudo_concepts(self, model: nn.Module, device: torch.device) -> None:
        """Record what the *current* model believes the concepts are.

        This is the COOL-style ingredient: the buffer carries the model's own
        concept assignments, so later phases are pulled toward the grounding the
        model already had rather than toward ground truth it was never given.
        """
        if self.x is None:
            return
        with torch.no_grad():
            xb = torch.from_numpy(self.x).to(device)
            if xb.ndim == 4:
                xb = xb.unsqueeze(2)
            self.pseudo_concepts = model.slot_log_probs(xb).argmax(dim=2).cpu().numpy()

    def __len__(self) -> int:
        return 0 if self.x is None else len(self.x)


@dataclass
class EWCState:
    """Diagonal Fisher penalty anchoring parameters to a previous phase."""

    lam: float = 100.0
    fisher: dict[str, torch.Tensor] = field(default_factory=dict)
    anchor: dict[str, torch.Tensor] = field(default_factory=dict)

    def consolidate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        labels: torch.Tensor,
        n_batches: int = 8,
        batch_size: int = 128,
    ) -> None:
        """Estimate the diagonal Fisher at the current parameters."""
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        n = min(len(x), n_batches * batch_size)
        for i in range(0, n, batch_size):
            model.zero_grad(set_to_none=True)
            loss = model.loss(x[i : i + batch_size], labels[i : i + batch_size])
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
        denom = max(1, (n + batch_size - 1) // batch_size)
        self.fisher = {k: v / denom for k, v in fisher.items()}
        self.anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros((), device=next(model.parameters()).device)
        total = None
        for name, p in model.named_parameters():
            if name in self.fisher:
                term = (self.fisher[name] * (p - self.anchor[name]) ** 2).sum()
                total = term if total is None else total + term
        return self.lam * (total if total is not None else 0.0)


def replay_loss(
    model: nn.Module,
    buf: ReplayBuffer,
    masks: list[torch.Tensor],
    device: torch.device,
    rng: np.random.Generator,
    batch_size: int,
    use_pseudo_concepts: bool,
) -> torch.Tensor | None:
    """Rehearsal term for one step: old task labels, or the model's old concepts."""
    if len(buf) == 0:
        return None
    idx = rng.choice(len(buf), size=min(batch_size, len(buf)), replace=False)
    xb = torch.from_numpy(buf.x[idx]).to(device)
    if xb.ndim == 4:
        xb = xb.unsqueeze(2)
    slot_lp = model.slot_log_probs(xb)

    if use_pseudo_concepts:
        if buf.pseudo_concepts is None:
            return None
        target = torch.from_numpy(buf.pseudo_concepts[idx]).long().to(device)
        return concept_supervision_loss(slot_lp, target).mean()

    lab = torch.from_numpy(buf.labels[idx]).long().to(device)
    total = None
    for j, mask in enumerate(masks[: lab.shape[1]]):
        li = semantic_nll(slot_lp, lab[:, j], mask).mean()
        total = li if total is None else total + li
    return None if total is None else total / max(1, lab.shape[1])


STRATEGIES = ("naive", "replay", "ewc", "cool")
