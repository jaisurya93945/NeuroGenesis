"""Sequential training over a stream of neuro-symbolic tasks.

The distinctive measurement, and the reason this file exists separately from
``loop.py``: after each phase the recovered relabelling ``alpha_hat`` is checked
against the oracle for **RS lock-in** --

    alpha_hat in RS(tasks seen before this phase)  but  NOT in RS(tasks seen so far)

i.e. the model is still implementing a shortcut that the constraints it has now
been shown provably forbid. Joint training can never exhibit this, because it never
occupies the intermediate state. It is the one thing a sequential experiment can
say about shortcuts that a batch experiment cannot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..continual.strategies import EWCState, ReplayBuffer, replay_loss
from ..data.tuples import TupleDataset
from ..metrics import compute_metrics
from ..models.nesy import MultiTaskNeSyModel
from ..oracle import enumerate as en
from ..tasks import Task

ORACLE = dict(mode="shared", closure="total", allow_noninjective=True)


@dataclass
class PhaseResult:
    """What the model looked like after one phase of the stream."""

    phase: int
    task_name: str
    acc_y: float
    acc_c: float
    alpha_hat: list[int]
    rs_count_seen: int
    in_rs_seen: bool
    in_rs_prev: bool
    locked_in: bool
    forgetting: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _labels_matrix(tasks: list[Task], concepts: np.ndarray) -> np.ndarray:
    return np.stack([t.label_of(concepts) for t in tasks], axis=1)


@torch.no_grad()
def _evaluate(model, ds: TupleDataset, tasks: list[Task], k: int, rs_seen, task_idx: int):
    device = next(model.parameters()).device
    xb = torch.from_numpy(ds.x).to(device)
    if xb.ndim == 4:
        xb = xb.unsqueeze(2)
    slot_lp = model.slot_log_probs(xb)
    from ..models.losses import label_log_probs

    pred_y = label_log_probs(slot_lp, model.masks()[task_idx]).argmax(dim=1).cpu().numpy()
    pred_c = slot_lp.argmax(dim=2).cpu().numpy()
    true_y = tasks[task_idx].label_of(ds.concepts)
    return compute_metrics(pred_y, true_y, pred_c, ds.concepts, k, rs=rs_seen)


def train_sequential(
    tasks: list[Task],
    encoder,
    train_ds: TupleDataset,
    eval_ds: TupleDataset,
    cfg,
    seed: int,
    strategy: str = "naive",
    replay_capacity: int = 2000,
    ewc_lambda: float = 100.0,
    replay_weight: float = 1.0,
) -> list[PhaseResult]:
    """Train on ``tasks`` in order, one phase each, and measure RS lock-in.

    ``strategy`` is one of ``naive``, ``replay``, ``ewc``, ``cool``. Every strategy
    sees exactly the same supervision for the current phase; they differ only in
    what they carry forward from earlier ones.
    """
    torch.set_num_threads(cfg.num_threads)
    device = torch.device(cfg.device)
    model = MultiTaskNeSyModel(encoder, tasks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_all = torch.from_numpy(train_ds.x)
    if x_all.ndim == 4:
        x_all = x_all.unsqueeze(2)
    x_all = x_all.to(device)
    labels_all = torch.from_numpy(_labels_matrix(tasks, train_ds.concepts)).long().to(device)

    buf = ReplayBuffer(capacity=replay_capacity)
    ewc = EWCState(lam=ewc_lambda)
    rng = np.random.default_rng(seed)
    results: list[PhaseResult] = []
    first_phase_acc_c: float | None = None

    for phase in range(len(tasks)):
        seen = tasks[: phase + 1]
        rs_seen = en.rs_set(seen, **ORACLE)
        rs_prev = en.rs_set(tasks[:phase], **ORACLE) if phase else None

        for _ in range(cfg.epochs):
            model.train()
            order = rng.permutation(len(train_ds))
            for i in range(0, len(order), cfg.batch_size):
                idx = order[i : i + cfg.batch_size]
                xb, yb = x_all[idx], labels_all[idx]
                # Only the CURRENT task supervises; that is what makes it continual.
                loss = model.loss_subset(xb, yb, [phase])

                if strategy in ("replay", "cool") and phase > 0:
                    aux = replay_loss(
                        model,
                        buf,
                        model.masks(),
                        device,
                        rng,
                        cfg.batch_size,
                        use_pseudo_concepts=(strategy == "cool"),
                    )
                    if aux is not None:
                        loss = loss + replay_weight * aux
                if strategy == "ewc" and phase > 0:
                    loss = loss + ewc.penalty(model)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        m = _evaluate(model, eval_ds, tasks, tasks[0].space.k, rs_seen, phase)
        alpha = np.array(m.alpha_hat, dtype=np.int8)
        in_seen = rs_seen.contains(alpha) and not rs_seen.truncated
        in_prev = bool(rs_prev is not None and not rs_prev.truncated and rs_prev.contains(alpha))
        if phase == 0:
            first_phase_acc_c = m.acc_c

        results.append(
            PhaseResult(
                phase=phase,
                task_name=tasks[phase].name,
                acc_y=m.acc_y,
                acc_c=m.acc_c,
                alpha_hat=m.alpha_hat,
                rs_count_seen=rs_seen.count,
                in_rs_seen=in_seen,
                in_rs_prev=in_prev,
                # The measurement: a shortcut that WAS permitted and now is not.
                locked_in=bool(in_prev and not in_seen),
                forgetting=(
                    None if first_phase_acc_c is None or phase == 0 else first_phase_acc_c - m.acc_c
                ),
                extra={"truncated": rs_seen.truncated, "collapse": m.collapse},
            )
        )

        # carry-forward bookkeeping for the next phase
        if strategy in ("replay", "cool"):
            buf.add(train_ds.x, _labels_matrix(tasks, train_ds.concepts), train_ds.concepts, rng)
            if strategy == "cool":
                buf.set_pseudo_concepts(model, device)
        if strategy == "ewc":
            ewc.consolidate(model, x_all, labels_all)

    return results


def run_stream(
    tasks: list[Task],
    encoder,
    train_ds: TupleDataset,
    eval_ds: TupleDataset,
    cfg,
    seed: int,
    strategy: str,
) -> dict:
    """Convenience wrapper returning a JSON-serialisable record."""
    t0 = time.perf_counter()
    phases = train_sequential(tasks, encoder, train_ds, eval_ds, cfg, seed, strategy)
    return {
        "strategy": strategy,
        "task_order": [t.name for t in tasks],
        "phases": [p.__dict__ for p in phases],
        "runtime_s": time.perf_counter() - t0,
    }
