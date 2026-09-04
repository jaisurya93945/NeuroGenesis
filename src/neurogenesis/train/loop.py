"""Training and evaluation for a NeSy predictor.

Deliberately plain: Adam, cosine decay, no augmentation, no tricks. Augmentation
in particular is excluded on purpose -- it changes the encoder's inductive bias,
which is the very thing competing with symmetry as an explanation for grounding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..data.tuples import TupleDataset
from ..metrics import ConceptMetrics, compute_metrics
from ..models.nesy import NeSyModel
from ..oracle.base import RSResult
from ..tasks import Task


@dataclass
class TrainConfig:
    """Frozen recipe. Tuned only on dev tasks (generator seeds 0-99)."""

    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    lr_final: float = 1e-4
    weight_decay: float = 0.0
    encoder: str = "cnn"
    device: str = "cpu"
    num_threads: int = 1
    log_every: int = 0  # 0 = silent


@dataclass
class TrainResult:
    metrics: dict[str, ConceptMetrics]
    history: list[dict] = field(default_factory=list)
    runtime_s: float = 0.0
    n_params: int = 0


def _batches(n: int, bs: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for i in range(0, n, bs):
        yield order[i : i + bs]


@torch.no_grad()
def evaluate(
    model: NeSyModel,
    ds: TupleDataset,
    k: int,
    rs: RSResult | None = None,
    batch_size: int = 512,
) -> ConceptMetrics:
    """Full-dataset evaluation, returning every concept metric."""
    model.eval()
    device = next(model.parameters()).device
    preds_y, preds_c = [], []
    for i in range(0, len(ds), batch_size):
        x = torch.from_numpy(ds.x[i : i + batch_size]).to(device)
        if x.ndim == 4:  # (B, n_slots, 28, 28) -> add channel
            x = x.unsqueeze(2)
        py, pc = model.predict(x)
        preds_y.append(py.cpu().numpy())
        preds_c.append(pc.cpu().numpy())
    return compute_metrics(
        pred_y=np.concatenate(preds_y),
        true_y=ds.labels,
        pred_c=np.concatenate(preds_c),
        true_c=ds.concepts,
        k=k,
        rs=rs,
    )


def train(
    task: Task,
    encoder: nn.Module,
    train_ds: TupleDataset,
    eval_sets: dict[str, TupleDataset],
    cfg: TrainConfig,
    init_seed: int,
    rs: RSResult | None = None,
) -> TrainResult:
    """Train one NeSy model and evaluate it on every provided split."""
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(init_seed)

    device = torch.device(cfg.device)
    model = NeSyModel(encoder, task).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_final)

    x_all = torch.from_numpy(train_ds.x)
    if x_all.ndim == 4:
        x_all = x_all.unsqueeze(2)
    y_all = torch.from_numpy(train_ds.labels)
    rng = np.random.default_rng(init_seed)

    t0 = time.perf_counter()
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        total, nb = 0.0, 0
        for idx in _batches(len(train_ds), cfg.batch_size, rng):
            xb = x_all[idx].to(device)
            yb = y_all[idx].to(device)
            loss = model.loss(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        sched.step()
        rec = {"epoch": epoch, "train_loss": total / max(nb, 1)}
        if cfg.log_every and (epoch + 1) % cfg.log_every == 0:
            m = evaluate(model, eval_sets["val"], task.space.k, rs)
            rec |= {"val_acc_y": m.acc_y, "val_acc_c": m.acc_c}
            print(
                f"  epoch {epoch + 1:3d}  loss {rec['train_loss']:.4f}  "
                f"val Acc(Y) {m.acc_y:.3f}  Acc(C) {m.acc_c:.3f}"
            )
        history.append(rec)

    runtime = time.perf_counter() - t0
    return TrainResult(
        metrics={name: evaluate(model, ds, task.space.k, rs) for name, ds in eval_sets.items()},
        history=history,
        runtime_s=runtime,
        n_params=sum(p.numel() for p in model.parameters()),
    )
