#!/usr/bin/env python
"""E4 -- RS lock-in under sequential task arrival.

Preregistered in ``paper/preregistration_e4.md`` before any run.

Usage:
    python experiments/e4_continual.py [--workers 4] [--seeds 8]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from neurogenesis.continual.strategies import STRATEGIES
from neurogenesis.data.tuples import make_synthetic_codebook, render_synthetic
from neurogenesis.generators.algebraic import modular_task
from neurogenesis.generators.pool import divisor_modular_pool
from neurogenesis.models.encoders import build_encoder
from neurogenesis.oracle import enumerate as en
from neurogenesis.train.continual import run_stream
from neurogenesis.train.loop import TrainConfig, train_multitask

ORACLE = dict(mode="shared", closure="total", allow_noninjective=True)
STORE = Path("results/runs/e4.jsonl")

K, DIM, NOISE, N_TRAIN, N_EVAL, EPOCHS = 6, 64, 0.1, 8000, 1500, 40
ORDERS = {
    "greedy": ["aux_div_mod3_c0", "aux_div_mod2_c0"],  # |RS| 6 -> 2 -> 1
    "reverse": ["aux_div_mod2_c0", "aux_div_mod3_c0"],  # |RS| 6 -> 3 -> 1
    "random": ["aux_div_mod3_sum", "aux_div_mod2_c0"],
}


def build_stream(order: str):
    base = modular_task([1, K - 1], K)
    by_name = {t.name: t for t in divisor_modular_pool(K)}
    return [base] + [by_name[n] for n in ORDERS[order]]


def _data(seed: int):
    base = modular_task([1, K - 1], K)
    book = make_synthetic_codebook(K, DIM, np.random.default_rng(12345))
    rng = np.random.default_rng(seed)
    return (
        render_synthetic(base, N_TRAIN, book, NOISE, rng),
        render_synthetic(base, N_EVAL, book, NOISE, rng, "test"),
    )


def _run(job: tuple[str, str, int]) -> dict:
    order, strategy, seed = job
    tasks = build_stream(order)
    tr, te = _data(seed)
    cfg = TrainConfig(epochs=EPOCHS, encoder="mlp")

    if strategy == "joint":
        enc = build_encoder("mlp", k=K, in_dim=DIM, seed=seed)
        rs = en.rs_set(tasks, **ORACLE)
        res = train_multitask(tasks, enc, tr, {"test": te}, cfg, seed, rs=rs)
        m = res.metrics["test"]
        rec = {
            "strategy": "joint",
            "task_order": [t.name for t in tasks],
            "phases": [
                {
                    "phase": len(tasks) - 1,
                    "task_name": "joint",
                    "acc_y": m.acc_y,
                    "acc_c": m.acc_c,
                    "alpha_hat": m.alpha_hat,
                    "rs_count_seen": rs.count,
                    "in_rs_seen": bool(m.rs_membership),
                    "in_rs_prev": False,
                    "locked_in": False,
                    "forgetting": None,
                    "extra": {"truncated": rs.truncated, "collapse": m.collapse},
                }
            ],
            "runtime_s": res.runtime_s,
        }
    else:
        enc = build_encoder("mlp", k=K, in_dim=DIM, seed=seed)
        rec = run_stream(tasks, enc, tr, te, cfg, seed, strategy)

    rec |= {"experiment": "e4", "order": order, "seed": seed, "k": K}
    final = rec["phases"][-1]
    print(
        f"  {order:8s} {strategy:7s} seed={seed}  final |RS|={final['rs_count_seen']} "
        f"Acc(C)={final['acc_c']:.3f} inRS={int(final['in_rs_seen'])} "
        f"lock={sum(int(p['locked_in']) for p in rec['phases'])}  {rec['runtime_s']:.0f}s",
        flush=True,
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--store", type=Path, default=STORE)
    args = ap.parse_args()

    arms = [*STRATEGIES, "joint"]
    jobs = [(o, s, sd) for o in ORDERS for s in arms for sd in range(args.seeds)]
    print(
        f"E4: {len(ORDERS)} orders x {len(arms)} arms x {args.seeds} seeds = {len(jobs)} streams",
        flush=True,
    )

    args.store.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(_run, jobs):
            with open(args.store, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    print("E4 complete.")


if __name__ == "__main__":
    main()
