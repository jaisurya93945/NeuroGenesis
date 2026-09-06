#!/usr/bin/env python
"""E5 -- does the cost comparison flip as the concept vocabulary grows?

Preregistered in ``paper/preregistration_e5.md`` before any run. E3 falsified H4 at
``k = 6``; E5 asks whether that verdict is an artefact of a small concept space.

The rule side is flat at 2 by deterministic combinatorics (established and
disclosed in the preregistration). The only unknown measured here is ``L(k)``, the
concept labels needed to ground.

Usage:
    python experiments/e5_cost_scaling.py [--workers 4] [--seeds 8]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from neurogenesis.data.tuples import make_synthetic_codebook, render_synthetic
from neurogenesis.generators.algebraic import modular_task
from neurogenesis.generators.pool import divisor_modular_pool
from neurogenesis.models.encoders import build_encoder
from neurogenesis.oracle import enumerate as en
from neurogenesis.selection import greedy_cover
from neurogenesis.train.loop import TrainConfig, train_multitask

ORACLE = dict(mode="shared", closure="total", allow_noninjective=True)
STORE = Path("results/runs/e5.jsonl")

# Frozen recipe from the dev screen, recorded in the preregistration.
DIM, NOISE, N_TRAIN, N_EVAL, EPOCHS = 64, 0.1, 15_000, 2000, 40

COMPOSITE_KS = (6, 10, 12, 15, 20, 24, 30)
PRIME_POWER_KS = (8, 16)  # the declared structural blind spot (P4)
LABEL_BUDGETS = (10, 20, 40, 80, 160, 320, 640, 1280)  # fixed in advance
RULE_BUDGETS = (0, 1, 2, 3)


def cells(ks: tuple[int, ...]) -> list[dict]:
    out = []
    for k in ks:
        for b in RULE_BUDGETS:
            out.append({"k": k, "arm": "rules", "budget": b})
        for n in LABEL_BUDGETS:
            out.append({"k": k, "arm": "labels", "budget": n})
    return out


def _run(job: tuple[dict, int]) -> dict:
    cell, seed = job
    k = cell["k"]
    base = modular_task([1, k - 1], k)
    pool = divisor_modular_pool(k)

    if cell["arm"] == "rules":
        idx = greedy_cover(base, pool, cell["budget"]).chosen if cell["budget"] else []
        n_labels = 0
    else:
        idx, n_labels = [], cell["budget"]

    tasks = [base] + [pool[i] for i in idx]
    rs = en.rs_set(tasks, **ORACLE)

    book = make_synthetic_codebook(k, DIM, np.random.default_rng(12345))
    rng = np.random.default_rng(seed)
    tr = render_synthetic(base, N_TRAIN, book, NOISE, rng)
    te = render_synthetic(base, N_EVAL, book, NOISE, rng, "test")

    enc = build_encoder("mlp", k=k, in_dim=DIM, seed=seed)
    res = train_multitask(
        tasks,
        enc,
        tr,
        {"test": te},
        TrainConfig(epochs=EPOCHS, encoder="mlp"),
        seed,
        rs=rs,
        n_concept_labels=n_labels,
    )
    m = res.metrics["test"]
    return {
        "experiment": "e5",
        "k": k,
        "arm": cell["arm"],
        "budget": cell["budget"],
        "n_rules": len(idx),
        "n_labels": n_labels,
        "chosen": [pool[i].name for i in idx],
        "rs_count": rs.count,
        "identifiable": rs.is_identifiable,
        "seed": seed,
        "acc_y": m.acc_y,
        "acc_c": m.acc_c,
        "f1_c": m.f1_c,
        "collapse": m.collapse,
        "alpha_hat": m.alpha_hat,
        "rs_membership": m.rs_membership,
        "runtime_s": res.runtime_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--store", type=Path, default=STORE)
    args = ap.parse_args()

    all_cells = cells(COMPOSITE_KS + PRIME_POWER_KS)
    jobs = [(c, s) for c in all_cells for s in range(args.seeds)]
    print(f"E5: {len(all_cells)} cells x {args.seeds} seeds = {len(jobs)} runs", flush=True)

    args.store.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(_run, jobs), 1):
            with open(args.store, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print("E5 complete.")


if __name__ == "__main__":
    main()
