#!/usr/bin/env python
"""E3 -- does selecting tasks by shortcut coverage beat the alternatives?

Preregistered in ``paper/preregistration_e3.md`` before any run. After E2, this is
the only remaining route to a *design* contribution.

Usage:
    python experiments/e3_selection.py [--workers 4] [--seeds 8]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from neurogenesis.data import mnist
from neurogenesis.data.tuples import make_synthetic_codebook, render_mnist, render_synthetic
from neurogenesis.generators.algebraic import modular_task
from neurogenesis.generators.pool import divisor_modular_pool
from neurogenesis.models.encoders import build_encoder
from neurogenesis.oracle import enumerate as en
from neurogenesis.selection import (
    exhaustive_optimal,
    greedy_cover,
    information_greedy,
    random_selection,
)
from neurogenesis.train.loop import TrainConfig, train_multitask

ORACLE = dict(mode="shared", closure="total", allow_noninjective=True)
STORE = Path("results/runs/e3.jsonl")

NOISE, DIM, N_TRAIN, N_EVAL, EPOCHS = 0.1, 32, 8000, 1500, 30
CONCEPT_BUDGETS = (25, 50, 100, 200, 400)
TASK_BUDGETS = (0, 1, 2, 3)

#: Tier M (MNIST digits) recipe, frozen by the dev screen in ``paper/preregistration_e6.md``.
#: Kept separate from the Tier S constants above so a replication cannot silently retune Tier S.
M_N_TRAIN, M_N_EVAL, M_EPOCHS = 8000, 1500, 30


def base_and_pool(k: int):
    return modular_task([1, k - 1], k), divisor_modular_pool(k)


def selections(k: int, concept_budgets: tuple[int, ...] = CONCEPT_BUDGETS) -> list[dict]:
    """Every (method, budget) cell, resolved to a concrete task subset."""
    base, pool = base_and_pool(k)
    out: list[dict] = []
    for b in TASK_BUDGETS:
        if b == 0:
            out.append({"method": "base_only", "budget": 0, "idx": []})
            continue
        out.append({"method": "greedy_rs", "budget": b, "idx": greedy_cover(base, pool, b).chosen})
        out.append(
            {
                "method": "information_greedy",
                "budget": b,
                "idx": information_greedy(base, pool, b).chosen,
            }
        )
        out.append(
            {
                "method": "exhaustive_optimal",
                "budget": b,
                "idx": exhaustive_optimal(base, pool, b).chosen,
            }
        )
        for d in range(5):
            out.append(
                {
                    "method": "random",
                    "budget": b,
                    "draw": d,
                    "idx": random_selection(base, pool, b, np.random.default_rng(1000 + d)).chosen,
                }
            )
    out.append({"method": "all_tasks", "budget": len(pool), "idx": list(range(len(pool)))})
    # the distractor-only arm (P5): costs budget, eliminates nothing
    inv = [i for i, t in enumerate(pool) if t.meta.get("shift_invariant")]
    out.append({"method": "distractors_only", "budget": len(inv), "idx": inv})
    for n in concept_budgets:
        out.append({"method": "concept_supervision", "budget": 0, "idx": [], "n_labels": n})
    return out


def _render(base, k: int, seed: int, tier: str):
    """Tier S = synthetic codebook vectors; Tier M = MNIST digit images.

    The only thing that changes between tiers is perception. The task, the pool, the
    oracle and the selection methods are identical, which is what makes the
    replication a test of the *conclusion* rather than of a different experiment.
    """
    rng = np.random.default_rng(seed)
    if tier == "M":
        data = mnist.load()
        tr = render_mnist(base, M_N_TRAIN, "train", data, rng)
        te = render_mnist(base, M_N_EVAL, "test", data, rng)
        return tr, te, "cnn", {}, M_EPOCHS
    book = make_synthetic_codebook(k, DIM, np.random.default_rng(12345))
    tr = render_synthetic(base, N_TRAIN, book, NOISE, rng)
    te = render_synthetic(base, N_EVAL, book, NOISE, rng, "test")
    return tr, te, "mlp", {"in_dim": DIM}, EPOCHS


def _run(job: tuple[int, dict, int, str, int | None]) -> dict:
    k, sel, seed, tier, epochs_override = job
    base, pool = base_and_pool(k)
    tasks = [base] + [pool[i] for i in sel["idx"]]
    rs = en.rs_set(tasks, **ORACLE)

    tr, te, kind, enc_kwargs, epochs = _render(base, k, seed, tier)
    if epochs_override is not None:
        epochs = epochs_override

    # seed inside the factory, before the weights are drawn -- never afterwards
    enc = build_encoder(kind, k=k, seed=seed, **enc_kwargs)
    res = train_multitask(
        tasks,
        enc,
        tr,
        {"test": te},
        TrainConfig(epochs=epochs, encoder=kind),
        seed,
        rs=rs,
        n_concept_labels=sel.get("n_labels", 0),
    )
    m = res.metrics["test"]
    return {
        "experiment": "e3",
        "tier": tier,
        "k": k,
        "method": sel["method"],
        "budget": sel["budget"],
        "draw": sel.get("draw"),
        "n_labels": sel.get("n_labels", 0),
        "chosen": [pool[i].name for i in sel["idx"]],
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
    ap.add_argument("--ks", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--tier", choices=["S", "M"], default="S")
    ap.add_argument("--seed-offset", type=int, default=0, help="use >=900 for dev screens")
    ap.add_argument("--methods", nargs="+", default=None, help="restrict to these methods")
    ap.add_argument("--concept-budgets", type=int, nargs="+", default=list(CONCEPT_BUDGETS))
    ap.add_argument("--epochs", type=int, default=None, help="override the frozen tier recipe")
    ap.add_argument("--store", type=Path, default=STORE)
    args = ap.parse_args()

    if args.tier == "M" and max(args.ks) > 10:
        raise SystemExit("Tier M renders MNIST digits: k <= 10")

    sels = {k: selections(k, tuple(args.concept_budgets)) for k in args.ks}
    if args.methods:
        keep = set(args.methods)
        sels = {k: [s for s in v if s["method"] in keep] for k, v in sels.items()}
    jobs = [
        (k, sel, args.seed_offset + s, args.tier, args.epochs)
        for k in args.ks
        for sel in sels[k]
        for s in range(args.seeds)
    ]
    print(
        f"E3 (tier {args.tier}): {len(jobs)} runs ({len(args.ks)} instances, {args.seeds} seeds)",
        flush=True,
    )

    args.store.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(_run, jobs), 1):
            with open(args.store, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print("E3 complete.")


if __name__ == "__main__":
    main()
