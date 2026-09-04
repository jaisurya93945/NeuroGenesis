#!/usr/bin/env python
"""Diagnostic: why do some modular-task runs fail to fit the label at all?

E1's first runs showed a bimodal outcome in the *identifiable* condition: some
seeds reach Acc(Y) ~ 0.98 with Acc(C) ~ 0.99, others stall at Acc(Y) ~ 0.43 with
Acc(C) = 0 and a **non-injective** alpha_hat that is not in RS(T). The latter are
not reasoning shortcuts -- they are degenerate optima where the encoder has
collapsed several concepts together and cannot escape.

That distinction matters for the science: a run that never fit its own objective
tells us nothing about whether identifiability predicts grounding. The
preregistered convergence gate excludes them, but a ~50% exclusion rate costs a
lot of power, so this script asks whether a better *frozen recipe* exists.

Run on DEV tasks only (task seed < 100), which is what the leakage guard permits
to be tuned on. Any recipe change found here requires re-preregistration before
confirmatory runs.

Usage: python experiments/diag_convergence.py [--seeds 6] [--workers 4]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

from neurogenesis.config import DataSpec, ModelSpec, OptimSpec, RunConfig, TaskSpec
from neurogenesis.runner import run

# Candidate recipes, all cheap. The question is which reliably escapes the
# degenerate optimum, not which squeezes out the last point of accuracy.
RECIPES = {
    "baseline_e1": dict(epochs=15, lr=1e-3, batch_size=128),
    "lower_lr": dict(epochs=15, lr=3e-4, batch_size=128),
    "smaller_batch": dict(epochs=15, lr=1e-3, batch_size=64),
    "longer": dict(epochs=30, lr=1e-3, batch_size=128),
    "lower_lr_longer": dict(epochs=30, lr=3e-4, batch_size=128),
}


def _run(job):
    name, w, seed = job
    r = RECIPES[name]
    cfg = RunConfig(
        experiment=f"diag_convergence.{name}",
        # dev task seed (<100): tuning here is permitted
        task=TaskSpec(family="algebraic", weights=(1, w), m=10, seed=0),
        data=DataSpec(tier="M", n_train=25_000, data_seed=seed),
        model=ModelSpec(encoder="cnn", init_seed=seed),
        optim=OptimSpec(
            epochs=r["epochs"], lr=r["lr"], batch_size=r["batch_size"], lr_final=r["lr"] / 10
        ),
        tuning_mode=True,
    )
    rec = run(cfg, store=None)
    m = rec["metrics"]["test"]
    print(
        f"  {name:18s} w={w} seed={seed}  Acc(Y)={m['acc_y']:.3f}  Acc(C)={m['acc_c']:.3f}  "
        f"distinct={m['n_distinct_concepts']:2d}  {rec['runtime_s']:.0f}s",
        flush=True,
    )
    return name, w, seed, m["acc_y"], m["acc_c"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--w", type=int, nargs="+", default=[2, 9])
    args = ap.parse_args()

    jobs = [(n, w, s) for n in RECIPES for w in args.w for s in range(args.seeds)]
    print(f"convergence diagnostic: {len(jobs)} runs\n")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results.extend(ex.map(_run, jobs))

    print("\n=== convergence rate by recipe (Acc(Y) > 0.9) ===")
    print(f"{'recipe':20s} {'w':>3} {'converged':>10} {'mean Acc(C)|conv':>18}")
    for name in RECIPES:
        for w in args.w:
            sel = [r for r in results if r[0] == name and r[1] == w]
            conv = [r for r in sel if r[3] > 0.9]
            mc = sum(r[4] for r in conv) / len(conv) if conv else float("nan")
            print(f"{name:20s} {w:>3} {len(conv):>4}/{len(sel):<5} {mc:>18.4f}")


if __name__ == "__main__":
    main()
