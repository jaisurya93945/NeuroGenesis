#!/usr/bin/env python
"""E1 -- does provable identifiability predict empirical symbol grounding?

Preregistered in ``paper/preregistration.md`` *before* any run. Four conditions of
``y = (c1 + w*c2) mod 10`` with |RS| in {1, 2, 5, 10}, 10 seeds each.

Usage:
    python experiments/e1_identifiability.py [--workers 4] [--seeds 10]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from neurogenesis.config import DataSpec, ModelSpec, RunConfig, TaskSpec
from neurogenesis.runner import DEFAULT_STORE, existing_hashes, run

# w -> |RS| = gcd(1+w, 10); verified by the oracle in tests/test_oracle_analytic.py
CONDITIONS = [(2, 1), (1, 2), (4, 5), (9, 10)]


def build_configs(n_seeds: int) -> list[RunConfig]:
    out = []
    for w, _rs in CONDITIONS:
        for seed in range(n_seeds):
            out.append(
                RunConfig(
                    experiment="e1",
                    task=TaskSpec(family="algebraic", weights=(1, w), m=10, seed=0),
                    data=DataSpec(tier="M", data_seed=seed),
                    model=ModelSpec(encoder="cnn", init_seed=seed),
                )
            )
    return out


def _run_one(cfg: RunConfig) -> dict:
    rec = run(cfg, store=None)  # workers return records; the parent writes them
    m = rec["metrics"]["test"]
    print(
        f"  w={cfg.task.weights[1]} |RS|={rec['oracle']['rs_count']:2d} "
        f"seed={cfg.model.init_seed}  Acc(Y)={m['acc_y']:.3f}  Acc(C)={m['acc_c']:.3f}  "
        f"alpha_hat={'id' if m['alpha_hat_is_identity'] else m['alpha_hat']}  "
        f"in_RS={m['rs_membership']}  {rec['runtime_s']:.0f}s",
        flush=True,
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = ap.parse_args()

    configs = build_configs(args.seeds)
    done = existing_hashes(args.store)
    todo = [c for c in configs if c.config_hash() not in done]
    print(f"E1: {len(configs)} configs, {len(todo)} to run, {args.workers} workers")

    args.store.parent.mkdir(parents=True, exist_ok=True)
    import json

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(_run_one, todo):
            with open(args.store, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    print("E1 complete.")


if __name__ == "__main__":
    main()
