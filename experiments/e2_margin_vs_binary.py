#!/usr/bin/env python
"""E2 -- does the margin predict grounding better than binary identifiability?

Preregistered in ``paper/preregistration_e2.md`` before any run. ~200 tasks across
five generator families x 5 seeds, Tier S.

Usage:
    python experiments/e2_margin_vs_binary.py [--workers 4] [--seeds 5]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from neurogenesis.config import (
    CONFIRMATORY_SEED_FLOOR,
    DataSpec,
    ModelSpec,
    OptimSpec,
    RunConfig,
    TaskSpec,
)
from neurogenesis.runner import DEFAULT_STORE, existing_hashes, run

# Frozen recipe from the dev calibration recorded in the preregistration.
DATA = dict(tier="S", n_train=8000, n_val=1500, n_test=1500, noise=0.1, dim=32)
OPTIM = dict(epochs=30)


def task_specs() -> list[TaskSpec]:
    """~200 confirmatory tasks. Generator seeds are all >= CONFIRMATORY_SEED_FLOOR."""
    out: list[TaskSpec] = []
    sid = CONFIRMATORY_SEED_FLOOR

    # 1. algebraic -- closed-form |RS|, spans {1,2,...} by construction
    for m in (5, 6, 7, 8, 9, 10):
        for w in range(1, m):
            out.append(TaskSpec(family="algebraic", weights=(1, w), m=m, k=m, seed=sid))
            sid += 1

    # 2. addition -- identifiable anchors at several vocabulary sizes
    for k in (4, 5, 6, 7, 8, 9, 10):
        out.append(TaskSpec(family="addition", k=k, m=k, seed=sid))
        sid += 1

    # 3. planted -- direct control of the shortcut set, incl. non-injective collapses
    for k in (5, 6, 7):
        for kind in ("identity", "cyclic", "collapse", "mixed"):
            for _ in range(3):
                out.append(TaskSpec(family="planted", k=k, m=k, planted_kind=kind, seed=sid))
                sid += 1

    # 4. support-thinned -- identifiability continuum at fixed knowledge
    for k in (6, 8):
        for density in (0.2, 0.3, 0.4, 0.5, 0.7, 0.9):
            for _ in range(2):
                out.append(TaskSpec(family="addition", k=k, m=k, support_density=density, seed=sid))
                sid += 1

    # 5. rarefied -- THE decisive H2 cell: |RS| = 1 with margin swept over 3 orders
    for k in (6, 8, 10):
        for rarity in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
            for swap in ((0, 1), (2, 3)):
                out.append(
                    TaskSpec(family="rarefied", k=k, m=k, rarity=rarity, swap=swap, seed=sid)
                )
                sid += 1

    # 6. random -- null family, no planted structure at all
    for k in (4, 5, 6):
        for _ in range(6):
            out.append(TaskSpec(family="random", k=k, m=k, seed=sid))
            sid += 1

    return out


#: Tier M (MNIST) overrides for the E6 replication arm. MNIST supplies 10 digits, so
#: only k <= 10 tasks can be rendered; noise/dim are Tier-S-only and are dropped.
M_DATA = dict(tier="M", n_train=8000, n_val=1500, n_test=1500)


def tier_m_subset(every: int = 7) -> list[TaskSpec]:
    """A spread-out `k <= 10` subset of the E2 task set, for the Tier-M replication.

    Taking every ``every``-th eligible spec keeps all five generator families
    represented rather than over-sampling whichever family happens to come first.
    """
    eligible = [s for s in task_specs() if (s.k or s.m) <= 10]
    return eligible[::every]


def build_configs(n_seeds: int, tier: str = "S", specs: list[TaskSpec] | None = None):
    specs = specs if specs is not None else task_specs()
    data = M_DATA if tier == "M" else DATA
    encoder = "cnn" if tier == "M" else "mlp"
    return [
        RunConfig(
            experiment="e2" if tier == "S" else "e6_e2subset",
            task=spec,
            data=DataSpec(**data, data_seed=s),
            model=ModelSpec(encoder=encoder, init_seed=s),
            optim=OptimSpec(**OPTIM),
        )
        for spec in specs
        for s in range(n_seeds)
    ]


def _run_one(cfg: RunConfig) -> dict:
    try:
        return run(cfg, store=None)
    except Exception as exc:  # noqa: BLE001 - one bad task must not kill the sweep
        return {
            "error": repr(exc),
            "config_hash": cfg.config_hash(),
            "experiment": "e2",
            "config": cfg.to_dict(),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tier", choices=["S", "M"], default="S")
    ap.add_argument("--subset-every", type=int, default=7, help="Tier M: take every Nth k<=10 task")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = ap.parse_args()

    specs = tier_m_subset(args.subset_every) if args.tier == "M" else task_specs()
    configs = build_configs(args.seeds, args.tier, specs)
    done = existing_hashes(args.store)
    todo = [c for c in configs if c.config_hash() not in done]
    print(
        f"E2 (tier {args.tier}): {len(specs)} tasks, {len(configs)} runs, "
        f"{len(todo)} to do, {args.workers} workers",
        flush=True,
    )

    args.store.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(_run_one, todo), 1):
            with open(args.store, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if "error" in rec:
                n_err += 1
            else:
                n_ok += 1
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}  ok={n_ok} err={n_err}", flush=True)
    print(f"E2 complete: {n_ok} runs, {n_err} errors.")


if __name__ == "__main__":
    main()
