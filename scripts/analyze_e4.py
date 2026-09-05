#!/usr/bin/env python
"""Analyse E4 exactly as preregistered in ``paper/preregistration_e4.md``.

Four confirmatory tests. P1 asks whether RS lock-in exists at all; a zero rate is a
real answer, not a failed experiment.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from neurogenesis.stats import bootstrap_diff, bootstrap_mean

GATE = 0.99
ORDERS = ("greedy", "reverse", "random")
SEQ_STRATEGIES = ("naive", "replay", "ewc", "cool")


def load(store: Path) -> list[dict]:
    with open(store) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def main() -> None:  # noqa: C901 - linear reporting script
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("results/runs/e4.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/e4_summary.json"))
    args = ap.parse_args()

    runs = [r for r in load(args.store) if r.get("experiment") == "e4"]
    if not runs:
        raise SystemExit("no E4 runs found")

    # Truncated oracle results make membership and lock-in undefined (RESULTS.md 7.5).
    trunc = [r for r in runs if any(p["extra"].get("truncated") for p in r["phases"])]
    if trunc:
        print(f"WARNING: {len(trunc)} streams had a truncated oracle result; excluded.")
    runs = [r for r in runs if r not in trunc]

    kept = [r for r in runs if r["phases"][-1]["acc_y"] >= GATE]
    print(
        f"E4: {len(runs)} streams; final-phase gate Acc(Y) >= {GATE} keeps {len(kept)} "
        f"({len(kept) / max(len(runs), 1):.1%})\n"
    )

    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in kept:
        cells[(r["order"], r["strategy"])].append(r)

    def final_acc(order: str, strategy: str) -> np.ndarray:
        return np.array([r["phases"][-1]["acc_c"] for r in cells.get((order, strategy), [])])

    def lock_rate(order: str, strategy: str) -> float:
        rs = cells.get((order, strategy), [])
        if not rs:
            return float("nan")
        return float(np.mean([any(p["locked_in"] for p in r["phases"]) for r in rs]))

    print(
        f"{'order':9s} {'strategy':9s} {'n':>3} {'final Acc(C) [95% CI]':>28} "
        f"{'lock-in rate':>13} {'forgetting':>11}"
    )
    summary: dict = {"cells": {}}
    for order in ORDERS:
        for strategy in (*SEQ_STRATEGIES, "joint"):
            acc = final_acc(order, strategy)
            if len(acc) == 0:
                continue
            est = bootstrap_mean(acc)
            lr = lock_rate(order, strategy)
            forg = [
                p["forgetting"]
                for r in cells[(order, strategy)]
                for p in r["phases"]
                if p["forgetting"] is not None
            ]
            fg = float(np.mean(forg)) if forg else float("nan")
            print(f"{order:9s} {strategy:9s} {len(acc):>3} {str(est):>28} {lr:>13.2f} {fg:>11.3f}")
            summary["cells"][f"{order}|{strategy}"] = {
                "n": len(acc),
                "acc_c": est.mean,
                "lo": est.lo,
                "hi": est.hi,
                "lock_rate": lr,
                "mean_forgetting": fg,
                "values": acc.tolist(),
            }
        print()

    # ---- P1: does lock-in occur at all? -------------------------------------
    all_lock = {f"{o}|{s}": lock_rate(o, s) for o in ORDERS for s in SEQ_STRATEGIES}
    any_lock = {k: v for k, v in all_lock.items() if v == v and v > 0}
    print(f"P1  lock-in detectable in any cell?  {'YES' if any_lock else 'NO'}")
    if any_lock:
        for k, v in sorted(any_lock.items(), key=lambda kv: -kv[1])[:5]:
            print(f"      {k}: {v:.2f}")
    else:
        print("      Zero in every cell: models escape shortcuts once the forbidding")
        print("      constraint arrives. A clean negative, reported as a finding.")
    print(f"    P1: {'MET' if any_lock else 'NOT MET (informative negative)'}")
    summary["P1"] = {"rates": all_lock, "met": bool(any_lock)}

    # ---- P2: naive worse than replay ----------------------------------------
    naive = np.concatenate([final_acc(o, "naive") for o in ORDERS])
    replay = np.concatenate([final_acc(o, "replay") for o in ORDERS])
    cool = np.concatenate([final_acc(o, "cool") for o in ORDERS])
    print("\nP2  naive vs rehearsal (pooled over orders)")
    if len(naive) and len(replay):
        d = bootstrap_diff(replay, naive)
        print(f"      replay − naive = {d}")
        print(f"      cool mean {cool.mean():.3f} | naive mean {naive.mean():.3f}")
        p2 = d.mean > 0
        print(f"    P2: {'MET' if p2 else 'NOT MET'}")
        summary["P2"] = {"diff_replay_minus_naive": d.__dict__, "met": bool(p2)}

    # ---- P3: sequential costs something -------------------------------------
    joint = np.concatenate([final_acc(o, "joint") for o in ORDERS])
    best_seq_name = max(
        SEQ_STRATEGIES,
        key=lambda s: (
            np.concatenate([final_acc(o, s) for o in ORDERS]).mean()
            if len(np.concatenate([final_acc(o, s) for o in ORDERS]))
            else -1
        ),
    )
    best_seq = np.concatenate([final_acc(o, best_seq_name) for o in ORDERS])
    print(f"\nP3  joint vs best sequential ({best_seq_name})")
    if len(joint) and len(best_seq):
        d = bootstrap_diff(joint, best_seq)
        print(f"      joint {joint.mean():.3f} vs {best_seq_name} {best_seq.mean():.3f}")
        print(f"      difference = {d}   (predicted > 0.05)")
        p3 = d.mean > 0.05
        print(f"    P3: {'MET' if p3 else 'NOT MET'}")
        if not p3:
            print("        -> sequential matches joint: the continual framing adds nothing here")
        summary["P3"] = {"best_seq": best_seq_name, "diff": d.__dict__, "met": bool(p3)}

    # ---- P4: order matters ---------------------------------------------------
    g = np.concatenate([final_acc("greedy", s) for s in SEQ_STRATEGIES])
    rv = np.concatenate([final_acc("reverse", s) for s in SEQ_STRATEGIES])
    print("\nP4  greedy order (|RS| 6→2→1) vs reverse (6→3→1), pooled over strategies")
    if len(g) and len(rv):
        d = bootstrap_diff(g, rv)
        print(f"      greedy {g.mean():.3f} vs reverse {rv.mean():.3f}")
        print(f"      difference = {d}   (predicted > 0.05)")
        p4 = d.mean > 0.05
        print(f"    P4: {'MET' if p4 else 'NOT MET'}")
        summary["P4"] = {"diff": d.__dict__, "met": bool(p4)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
