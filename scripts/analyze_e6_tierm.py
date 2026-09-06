#!/usr/bin/env python
"""Analyse E6 exactly as preregistered in ``paper/preregistration_e6.md``.

Four confirmatory tests. P4 is the one that matters: if it FAILS, greedy RS-cover wins
on cost under realistic perception and H4 is partially revived at Tier M -- a positive
result for a hypothesis this project has twice declared dead, to be reported as loudly
as the falsifications were.

Written and committed before any confirmatory E6 run existed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from neurogenesis.stats import bootstrap_diff, bootstrap_mean

GATE = 0.97  # Tier-M absolute gate; see preregistration_e6.md section 2(b)
SENSITIVITY_GATES = (0.95, 0.98)
CONCEPT_BUDGETS = (2, 5, 10, 15, 25, 50)
TASK_BUDGETS = (1, 2, 3)
TAUS = (25, 50, 100, 200, 400)
GROUND = 0.9


def load(store: Path) -> list[dict]:
    with open(store) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def cell_key(r: dict) -> tuple:
    return (r["method"], r["budget"], r.get("n_labels", 0))


def summarise(runs: list[dict], gate: float) -> dict[tuple, dict]:
    kept = [r for r in runs if r["acc_y"] >= gate]
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in kept:
        cells[cell_key(r)].append(r)
    out = {}
    for key, rs in cells.items():
        acc = np.array([r["acc_c"] for r in rs])
        est = bootstrap_mean(acc)
        out[key] = {
            "n": len(rs),
            "n_total": sum(1 for r in runs if cell_key(r) == key),
            "acc_c": float(acc.mean()),
            "lo": est.lo,
            "hi": est.hi,
            "rs_count": rs[0]["rs_count"],
            "grounded": float((acc > GROUND).mean()),
            "values": acc.tolist(),
        }
    return out


def acc_of(cells: dict, method: str, budget: int, n_labels: int = 0) -> np.ndarray:
    c = cells.get((method, budget, n_labels))
    return np.array(c["values"]) if c else np.array([])


def main() -> None:  # noqa: C901 - linear reporting script
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("results/runs/e6.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/e6_summary.json"))
    args = ap.parse_args()

    runs = [r for r in load(args.store) if r.get("tier") == "M" and r.get("k") == 6]
    if not runs:
        raise SystemExit("no Tier-M k=6 runs found")

    # The truncation trap from RESULTS.md 7.5: membership is undefined on a truncated
    # oracle result. At k=6 none is expected; assert rather than assume.
    trunc = [r for r in runs if r.get("truncated")]
    if trunc:
        raise SystemExit(f"{len(trunc)} runs carry a truncated oracle result; membership undefined")

    kept = [r for r in runs if r["acc_y"] >= GATE]
    print(
        f"E6 (Tier M, MNIST): {len(runs)} runs; absolute gate Acc(Y) >= {GATE} "
        f"keeps {len(kept)} ({len(kept) / len(runs):.1%})"
    )
    print(
        f"    Acc(Y) range {min(r['acc_y'] for r in runs):.4f}-{max(r['acc_y'] for r in runs):.4f}"
    )
    for g in SENSITIVITY_GATES:
        n = sum(1 for r in runs if r["acc_y"] >= g)
        print(f"    sensitivity: gate {g} would keep {n} ({n / len(runs):.1%})")
    print()

    cells = summarise(runs, GATE)
    summary: dict = {"gate": GATE, "n_runs": len(runs), "n_gated": len(kept), "cells": {}}

    print(
        f"{'method':22s} {'budget':>6} {'labels':>6} {'|RS|':>4} {'n':>3} "
        f"{'Acc(C) [95% CI]':>26} {'grounded':>8}"
    )
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2])):
        c = cells[key]
        ci = f"{c['acc_c']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]"
        print(
            f"{key[0]:22s} {key[1]:>6} {key[2]:>6} {c['rs_count']:>4} {c['n']:>3} "
            f"{ci:>26} {c['grounded']:>8.2f}"
        )
        summary["cells"]["|".join(map(str, key))] = c
    print()

    # ---- P1: the method still works ----------------------------------------
    g2 = acc_of(cells, "greedy_rs", 2)
    p1_ground = len(g2) > 0 and g2.mean() > GROUND
    matches = []
    for b in TASK_BUDGETS:
        gc = cells.get(("greedy_rs", b, 0))
        oc = cells.get(("exhaustive_optimal", b, 0))
        if gc and oc:
            matches.append(gc["rs_count"] == oc["rs_count"])
            print(
                f"P1  budget {b}: greedy |RS|={gc['rs_count']}  optimal |RS|={oc['rs_count']}  "
                f"{'match' if matches[-1] else 'DIFFER'}"
            )
    p1 = bool(p1_ground and matches and all(matches))
    print(
        f"    greedy at budget 2: Acc(C) = {g2.mean() if len(g2) else float('nan'):.4f} "
        f"(predicted > {GROUND})"
    )
    print(f"    P1: {'MET' if p1 else 'NOT MET'}\n")
    summary["P1"] = {"greedy_b2_acc_c": float(g2.mean()) if len(g2) else None, "met": p1}

    # ---- P2: ordering among selection methods ------------------------------
    r2 = acc_of(cells, "random", 2)
    d = bootstrap_diff(g2, r2) if len(g2) and len(r2) else None
    beats_info = []
    for b in TASK_BUDGETS:
        gb, ib = acc_of(cells, "greedy_rs", b), acc_of(cells, "information_greedy", b)
        if len(gb) and len(ib):
            beats_info.append(gb.mean() >= ib.mean())
            print(f"P2  budget {b}: greedy {gb.mean():.3f} vs information-greedy {ib.mean():.3f}")
    if d is not None:
        print(f"    budget 2: greedy − random = {d}")
    p2 = bool(d is not None and d.lo > 0 and beats_info and all(beats_info))
    print(f"    P2: {'MET' if p2 else 'NOT MET'}\n")
    summary["P2"] = {"greedy_minus_random_b2": d.__dict__ if d else None, "met": p2}

    # ---- P3 lives in the E2-subset arm; reported by its own block ----------
    print(
        "P3  |RS| still predicts Acc(C): see the E2-subset arm "
        "(scripts/analyze_e2.py --store results/runs/e6_e2subset.jsonl)\n"
    )

    # ---- P4: the cost verdict ----------------------------------------------
    L_M = None
    for n in CONCEPT_BUDGETS:
        c = cells.get(("concept_supervision", 0, n))
        if c and c["acc_c"] > GROUND:
            L_M = n
            break
    R_M = None
    for b in TASK_BUDGETS:
        a = acc_of(cells, "greedy_rs", b)
        if len(a) and a.mean() > GROUND:
            R_M = b
            break

    print("P4  cost to reach Acc(C) > 0.9:  greedy (budget x tau) vs concept labels")
    censored = CONCEPT_BUDGETS[0] == L_M
    print(
        f"      cheapest concept supervision that grounds: "
        f"{'<= ' if censored else ''}{L_M}   (grid {CONCEPT_BUDGETS})"
    )
    print(f"      greedy grounds at task budget: {R_M}")
    verdicts = {}
    if L_M is not None and R_M is not None:
        print(f"      {'tau':>6} {'greedy cost':>12} {'concept cost':>13} {'winner':>10}")
        for tau in TAUS:
            gcost, ccost = R_M * tau, L_M
            w = "tie" if gcost == ccost else ("greedy" if gcost < ccost else "concept")
            verdicts[tau] = w
            print(f"      {tau:>6} {gcost:>12} {ccost:>13} {w:>10}")
    # P4 as preregistered: concept supervision still wins or ties at every tau above the smallest
    p4 = bool(verdicts) and all(v != "greedy" for t, v in verdicts.items() if t > min(TAUS))
    print(f"    P4: {'MET' if p4 else 'NOT MET'}")
    if not p4:
        print("        -> greedy WINS on cost at Tier M. H4 is partially revived at the")
        print("           realistic-perception tier. Report at full prominence.")
    summary["P4"] = {
        "L_M": L_M,
        "L_M_censored": censored,
        "R_M": R_M,
        "verdicts": verdicts,
        "met": p4,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
