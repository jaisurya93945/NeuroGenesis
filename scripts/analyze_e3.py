#!/usr/bin/env python
"""Analyse E3 exactly as preregistered in ``paper/preregistration_e3.md``.

Five confirmatory tests. P4 is the one that decides whether this project has a
design contribution at all.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from neurogenesis.stats import bootstrap_diff, bootstrap_mean

GATE = 0.99  # absolute, per the preregistration -- not E2's relative gate
GROUNDED = 0.9
TAUS = (25, 50, 100, 200, 400)


def load(store: Path) -> list[dict]:
    with open(store) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def cell_key(r: dict) -> tuple:
    return (r["k"], r["method"], r["budget"], r["n_labels"])


def main() -> None:  # noqa: C901 - reporting script
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("results/runs/e3.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/e3_summary.json"))
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    runs = [r for r in load(args.store) if r.get("experiment") == "e3"]
    if not runs:
        raise SystemExit("no E3 runs found")

    n_all = len(runs)
    kept = [r for r in runs if r["acc_y"] >= GATE]
    print(
        f"E3: {n_all} runs; absolute gate Acc(Y) >= {GATE} keeps {len(kept)} "
        f"({len(kept) / n_all:.1%})\n"
    )

    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in kept:
        cells[cell_key(r)].append(r)

    def stat(k: int, method: str, budget: int, n_labels: int = 0):
        rs = cells.get((k, method, budget, n_labels), [])
        if not rs:
            return None
        acc = np.array([r["acc_c"] for r in rs])
        return {
            "n": len(rs),
            "acc_c": float(acc.mean()),
            "est": bootstrap_mean(acc),
            "frac_grounded": float((acc > GROUNDED).mean()),
            "rs_count": rs[0]["rs_count"],
            "chosen": rs[0]["chosen"],
            "values": acc.tolist(),
        }

    summary: dict = {"n_runs": n_all, "n_gated": len(kept), "gate": GATE}

    for k in sorted({r["k"] for r in kept}):
        primary = k == args.k
        print(f"{'=' * 78}\nINSTANCE k={k}{'  (primary)' if primary else '  (secondary)'}")
        print(
            f"{'method':22s} {'budget':>6} {'|RS|':>5} {'n':>3} "
            f"{'Acc(C) [95% CI]':>26} {'grounded':>9}  chosen"
        )
        rows = []
        for method in (
            "base_only",
            "greedy_rs",
            "information_greedy",
            "exhaustive_optimal",
            "random",
            "distractors_only",
            "all_tasks",
        ):
            for budget in sorted({b for (kk, m, b, _) in cells if kk == k and m == method}):
                s = stat(k, method, budget)
                if not s:
                    continue
                rows.append((method, budget, s))
                ch = ",".join(c.replace("aux_", "") for c in s["chosen"])[:34]
                print(
                    f"{method:22s} {budget:>6} {s['rs_count']:>5} {s['n']:>3} "
                    f"{str(s['est']):>26} {s['frac_grounded']:>9.2f}  {ch}"
                )
        for n in (25, 50, 100, 200, 400):
            s = stat(k, "concept_supervision", 0, n)
            if s:
                rows.append((f"concept_sup_{n}", 0, s))
                print(
                    f"{'concept_supervision':22s} {n:>6} {s['rs_count']:>5} {s['n']:>3} "
                    f"{str(s['est']):>26} {s['frac_grounded']:>9.2f}  {n} labels"
                )
        summary[f"k{k}"] = {
            f"{m}_b{b}": {kk: vv for kk, vv in s.items() if kk != "est"} for m, b, s in rows
        }
        print()

    k = args.k

    def min_budget_to_ground(method: str) -> int | None:
        for b in sorted({bb for (kk, m, bb, _) in cells if kk == k and m == method}):
            s = stat(k, method, b)
            if s and s["acc_c"] > GROUNDED:
                return b
        return None

    print("=" * 78)
    gb = min_budget_to_ground("greedy_rs")
    ib = min_budget_to_ground("information_greedy")
    print(f"P1  minimum task budget reaching mean Acc(C) > {GROUNDED}")
    print(f"      greedy RS-cover   : {gb}")
    print(f"      information-greedy: {ib}")
    p1 = gb is not None and (ib is None or gb < ib)
    print(f"    P1: {'MET' if p1 else 'NOT MET'}")
    summary["P1"] = {"greedy_budget": gb, "info_budget": ib, "met": bool(p1)}

    print("\nP2  greedy vs exhaustive optimum on final |RS|")
    p2 = True
    for b in (1, 2, 3):
        g, e = stat(k, "greedy_rs", b), stat(k, "exhaustive_optimal", b)
        if g and e:
            same = g["rs_count"] == e["rs_count"]
            p2 &= same
            print(
                f"      budget {b}: greedy |RS|={g['rs_count']}  optimal |RS|={e['rs_count']}"
                f"  {'match' if same else 'DIFFER'}"
            )
    print(f"    P2: {'MET' if p2 else 'NOT MET'}")
    summary["P2"] = {"met": bool(p2)}

    print("\nP3  greedy vs random at matched budget")
    p3_ok = []
    for b in (1, 2, 3):
        g, r = stat(k, "greedy_rs", b), stat(k, "random", b)
        if g and r:
            d = bootstrap_diff(np.array(g["values"]), np.array(r["values"]))
            p3_ok.append(d.mean > 0)
            print(f"      budget {b}: greedy {g['acc_c']:.3f} vs random {r['acc_c']:.3f}  diff {d}")
    p3 = bool(p3_ok) and all(p3_ok)
    print(f"    P3: {'MET' if p3 else 'NOT MET'}")
    summary["P3"] = {"met": p3}

    # ---- P4: the one that decides whether there is a contribution ----------
    print("\nP4  cost to reach Acc(C) > 0.9:  greedy (budget x tau) vs concept labels")
    cs_min = None
    for n in (25, 50, 100, 200, 400):
        s = stat(k, "concept_supervision", 0, n)
        if s and s["acc_c"] > GROUNDED:
            cs_min = n
            break
    print(
        f"      cheapest concept supervision that grounds: "
        f"{cs_min if cs_min else 'none of the budgets tried'}"
    )
    print(f"      greedy grounds at task budget: {gb}")
    tau_rows = {}
    if gb is not None and cs_min is not None:
        print(f"      {'tau':>6} {'greedy cost':>12} {'concept cost':>13} {'winner':>10}")
        for tau in TAUS:
            gc = gb * tau
            win = "greedy" if gc < cs_min else ("concept" if gc > cs_min else "tie")
            tau_rows[str(tau)] = {"greedy_cost": gc, "concept_cost": cs_min, "winner": win}
            print(f"      {tau:>6} {gc:>12} {cs_min:>13} {win:>10}")
        p4 = any(v["winner"] == "greedy" for v in tau_rows.values())
    else:
        p4 = False
    print(
        f"    P4: {'MET' if p4 else 'NOT MET'}"
        f"{'' if p4 else '  <-- no cost advantage: no design contribution'}"
    )
    summary["P4"] = {
        "greedy_budget": gb,
        "concept_min_labels": cs_min,
        "tau_table": tau_rows,
        "met": bool(p4),
    }

    print("\nP5  shift-invariant distractors change Acc(C) by < 0.05")
    b0, dd = stat(k, "base_only", 0), stat(k, "distractors_only", 2)
    if b0 and dd:
        diff = abs(dd["acc_c"] - b0["acc_c"])
        print(
            f"      base only {b0['acc_c']:.3f} vs distractors {dd['acc_c']:.3f}  |Δ| = {diff:.3f}"
        )
        p5 = diff < 0.05
        print(f"    P5: {'MET' if p5 else 'NOT MET'}")
        summary["P5"] = {"diff": diff, "met": bool(p5)}
        info1 = stat(k, "information_greedy", 1)
        if info1:
            print(f"      (information-greedy at budget 1 chose: {info1['chosen']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
