#!/usr/bin/env python
"""Analyse E5 exactly as preregistered in ``paper/preregistration_e5.md``.

Four confirmatory tests. P3 is the one that decides whether H4 -- killed by E3 --
is conditionally revived for large concept spaces.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from neurogenesis.stats import bootstrap_mean, spearman

GATE = 0.99
GROUNDED = 0.9
TAUS = (25, 50, 100, 200, 400)
COMPOSITE = (6, 10, 12, 15, 20, 24, 30)
PRIME_POWER = (8, 16)
LABEL_BUDGETS = (10, 20, 40, 80, 160, 320, 640, 1280)
RULE_BUDGETS = (0, 1, 2, 3)


def load(store: Path) -> list[dict]:
    with open(store) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def main() -> None:  # noqa: C901 - linear reporting script
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("results/runs/e5.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/e5_summary.json"))
    ap.add_argument("--figure", type=Path, default=Path("results/figures/e5.png"))
    args = ap.parse_args()

    runs = [r for r in load(args.store) if r.get("experiment") == "e5"]
    if not runs:
        raise SystemExit("no E5 runs found")
    kept = [r for r in runs if r["acc_y"] >= GATE]
    print(
        f"E5: {len(runs)} runs; absolute gate Acc(Y) >= {GATE} keeps {len(kept)} "
        f"({len(kept) / len(runs):.1%})\n"
    )

    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in kept:
        cells[(r["k"], r["arm"], r["budget"])].append(r)

    def cell(k: int, arm: str, budget: int):
        rs = cells.get((k, arm, budget), [])
        if not rs:
            return None
        acc = np.array([r["acc_c"] for r in rs])
        return {
            "n": len(rs),
            "mean": float(acc.mean()),
            "est": bootstrap_mean(acc),
            "grounded": float((acc > GROUNDED).mean()),
            "rs_count": rs[0]["rs_count"],
            "values": acc.tolist(),
        }

    def min_budget(k: int, arm: str, grid) -> int | None:
        """Smallest budget on the *fixed* grid whose mean Acc(C) clears the bar."""
        for b in grid:
            c = cell(k, arm, b)
            if c and c["mean"] > GROUNDED:
                return b
        return None

    # ---- the two cost curves ------------------------------------------------
    print("COST CURVES  (L = concept labels to ground, R = rules to ground)")
    print(
        f"{'k':>4} {'factors':>10} {'L(k)':>8} {'R(k)':>6} {'|RS| base':>10}  label-arm Acc(C) by budget"
    )
    rows = {}
    for k in sorted(COMPOSITE + PRIME_POWER):
        L = min_budget(k, "labels", LABEL_BUDGETS)
        R = min_budget(k, "rules", RULE_BUDGETS)
        base = cell(k, "rules", 0)
        curve = " ".join(
            f"{(cell(k, 'labels', b) or {}).get('mean', float('nan')):.2f}" for b in LABEL_BUDGETS
        )
        fac = _factor_str(k)
        rows[k] = {
            "L": L,
            "R": R,
            "L_censored": L is None,
            "base_rs": base["rs_count"] if base else None,
            "label_curve": {
                str(b): (cell(k, "labels", b) or {}).get("mean") for b in LABEL_BUDGETS
            },
            "rule_curve": {str(b): (cell(k, "rules", b) or {}).get("mean") for b in RULE_BUDGETS},
        }
        Ls = f">{LABEL_BUDGETS[-1]}" if L is None else str(L)
        Rs = "never" if R is None else str(R)
        print(f"{k:>4} {fac:>10} {Ls:>8} {Rs:>6} {rows[k]['base_rs']:>10}  {curve}")

    summary: dict = {"n_runs": len(runs), "n_gated": len(kept), "rows": rows}

    # ---- P1: labels scale with vocabulary -----------------------------------
    comp = [k for k in COMPOSITE if rows[k]["L"] is not None]
    ks = np.array(comp, dtype=float)
    Ls = np.array([rows[k]["L"] for k in comp], dtype=float)
    n_cens = sum(1 for k in COMPOSITE if rows[k]["L"] is None)
    print(
        f"\nP1  L(k) increases with k  (composite k only; {n_cens} censored at >{LABEL_BUDGETS[-1]})"
    )
    if len(comp) >= 3:
        s = spearman(ks, Ls)
        ratio = rows[30]["L"] / rows[6]["L"] if rows[6]["L"] and rows[30]["L"] else float("nan")
        print(f"      Spearman(k, L) = {s:+.3f}   (predicted > 0.8)")
        print(f"      L(30)/L(6) = {ratio:.2f}   (predicted >= 4)")
        p1 = s > 0.8 and ratio >= 4
        print(f"    P1: {'MET' if p1 else 'NOT MET'}")
        summary["P1"] = {"spearman": s, "ratio_30_over_6": ratio, "met": bool(p1)}
    else:
        print("      too few uncensored points to test")
        summary["P1"] = {"met": None}

    # ---- P2: rules do not ---------------------------------------------------
    r_vals = {k: rows[k]["R"] for k in COMPOSITE}
    p2 = all(v == 2 for v in r_vals.values())
    print("\nP2  R(k) = 2 for every composite k")
    print(f"      {r_vals}")
    print(f"    P2: {'MET' if p2 else 'NOT MET'}")
    summary["P2"] = {"R_by_k": r_vals, "met": bool(p2)}

    # ---- P3: the crossover --------------------------------------------------
    print("\nP3  crossover: is 2 rules cheaper than L(k) labels?")
    print(f"      {'tau':>6} {'rule cost':>10}  crossover k (first k where L(k) > 2*tau)")
    tau_rows = {}
    for tau in TAUS:
        rule_cost = 2 * tau
        cross = None
        for k in COMPOSITE:
            L = rows[k]["L"]
            eff = LABEL_BUDGETS[-1] + 1 if L is None else L  # censored counts as larger
            if eff > rule_cost:
                cross = k
                break
        tau_rows[str(tau)] = {"rule_cost": rule_cost, "crossover_k": cross}
        print(f"      {tau:>6} {rule_cost:>10}  {cross if cross else 'none within k <= 30'}")
    p3 = tau_rows["100"]["crossover_k"] is not None
    print(f"    P3 (crossover exists at tau=100, k <= 30): {'MET' if p3 else 'NOT MET'}")
    if not p3:
        print("        -> no cost advantage even at large k: H4 stays dead")
    summary["P3"] = {"tau_table": tau_rows, "met": bool(p3)}

    # ---- P4: the declared structural limit ----------------------------------
    print("\nP4  prime powers cannot be grounded by rules, but can by labels")
    ok = True
    for k in PRIME_POWER:
        R, L = rows[k]["R"], rows[k]["L"]
        best = max((cell(k, "rules", b) or {"mean": 0.0})["mean"] for b in RULE_BUDGETS)
        ok &= R is None and L is not None
        print(
            f"      k={k:<3} rules: {'never' if R is None else R} (best Acc(C) {best:.3f})"
            f"   labels: {'>' + str(LABEL_BUDGETS[-1]) if L is None else L}"
        )
    print(f"    P4: {'MET' if ok else 'NOT MET'}")
    summary["P4"] = {"met": bool(ok)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out}")
    _figure(rows, args.figure)


def _factor_str(n: int) -> str:
    out, d, m = [], 2, n
    while d * d <= m:
        e = 0
        while m % d == 0:
            m //= d
            e += 1
        if e:
            out.append(f"{d}^{e}" if e > 1 else str(d))
        d += 1
    if m > 1:
        out.append(str(m))
    return "*".join(out)


def _figure(rows: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp = [k for k in COMPOSITE if rows[k]["L"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    ax.plot(
        comp,
        [rows[k]["L"] for k in comp],
        "o-",
        color="#C44E52",
        label="concept labels to ground,  L(k)",
    )
    ax.plot(
        comp,
        [rows[k]["R"] for k in comp],
        "s-",
        color="#4C72B0",
        label="auxiliary rules to ground,  R(k)",
    )
    for tau in (50, 100):
        ax.axhline(2 * tau, ls=":", lw=1, color="grey")
        ax.text(comp[-1], 2 * tau, f"  2 rules @ τ={tau}", va="center", fontsize=8, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("concept vocabulary size  k")
    ax.set_ylabel("cost to ground (log scale)")
    ax.set_title("E5: does the cost comparison flip with vocabulary size?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
