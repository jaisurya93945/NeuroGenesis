#!/usr/bin/env python
"""Analyse E1 exactly as preregistered. Produces the table and figure from raw JSONL.

No number in the paper is typed by hand: this script is the only path from run
records to reported results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurogenesis.runner import DEFAULT_STORE, load_runs
from neurogenesis.stats import (
    bootstrap_diff,
    bootstrap_mean,
    cliffs_delta,
    permutation_trend_test,
)

CONDITION_ORDER = [1, 2, 5, 10]  # |RS|, ascending -- the preregistered ordering
GATE_RATIO = 0.95


def collect(runs: list[dict]) -> dict[int, list[dict]]:
    """Group E1 runs by oracle |RS|."""
    out: dict[int, list[dict]] = {r: [] for r in CONDITION_ORDER}
    for rec in runs:
        if rec.get("experiment") != "e1":
            continue
        rs = rec["oracle"]["rs_count"]
        if rs in out:
            out[rs].append(rec)
    return out


def apply_gate(recs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Preregistered convergence gate: Acc(Y) >= 0.95 * best Acc(Y) in this condition."""
    if not recs:
        return [], []
    accs = np.array([r["metrics"]["test"]["acc_y"] for r in recs])
    ref = accs.max()
    keep = [r for r, a in zip(recs, accs, strict=True) if a >= GATE_RATIO * ref]
    drop = [r for r, a in zip(recs, accs, strict=True) if a < GATE_RATIO * ref]
    return keep, drop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--out", type=Path, default=Path("results/e1_summary.json"))
    ap.add_argument("--figure", type=Path, default=Path("results/figures/e1.png"))
    args = ap.parse_args()

    groups = collect(load_runs(args.store))
    total = sum(len(v) for v in groups.values())
    if total == 0:
        raise SystemExit("no E1 runs found; run experiments/e1_identifiability.py first")

    print(f"E1: {total} runs\n")
    print(
        f"{'|RS|':>5} {'n':>3} {'excl':>5}  {'Acc(C) mean [95% CI]':<30} "
        f"{'Acc(Y) mean':<12} {'grounded':>8} {'in_RS':>7}"
    )
    print("-" * 88)

    summary: dict[str, dict] = {}
    acc_c_by_rs: dict[int, np.ndarray] = {}
    for rs in CONDITION_ORDER:
        keep, drop = apply_gate(groups[rs])
        if not keep:
            print(f"{rs:>5} {0:>3} {len(drop):>5}  (no surviving runs)")
            continue
        acc_c = np.array([r["metrics"]["test"]["acc_c"] for r in keep])
        acc_y = np.array([r["metrics"]["test"]["acc_y"] for r in keep])
        grounded = float((acc_c > 0.95).mean())
        memb = [r["metrics"]["test"]["rs_membership"] for r in keep]
        in_rs = float(np.mean([bool(m) for m in memb]))
        est = bootstrap_mean(acc_c)
        acc_c_by_rs[rs] = acc_c
        summary[str(rs)] = {
            "n": len(keep),
            "n_excluded": len(drop),
            "acc_c_mean": est.mean,
            "acc_c_lo": est.lo,
            "acc_c_hi": est.hi,
            "acc_c_values": acc_c.tolist(),
            "acc_y_mean": float(acc_y.mean()),
            "acc_y_values": acc_y.tolist(),
            "frac_grounded": grounded,
            "frac_alpha_hat_in_rs": in_rs,
            "alpha_hats": [r["metrics"]["test"]["alpha_hat"] for r in keep],
        }
        print(
            f"{rs:>5} {len(keep):>3} {len(drop):>5}  {str(est):<30} "
            f"{acc_y.mean():<12.4f} {grounded:>8.2f} {in_rs:>7.2f}"
        )

    print()
    if 1 in acc_c_by_rs and 10 in acc_c_by_rs:
        d = bootstrap_diff(acc_c_by_rs[1], acc_c_by_rs[10])
        delta = cliffs_delta(acc_c_by_rs[1], acc_c_by_rs[10])
        summary["contrast_1_vs_10"] = {
            "diff": d.mean,
            "lo": d.lo,
            "hi": d.hi,
            "cliffs_delta": delta,
        }
        print(f"P2  Acc(C)[|RS|=1] - Acc(C)[|RS|=10] = {d}")
        print(f"    Cliff's delta = {delta:+.3f}")
        print(f"    P2 predicted difference > 0.5:  {'MET' if d.mean > 0.5 else 'NOT MET'}")

    ordered = [acc_c_by_rs[r] for r in CONDITION_ORDER if r in acc_c_by_rs]
    if len(ordered) >= 2:
        means = [float(g.mean()) for g in ordered]
        monotone = all(means[i] >= means[i + 1] - 1e-9 for i in range(len(means) - 1))
        p = permutation_trend_test(ordered, n_perm=20_000)
        summary["trend"] = {"means": means, "monotone_decreasing": monotone, "perm_p": p}
        print(f"\nP1  condition means (|RS| ascending): {[f'{m:.4f}' for m in means]}")
        print(f"    monotone non-increasing: {'MET' if monotone else 'NOT MET'}")
        print(f"    permutation trend p = {p:.5f}   (reported once; no stars)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    _figure(summary, args.figure)


def _figure(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(int(k), v) for k, v in summary.items() if k.isdigit()]
    if not rows:
        return
    rows.sort()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    rng = np.random.default_rng(0)

    for ax, key, title, ylab in (
        (ax1, "acc_c_values", "Concept accuracy vs shortcut count", "test Acc(C)"),
        (ax2, "acc_y_values", "Label accuracy (should be flat)", "test Acc(Y)"),
    ):
        for i, (_rs, v) in enumerate(rows):
            vals = np.array(v[key])
            ax.scatter(
                np.full(len(vals), i) + rng.uniform(-0.09, 0.09, len(vals)),
                vals,
                s=26,
                alpha=0.75,
                color="#4C72B0",
                zorder=3,
            )
            ax.hlines(vals.mean(), i - 0.25, i + 0.25, color="#C44E52", lw=2.5, zorder=4)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"|RS|={rs}" for rs, _ in rows])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("E1: does provable identifiability predict symbol grounding?", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
