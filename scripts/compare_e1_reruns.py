#!/usr/bin/env python
"""Compare E1 before and after the seeding fix.

The original 40 E1 runs were produced while ``init_seed`` did not control weight
initialisation (see ``REPRODUCIBILITY.md``). They are archived rather than deleted,
and E1 was re-run with correct seeding. This script puts the two side by side.

The point is not to pick the nicer answer. It is to show what a genuine
reproducibility defect did and did not change, so a reader can judge how much the
conclusion depended on it. If the two disagree materially, the pre-fix numbers were
load-bearing and that must be said plainly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neurogenesis.runner import load_runs
from neurogenesis.stats import bootstrap_mean

CONDITIONS = [1, 2, 5, 10]
GATE_RATIO = 0.95


def summarise(runs: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for rs in CONDITIONS:
        recs = [r for r in runs if r.get("experiment") == "e1" and r["oracle"]["rs_count"] == rs]
        if not recs:
            continue
        accs_y = np.array([r["metrics"]["test"]["acc_y"] for r in recs])
        ref = accs_y.max()
        keep = [r for r, a in zip(recs, accs_y, strict=True) if a >= GATE_RATIO * ref]
        acc_c = np.array([r["metrics"]["test"]["acc_c"] for r in keep])
        est = bootstrap_mean(acc_c) if len(acc_c) else None
        out[rs] = {
            "n_total": len(recs),
            "n_gated": len(keep),
            "acc_c_mean": est.mean if est else float("nan"),
            "acc_c_lo": est.lo if est else float("nan"),
            "acc_c_hi": est.hi if est else float("nan"),
            "frac_grounded": float((acc_c > 0.95).mean()) if len(acc_c) else float("nan"),
            "frac_in_rs": float(
                np.mean([bool(r["metrics"]["test"]["rs_membership"]) for r in keep])
            )
            if keep
            else float("nan"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, default=Path("results/runs/archived_e1_preseedfix.jsonl"))
    ap.add_argument("--new", type=Path, default=Path("results/runs/runs.jsonl"))
    args = ap.parse_args()

    old = summarise(load_runs(args.old))
    new = summarise(load_runs(args.new))
    if not new:
        raise SystemExit("no post-fix E1 runs found")

    print("E1 before vs after the seeding fix\n")
    print(f"{'|RS|':>5}  {'gated (old→new)':>16}  {'Acc(C) old':>24}  {'Acc(C) new':>24}  {'Δ':>7}")
    print("-" * 88)
    for rs in CONDITIONS:
        o, n = old.get(rs), new.get(rs)
        if not n:
            continue
        o_txt = f"{o['acc_c_mean']:.4f} [{o['acc_c_lo']:.3f},{o['acc_c_hi']:.3f}]" if o else "—"
        n_txt = f"{n['acc_c_mean']:.4f} [{n['acc_c_lo']:.3f},{n['acc_c_hi']:.3f}]"
        gates = f"{o['n_gated'] if o else '—'}/{o['n_total'] if o else '—'} → {n['n_gated']}/{n['n_total']}"
        delta = f"{n['acc_c_mean'] - o['acc_c_mean']:+.4f}" if o else "—"
        print(f"{rs:>5}  {gates:>16}  {o_txt:>24}  {n_txt:>24}  {delta:>7}")

    print(f"\n{'|RS|':>5}  {'grounded old→new':>18}  {'α̂ ∈ RS old→new':>18}")
    for rs in CONDITIONS:
        o, n = old.get(rs), new.get(rs)
        if not n:
            continue
        g = f"{o['frac_grounded']:.2f} → {n['frac_grounded']:.2f}" if o else "—"
        m = f"{o['frac_in_rs']:.2f} → {n['frac_in_rs']:.2f}" if o else "—"
        print(f"{rs:>5}  {g:>18}  {m:>18}")

    ordered = [new[rs]["acc_c_mean"] for rs in CONDITIONS if rs in new]
    mono = all(ordered[i] >= ordered[i + 1] - 1e-9 for i in range(len(ordered) - 1))
    print(f"\npost-fix monotone non-increasing in |RS|: {'YES' if mono else 'NO'}")
    if 1 in new and 10 in new:
        print(
            f"post-fix P2 contrast |RS|=1 − |RS|=10: "
            f"{new[1]['acc_c_mean'] - new[10]['acc_c_mean']:+.4f}  (predicted > 0.5)"
        )


if __name__ == "__main__":
    main()
