#!/usr/bin/env python
"""H7 triage: are out-of-set relabellings a real gap, or an artefact?

E2 found that only 78.4% of converged non-grounding runs recover an ``alpha_hat``
inside the deterministic RS set, against 100% in E1's narrow family. Before that
can be called a gap in the theory, the cheap explanations have to be eliminated.

Three competing accounts, and what separates them:

1. **Estimator artefact.** ``alpha_hat(c)`` is a *mode* over test examples. If the
   encoder is not implementing any deterministic map, the mode is a poor summary and
   landing outside ``RS`` says nothing. Signature: low ``alpha_hat_coverage``, or
   out-of-set runs concentrated at low ``Acc(Y)``.
2. **Approximate shortcut.** The model found a relabelling that violates the
   constraint on only a *little* probability mass -- a member of
   ``RS_eps = {alpha : violation <= eps}`` for small eps, which the deterministic
   theory excludes by construction. Signature: violation masses clustered near zero.
3. **Just wrong.** The run is unconverged or the encoder is incoherent.
   Signature: violation masses spread out and large.

Only (2) makes H7 a finding. This script measures the violation mass of every
recovered ``alpha_hat`` directly, which is what distinguishes them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurogenesis.config import RunConfig, TaskSpec
from neurogenesis.runner import build_task, load_runs
from neurogenesis.stats import bootstrap_mean

GATE_RATIO = 0.95


def _spec_from_cfg(cfg: dict) -> TaskSpec:
    t = cfg["task"]
    return TaskSpec(
        family=t["family"],
        weights=tuple(t["weights"]),
        m=t["m"],
        k=t["k"],
        n_slots=t.get("n_slots", 2),
        support_density=t.get("support_density", 1.0),
        planted_kind=t.get("planted_kind", "cyclic"),
        rarity=t.get("rarity", 1.0),
        swap=tuple(t.get("swap", (0, 1))),
        seed=t["seed"],
    )


def violation_mass(task, alpha: np.ndarray) -> float:
    """Fraction of support probability mass whose label ``alpha`` fails to preserve.

    Zero exactly when ``alpha`` is a deterministic shortcut. This is the same
    quantity the margin minimises, evaluated at the *recovered* map instead.
    """
    alpha = np.asarray(alpha, dtype=np.int64)
    mapped = alpha[task.support.astype(np.int64)]
    changed = task.label_of(mapped) != task.support_labels
    return float(task.support_weights[changed].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("results/runs/runs.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/h7_diagnosis.json"))
    args = ap.parse_args()

    runs = [r for r in load_runs(args.store) if r.get("experiment") == "e2" and "error" not in r]
    by_task: dict[str, list[dict]] = {}
    for r in runs:
        by_task.setdefault(r["task"]["content_hash"], []).append(r)

    task_cache: dict[str, object] = {}
    rows = []
    n_trunc = 0
    for key, recs in by_task.items():
        if recs[0]["oracle"]["truncated"]:
            # `contains` is unreliable on a truncated enumeration: it can report
            # False for a genuine shortcut that fell outside the first `limit` maps.
            n_trunc += 1
            continue
        accs_y = np.array([r["metrics"]["test"]["acc_y"] for r in recs])
        ref = accs_y.max()
        for r, ay in zip(recs, accs_y, strict=True):
            if ay < GATE_RATIO * ref:
                continue  # not converged -- excluded upstream too
            m = r["metrics"]["test"]
            if m["acc_c"] > 0.9:
                continue  # grounded runs are not the question
            if key not in task_cache:
                task_cache[key] = build_task(RunConfig(task=_spec_from_cfg(r["config"])))
            task = task_cache[key]
            rows.append(
                {
                    "family": r["config"]["task"]["family"],
                    "rs_count": r["oracle"]["rs_count"],
                    "in_rs": bool(m["rs_membership"]),
                    "violation": violation_mass(task, m["alpha_hat"]),
                    "acc_y": ay,
                    "acc_c": m["acc_c"],
                    "coverage": m["alpha_hat_coverage"],
                    "k": r["task"]["k"],
                }
            )

    if not rows:
        raise SystemExit("no converged non-grounding runs found")

    inside = [r for r in rows if r["in_rs"]]
    outside = [r for r in rows if not r["in_rs"]]
    print(f"tasks skipped (oracle truncated, membership undefined): {n_trunc}")
    print(f"converged non-grounding runs: {len(rows)}")
    print(f"  alpha_hat INSIDE  RS: {len(inside)} ({len(inside) / len(rows):.1%})")
    print(f"  alpha_hat OUTSIDE RS: {len(outside)} ({len(outside) / len(rows):.1%})\n")

    # --- explanation 1: estimator artefact -------------------------------
    cov_out = np.array([r["coverage"] for r in outside])
    cov_in = np.array([r["coverage"] for r in inside])
    print("1. ESTIMATOR ARTEFACT?")
    print(
        f"   alpha_hat coverage, outside: {cov_out.min():.3f}–{cov_out.max():.3f} "
        f"(mean {cov_out.mean():.3f})"
    )
    print(
        f"   alpha_hat coverage, inside : {cov_in.min():.3f}–{cov_in.max():.3f} "
        f"(mean {cov_in.mean():.3f})"
    )
    ay_out = np.array([r["acc_y"] for r in outside])
    ay_in = np.array([r["acc_y"] for r in inside])
    print(f"   Acc(Y) outside: mean {ay_out.mean():.4f} | inside: mean {ay_in.mean():.4f}")
    artefact = cov_out.mean() < 0.99 or ay_out.mean() < ay_in.mean() - 0.05
    print(f"   -> supported: {artefact}\n")

    # --- explanations 2 vs 3: how close to being a shortcut? -------------
    viol = np.array([r["violation"] for r in outside])
    print("2 vs 3. APPROXIMATE SHORTCUT, OR JUST WRONG?")
    print("   violation mass of the recovered map (0 = exact shortcut):")
    for q in (10, 25, 50, 75, 90):
        print(f"     p{q:<3d} = {np.percentile(viol, q):.4f}")
    print(f"   mean = {bootstrap_mean(viol)}")
    for eps in (0.001, 0.01, 0.05, 0.10, 0.25):
        frac = float((viol <= eps).mean())
        print(f"   within RS_eps for eps={eps:<5}: {frac:.1%} of out-of-set maps")
    print()

    # --- the decisive comparison: stratify by how well the run fit the label ---
    # "Outside RS" turns out to track incomplete convergence, so a verdict that
    # ignores Acc(Y) would credit the theory-gap story with runs that simply had
    # not finished learning. Restricting to near-perfect label fit removes that.
    print("STRATIFIED BY CONVERGENCE (the decisive comparison)")
    print(f"   {'Acc(Y) >=':>10} {'n':>5} {'outside RS':>11} {'median viol':>12}")
    strat = {}
    for thr in (0.0, 0.95, 0.99, 0.999):
        sub = [r for r in rows if r["acc_y"] >= thr]
        if not sub:
            continue
        out = [r for r in sub if not r["in_rs"]]
        med = float(np.median([r["violation"] for r in out])) if out else float("nan")
        strat[str(thr)] = {"n": len(sub), "frac_outside": len(out) / len(sub), "median_viol": med}
        print(f"   {thr:>10.3f} {len(sub):>5} {len(out) / len(sub):>10.1%} {med:>12.4f}")

    frac_hi = strat.get("0.999", {}).get("frac_outside", float("nan"))
    survives = frac_hi >= 0.15
    verdict = (
        "H7 SURVIVES: out-of-set maps persist among fully converged runs"
        if survives
        else "H7 DOES NOT SURVIVE: out-of-set maps are explained by incomplete convergence"
    )
    print(f"\nVERDICT: {verdict}")
    print(f"  Among runs with Acc(Y) >= 0.999, only {frac_hi:.1%} land outside RS,")
    print(f"  versus {strat['0.0']['frac_outside']:.1%} unrestricted. Out-of-set runs average")
    print(f"  Acc(Y) = {ay_out.mean():.4f} against {ay_in.mean():.4f} for in-set runs.")

    near = float((viol <= 0.05).mean())
    summary = {
        "n_converged_nongrounding": len(rows),
        "n_inside": len(inside),
        "n_outside": len(outside),
        "frac_inside": len(inside) / len(rows),
        "coverage_outside_mean": float(cov_out.mean()),
        "coverage_inside_mean": float(cov_in.mean()),
        "acc_y_outside_mean": float(ay_out.mean()),
        "acc_y_inside_mean": float(ay_in.mean()),
        "violation_percentiles": {
            str(q): float(np.percentile(viol, q)) for q in (10, 25, 50, 75, 90)
        },
        "frac_within_eps": {
            str(e): float((viol <= e).mean()) for e in (0.001, 0.01, 0.05, 0.10, 0.25)
        },
        "frac_near_shortcut_eps05": near,
        "stratified_by_convergence": strat,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
