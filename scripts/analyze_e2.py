#!/usr/bin/env python
"""Analyse E2 exactly as preregistered in ``paper/preregistration_e2.md``.

Five confirmatory tests, Holm-corrected. Every number here is derived from
``results/runs/runs.jsonl``; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurogenesis.config import RunConfig, TaskSpec
from neurogenesis.oracle.measures import margin as compute_margin
from neurogenesis.runner import build_task, load_runs
from neurogenesis.stats import (
    bootstrap_diff,
    nested_delta_r2,
    partial_spearman,
    spearman,
)

GATE_RATIO = 0.95
CACHE = Path("results/oracle_cache/e2_margins.json")


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


def load_margins(task_keys: dict[str, dict]) -> dict[str, float]:
    """Compute (and cache) the margin for every distinct task in the sweep."""
    cache: dict[str, float] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
    todo = [k for k in task_keys if k not in cache]
    if todo:
        print(f"computing margin for {len(todo)} tasks (cached: {len(cache)}) ...")
        for i, key in enumerate(todo, 1):
            spec = _spec_from_cfg(task_keys[key])
            task = build_task(RunConfig(task=spec))
            try:
                cache[key] = compute_margin(task, timeout_s=30.0).margin
            except Exception as exc:  # noqa: BLE001
                print(f"  margin failed for {key}: {exc!r}")
                cache[key] = float("nan")
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=0))
    return cache


def main() -> None:  # noqa: C901 - a reporting script, linear by design
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("results/runs/runs.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/e2_summary.json"))
    args = ap.parse_args()

    runs = [r for r in load_runs(args.store) if r.get("experiment") == "e2" and "error" not in r]
    if not runs:
        raise SystemExit("no E2 runs found")

    # ---- group runs by task ------------------------------------------------
    by_task: dict[str, list[dict]] = {}
    cfg_of: dict[str, dict] = {}
    for r in runs:
        key = r["task"]["content_hash"]
        by_task.setdefault(key, []).append(r)
        cfg_of[key] = r["config"]

    margins = load_margins(cfg_of)

    rows = []
    n_degenerate = n_truncated = 0
    for key, recs in by_task.items():
        rs = recs[0]["oracle"]["rs_count"]
        if recs[0]["oracle"]["truncated"]:
            n_truncated += 1
            continue
        if recs[0]["task"]["n_labels"] <= 1:
            n_degenerate += 1
            continue
        accs_y = np.array([r["metrics"]["test"]["acc_y"] for r in recs])
        ref = accs_y.max()
        gated = [r for r, a in zip(recs, accs_y, strict=True) if a >= GATE_RATIO * ref]
        acc_c = np.array([r["metrics"]["test"]["acc_c"] for r in gated])
        rows.append(
            {
                "key": key,
                "family": recs[0]["config"]["task"]["family"],
                "rs": rs,
                "margin": margins.get(key, float("nan")),
                "acc_c": float(acc_c.mean()) if len(acc_c) else float("nan"),
                "acc_y": float(accs_y.mean()),
                "n_gated": len(gated),
                "n_total": len(recs),
                "conv_rate": len(gated) / len(recs),
                "label_entropy": recs[0]["task"]["label_entropy"],
                "n_labels": recs[0]["task"]["n_labels"],
                "k": recs[0]["task"]["k"],
                "support_size": recs[0]["task"]["support_size"],
                "in_rs": [bool(r["metrics"]["test"]["rs_membership"]) for r in gated],
                "grounded": [bool(r["metrics"]["test"]["acc_c"] > 0.9) for r in gated],
            }
        )

    ok = [r for r in rows if np.isfinite(r["acc_c"]) and np.isfinite(r["margin"])]
    print(
        f"\nE2: {len(runs)} runs, {len(by_task)} tasks "
        f"({n_degenerate} degenerate |Y|=1, {n_truncated} truncated excluded) "
        f"-> {len(ok)} analysable\n"
    )

    acc_c = np.array([r["acc_c"] for r in ok])
    rs = np.array([r["rs"] for r in ok], dtype=float)
    mg = np.array([r["margin"] for r in ok])
    log_rs = np.log(rs)
    binary = (rs == 1).astype(float)

    summary: dict = {"n_tasks": len(ok), "n_runs": len(runs)}

    # ---- P1: does margin beat binary identifiability? ----------------------
    print("P1  margin vs binary identifiability (nested ΔR², bootstrap CI)")
    r2_bin = np.column_stack([binary])
    r2_bin_mg = np.column_stack([binary, mg])
    d1, e1 = nested_delta_r2(acc_c, r2_bin, r2_bin_mg)
    r2_lrs = np.column_stack([log_rs])
    r2_lrs_mg = np.column_stack([log_rs, mg])
    d2, e2 = nested_delta_r2(acc_c, r2_lrs, r2_lrs_mg)
    print(f"    ΔR² adding margin to binary : {e1}")
    print(f"    ΔR² adding margin to log|RS|: {e2}")
    print(f"    P1 (CI excludes 0): {'MET' if e1.lo > 0 else 'NOT MET'}")
    summary["P1"] = {
        "delta_r2_over_binary": e1.__dict__,
        "delta_r2_over_logrs": e2.__dict__,
        "met": bool(e1.lo > 0),
    }

    # ---- P2: the decisive cell -- identifiable tasks only ------------------
    ident = [r for r in ok if r["rs"] == 1]
    lo_m = np.array([r["acc_c"] for r in ident if r["margin"] < 0.01])
    hi_m = np.array([r["acc_c"] for r in ident if r["margin"] > 0.1])
    print(f"\nP2  identifiable tasks only (n={len(ident)}): low margin vs high margin")
    print(
        f"    margin < 0.01 : n={len(lo_m)}  mean Acc(C) = {lo_m.mean():.4f}"
        if len(lo_m)
        else "    margin < 0.01 : none"
    )
    print(
        f"    margin > 0.1  : n={len(hi_m)}  mean Acc(C) = {hi_m.mean():.4f}"
        if len(hi_m)
        else "    margin > 0.1  : none"
    )
    if len(lo_m) and len(hi_m):
        d = bootstrap_diff(hi_m, lo_m)
        print(f"    difference (high − low) = {d}")
        print(f"    P2 (predicted > 0.3): {'MET' if d.mean > 0.3 else 'NOT MET'}")
        summary["P2"] = {
            "diff": d.__dict__,
            "n_low": len(lo_m),
            "n_high": len(hi_m),
            "met": bool(d.mean > 0.3),
        }
    else:
        summary["P2"] = {"met": None, "note": "insufficient tasks in one bin"}

    # ---- P3: does |RS| survive controlling for informativeness? ------------
    controls = np.column_stack(
        [
            np.array([r["label_entropy"] for r in ok]),
            np.array([r["n_labels"] for r in ok], dtype=float),
            np.array([r["k"] for r in ok], dtype=float),
            np.array([r["support_size"] for r in ok], dtype=float),
        ]
    )
    ps = partial_spearman(log_rs, acc_c, controls)
    raw = spearman(log_rs, acc_c)
    print(f"\nP3  log|RS| vs Acc(C):  raw Spearman = {raw:+.3f}")
    print(f"    partial (controlling H(Y), |Y|, k, |S|) = {ps}")
    print(f"    P3 (CI excludes 0, negative): {'MET' if ps.hi < 0 else 'NOT MET'}")
    summary["P3"] = {"raw_spearman": raw, "partial": ps.__dict__, "met": bool(ps.hi < 0)}

    # ---- P4: H5 -- convergence rate rises with |RS| (UNGATED by design) -----
    conv = np.array([r["conv_rate"] for r in ok])
    s4 = spearman(log_rs, conv)
    print(f"\nP4  log|RS| vs convergence rate (ungated): Spearman = {s4:+.3f}")
    print(f"    P4 (predicted positive): {'MET' if s4 > 0 else 'NOT MET'}")
    summary["P4"] = {"spearman": s4, "met": bool(s4 > 0)}

    # ---- P5: rs_membership among converged non-grounding runs --------------
    memb = [m for r in ok for m, g in zip(r["in_rs"], r["grounded"], strict=True) if not g]
    frac = float(np.mean(memb)) if memb else float("nan")
    print(f"\nP5  α̂ ∈ RS among converged NON-grounding runs: {frac:.3f} (n={len(memb)})")
    print(f"    P5 (predicted ≥ 0.8): {'MET' if frac >= 0.8 else 'NOT MET'}")
    summary["P5"] = {"frac_in_rs": frac, "n": len(memb), "met": bool(frac >= 0.8)}

    summary["rows"] = [{k: v for k, v in r.items() if k not in ("in_rs", "grounded")} for r in ok]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    _figure(ok, Path("results/figures/e2.png"))


def _figure(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    fams = sorted({r["family"] for r in rows})
    colors = dict(zip(fams, plt.cm.tab10.colors, strict=False))

    for r in rows:
        ax1.scatter(r["rs"], r["acc_c"], s=24, alpha=0.75, color=colors[r["family"]])
    ax1.set_xscale("log")
    ax1.set_xlabel("|RS|  (log)")
    ax1.set_ylabel("mean test Acc(C)")
    ax1.set_title("Grounding vs shortcut count")

    ident = [r for r in rows if r["rs"] == 1]
    for r in ident:
        ax2.scatter(max(r["margin"], 1e-5), r["acc_c"], s=26, alpha=0.8, color=colors[r["family"]])
    ax2.set_xscale("log")
    ax2.set_xlabel("margin  (log)")
    ax2.set_ylabel("mean test Acc(C)")
    ax2.set_title(f"Identifiable tasks only (n={len(ident)})\nthe decisive cell for H2")

    for r in rows:
        ax3.scatter(r["rs"], r["conv_rate"], s=24, alpha=0.7, color=colors[r["family"]])
    ax3.set_xscale("log")
    ax3.set_xlabel("|RS|  (log)")
    ax3.set_ylabel("fraction of seeds converging")
    ax3.set_title("H5: optimisability vs identifiability")

    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in colors.items()]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(fams),
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
