#!/usr/bin/env python
"""Regenerate every figure in the paper from the committed run records.

The only path from data to figures. Nothing is drawn by hand, and nothing is
redrawn from a summary that was itself typed by hand.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FIGURES = [
    ("E1  grounding vs shortcut count", "scripts/analyze_e1.py", "results/figures/e1.png"),
    ("E2  margin vs binary identifiability", "scripts/analyze_e2.py", "results/figures/e2.png"),
    ("E5  cost curves vs vocabulary size", "scripts/analyze_e5.py", "results/figures/e5.png"),
]


def main() -> int:
    py = sys.executable
    made, skipped = [], []
    for name, script, out in FIGURES:
        if not Path(script).exists():
            skipped.append((name, "analysis script missing"))
            continue
        proc = subprocess.run([py, script], capture_output=True, text=True)
        if proc.returncode != 0:
            reason = (proc.stderr.strip().splitlines() or ["failed"])[-1]
            skipped.append((name, reason))
            continue
        made.append((name, out)) if Path(out).exists() else skipped.append((name, "no figure"))

    for name, out in made:
        print(f"  OK    {name:44s} -> {out}")
    for name, why in skipped:
        print(f"  SKIP  {name:44s}    {why}")
    print(f"\n{len(made)} figure(s) regenerated, {len(skipped)} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
