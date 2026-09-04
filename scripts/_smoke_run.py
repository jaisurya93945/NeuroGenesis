#!/usr/bin/env python
"""One short training run, used by ``scripts/smoke.sh`` to check the pipeline end to end.

Uses plain integer addition rather than the modular family. The convergence
diagnostic (``RESULTS.md`` §6) found that identifiable *modular* tasks need a full
training budget to escape a degenerate optimum, so a short modular run makes a
flaky smoke test -- it fails for a real reason that has nothing to do with whether
the pipeline works. Addition is identifiable too and converges quickly.
"""

from neurogenesis.config import DataSpec, OptimSpec, RunConfig, TaskSpec
from neurogenesis.runner import run

cfg = RunConfig(
    experiment="smoke",
    task=TaskSpec(family="addition", weights=(1, 1), m=10, k=10),
    data=DataSpec(tier="M", n_train=8000, n_val=1000, n_test=1000),
    optim=OptimSpec(epochs=4),
)
rec = run(cfg, store=None)
m = rec["metrics"]["test"]
print(
    f"|RS|={rec['oracle']['rs_count']}  Acc(Y)={m['acc_y']:.3f}  "
    f"Acc(C)={m['acc_c']:.3f}  {rec['runtime_s']:.0f}s"
)
assert rec["oracle"]["rs_count"] == 1, "addition must be identifiable"
assert m["acc_y"] > 0.5, f"smoke run barely learned: Acc(Y)={m['acc_y']:.3f}"
print("SMOKE OK")
