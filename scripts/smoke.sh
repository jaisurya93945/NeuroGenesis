#!/usr/bin/env bash
# End-to-end smoke test: a few minutes on 4 CPU cores, proves the pipeline works.
set -euo pipefail
PY="${PY:-.venv/bin/python}"

echo "== 1. oracle correctness gates =="
$PY -m pytest tests/test_oracle_analytic.py tests/test_oracle_vs_loss.py -q

echo "== 2. data =="
$PY scripts/download_mnist.py

echo "== 3. one short training run =="
$PY - <<'PYEOF'
import numpy as np
from neurogenesis.config import DataSpec, ModelSpec, OptimSpec, RunConfig, TaskSpec
from neurogenesis.runner import run

cfg = RunConfig(
    experiment="smoke",
    task=TaskSpec(family="algebraic", weights=(1, 2), m=10),
    data=DataSpec(tier="M", n_train=4000, n_val=1000, n_test=1000),
    optim=OptimSpec(epochs=3),
)
rec = run(cfg, store=None)
m = rec["metrics"]["test"]
print(f"|RS|={rec['oracle']['rs_count']}  Acc(Y)={m['acc_y']:.3f}  "
      f"Acc(C)={m['acc_c']:.3f}  {rec['runtime_s']:.0f}s")
assert m["acc_y"] > 0.2, "smoke run learned nothing at all"
print("SMOKE OK")
PYEOF
