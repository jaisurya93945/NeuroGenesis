#!/usr/bin/env bash
# End-to-end smoke test: a couple of minutes on 4 CPU cores, proves the pipeline works.
set -euo pipefail
PY="${PY:-.venv/bin/python}"

echo "== 1. oracle correctness gates =="
$PY -m pytest tests/test_oracle_analytic.py tests/test_oracle_vs_loss.py -q

echo "== 2. data =="
$PY scripts/download_mnist.py

echo "== 3. one short training run =="
$PY scripts/_smoke_run.py
