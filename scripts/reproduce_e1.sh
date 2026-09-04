#!/usr/bin/env bash
# Reproduce E1 end to end: 40 runs, then the table and figure.
# ~15 min on 4 CPU cores. Resumable -- reruns skip configs already in the store.
set -euo pipefail
PY="${PY:-.venv/bin/python}"
WORKERS="${WORKERS:-4}"

$PY scripts/download_mnist.py
$PY experiments/e1_identifiability.py --workers "$WORKERS" --seeds 10
$PY scripts/analyze_e1.py
echo
echo "Results: results/e1_summary.json  |  Figure: results/figures/e1.png"
echo "Predictions were fixed in advance: paper/preregistration.md"
