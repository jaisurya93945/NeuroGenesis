#!/usr/bin/env bash
# Regenerate every table and figure from the committed run records.
# Does NOT retrain anything -- it reads results/runs/*.jsonl and takes seconds.
#
# To re-run the experiments themselves (hours, 4 CPU cores), see the commands at
# the bottom of this file.
set -euo pipefail
PY="${PY:-.venv/bin/python}"

echo "== correctness gates (these protect every number below) =="
$PY -m pytest tests/test_oracle_analytic.py tests/test_oracle_vs_loss.py \
              tests/test_determinism.py -q

echo
echo "== E1: does |RS| predict grounding? =="
$PY scripts/analyze_e1.py

echo
echo "== E2: does the margin predict it better? =="
$PY scripts/analyze_e2.py

echo
echo "== H7 triage: is the membership shortfall real? =="
$PY scripts/diagnose_h7.py

echo
echo "== E3: does selection beat the baselines on cost? =="
$PY scripts/analyze_e3.py

for e in e4 e5; do
  if [ -f "results/runs/$e.jsonl" ]; then
    echo
    echo "== ${e^^} =="
    $PY "scripts/analyze_$e.py"
  fi
done

if [ -f "results/runs/e6.jsonl" ]; then
  echo
  echo "== E6: does the negative survive real perception (Tier M, MNIST)? =="
  $PY scripts/analyze_e6_tierm.py
fi

echo
echo "== figures =="
$PY scripts/make_all_figures.py

cat <<'NOTE'

Done. Tables above, figures in results/figures/.

To re-run the experiments from scratch (hours, not minutes):
  python experiments/e1_identifiability.py  --workers 4 --seeds 10
  python experiments/e2_margin_vs_binary.py --workers 4 --seeds 5
  python experiments/e3_selection.py        --workers 4 --seeds 8
  python experiments/e4_continual.py        --workers 4 --seeds 8
  python experiments/e5_cost_scaling.py     --workers 4 --seeds 8
  python experiments/e3_selection.py --tier M --ks 6 --seeds 8 \
         --concept-budgets 2 5 10 15 25 50 --workers 4 --store results/runs/e6.jsonl
NOTE
