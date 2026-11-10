#!/usr/bin/env bash
# Author: Zwe Min Htet Aung
# Run every controller under every test condition on the fixed test seeds, then summarize.
# Usage (from catch_it/, env active):
#   bash run_eval_matrix.sh                                   # classical + oracle + random
#   CHECKPOINT=outputs/<run>/nn/best.pth bash run_eval_matrix.sh   # also the RL policy
# Options via env vars: SEEDS (default 10000:11000), WORKERS (default 16)
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-10000:11000}
WORKERS=${WORKERS:-16}
CONDITIONS="nominal noise_x2 noise_x4 tracker_noise gravity_fixed delay_1 delay_2"
CONTROLLERS="oracle classical_single classical_multi classical_perfect"

run() { python eval_catch.py --seeds "$SEEDS" --workers "$WORKERS" "$@" 2>&1 | tail -n 2 | head -n 1; }

run --controller random --condition nominal
for cond in $CONDITIONS; do
  for c in $CONTROLLERS; do run --controller "$c" --condition "$cond"; done
  if [ -n "${CHECKPOINT:-}" ]; then run --controller rl --checkpoint "$CHECKPOINT" --condition "$cond"; fi
done
python analyze_results.py --markdown results/summary.md > /dev/null
echo "summary: results/summary.md"
