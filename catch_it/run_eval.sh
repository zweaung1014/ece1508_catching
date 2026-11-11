#!/usr/bin/env bash
# Author: Zwe Min Htet Aung
# Run every controller on the fixed test seeds into results/$EXP, then summarize.
# The env runs as configured in configs/env/DcmmCfg.py (every throw draws its own disturbances).
# Commit the code first, so each .json pins the commit that made the results.
# Usage (from catch_it/, env active):
#   EXP=exp03_harder bash run_eval.sh                                        # classical + oracle + random
#   EXP=exp03_harder CHECKPOINT=outputs/<run>/nn/best.pth bash run_eval.sh   # also the RL policy
# Options via env vars: SEEDS (default 10000:15000), WORKERS (default 16)
set -euo pipefail
cd "$(dirname "$0")"
: "${EXP:?set EXP to the experiment name, e.g. EXP=exp03_harder}"
SEEDS=${SEEDS:-10000:15000}
WORKERS=${WORKERS:-16}
CONTROLLERS="random oracle classical_single classical_multi classical_perfect"

run() { python eval_catch.py --exp "$EXP" --seeds "$SEEDS" --workers "$WORKERS" "$@" 2>&1 | tail -n 2 | head -n 1; }

for c in $CONTROLLERS; do run --controller "$c"; done
if [ -n "${CHECKPOINT:-}" ]; then run --controller rl --checkpoint "$CHECKPOINT"; fi
python analyze_results.py --exp "$EXP" --markdown "results/$EXP/summary.md" > /dev/null
echo "summary: results/$EXP/summary.md"
