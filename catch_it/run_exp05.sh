#!/usr/bin/env bash
# Author: Zwe Min Htet Aung
# exp05 pipeline: Pengze's PPO with ball sensor noise only (no delay, no joint noise) and steadier PPO
# settings. Checks the settings on an easy version first, runs the baselines, trains, picks a checkpoint on
# the validation throws, plots the training curves, tests once, and summarizes next to the classical catcher.
# See results/exp05_ball_noise/README.md for what and why. Takes ~14 h.
#
# Launch from catch_it/ as a background user service (survives closing the terminal; memory-capped):
#   systemd-run --user --unit=exp05 -p MemoryMax=11G -p MemorySwapMax=0 --nice=10 \
#     --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
#     --working-directory=$PWD systemd-inhibit --what=sleep:idle --why=exp05 bash run_exp05.sh
# Progress: cat results/exp05_ball_noise/STATUS ; tail -f results/exp05_ball_noise/pipeline.log
#           grep "Agent Steps" outputs/rl_exp05/train.log | tail -n 3
# Stop:     systemctl --user stop exp05      (rerunning skips the stages already done; an interrupted
#                                             training stage starts over)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/dl-rl/bin/python}
EXP=exp05_ball_noise
RES=results/$EXP
WORK=outputs/exp05_pipeline          # stage markers and the training runs' paths
VAL=outputs/exp05_val                # validation CSVs (kept out of $RES: analyze_results reads every CSV there)
TEST_SEEDS=10000:15000
VAL_SEEDS=0:500
WORKERS=8                            # eval workers, ~0.75 GB each; 16 would go over the memory cap
STEPS=${STEPS:-25000000}             # Pengze's training budget
SANITY_STEPS=${SANITY_STEPS:-2000000}
SANITY_MIN=0.5                       # old settings ended at 0.315 on this check
TRAIN_ARGS="task=HoopCatch num_envs=16 device_id=0 seed=1"
mkdir -p "$RES" "$WORK" "$VAL"
exec > >(tee -a "$RES/pipeline.log") 2>&1

STAGE=start
trap 'echo "failed $STAGE $(date "+%F %T")" > "$RES/STATUS"; echo "### FAILED in stage $STAGE"' ERR
stage() {   # stage <name>: returns 1 (skip) if the stage already finished
    STAGE=$1
    if [ -f "$WORK/$1.done" ]; then echo "### skip $1 (done before)"; return 1; fi
    echo "running $1 $(date "+%F %T")" > "$RES/STATUS"
    echo "### $(date "+%F %T") $1"
}
done_stage() { touch "$WORK/$STAGE.done"; }
eval_test() { "$PY" eval_catch.py --exp "$EXP" --seeds "$TEST_SEEDS" --workers "$WORKERS" "$@" | tail -n 2; }
newest_run() { ls -td outputs/"$1"/*/*/ | head -n 1; }

echo "### exp05 pipeline at commit $(git rev-parse --short HEAD), $(date "+%F %T")"
git diff --quiet HEAD || echo "### WARNING: uncommitted changes; the results cannot pin the code"

# 1. Smoke test: trains, saves, and loads in eval_catch.py
if stage smoke; then
    "$PY" train_DCMM.py task=HoopCatch num_envs=8 device_id=0 seed=1 \
        train.ppo.max_agent_steps=30000 output_name=exp05_smoke | tail -n 3
    "$PY" eval_catch.py --exp exp05_smoke --controller rl --checkpoint "$(newest_run exp05_smoke)nn/last.pth" \
        --seeds 0:20 --workers 4 --out_dir outputs/exp05_smoke_eval | tail -n 2
    done_stage
fi

# 2. Sanity gate: with the aim point visible the task is easy; the settings must learn it and keep it
if stage sanity; then
    mkdir -p outputs/exp05_sanity
    "$PY" tools/sanity_aim_obs.py train $TRAIN_ARGS train.ppo.max_agent_steps="$SANITY_STEPS" \
        output_name=exp05_sanity > outputs/exp05_sanity/train.log 2>&1
    grep "Agent Steps" outputs/exp05_sanity/train.log | tail -n 2
    run=$(newest_run exp05_sanity)
    "$PY" plot_training.py --run_dir "$run" --out "$RES/sanity_curves" | tail -n 1
    "$PY" tools/sanity_aim_obs.py eval --checkpoint "${run}nn/last.pth" --seeds 0:200 --workers "$WORKERS" \
        --min "$SANITY_MIN" | tee "$RES/sanity.txt"
    done_stage
fi

# 3. Baselines on the test throws
if stage baselines; then
    for c in classical_multi random oracle classical_single classical_perfect; do eval_test --controller "$c"; done
    done_stage
fi

# 4. Train
if stage train; then
    mkdir -p outputs/rl_exp05
    "$PY" train_DCMM.py $TRAIN_ARGS train.ppo.max_agent_steps="$STEPS" output_name=rl_exp05 \
        > outputs/rl_exp05/train.log 2>&1
    grep "Agent Steps" outputs/rl_exp05/train.log | tail -n 2
    newest_run rl_exp05 > "$WORK/run_dir"
    done_stage
fi

# 5. Validation seeds 0-499 only: every snapshot, and classical_multi on the same throws (for the curves)
if stage validate; then
    rm -f "$RES/validation.md"
    printf "# exp05 checkpoint choice on the validation throws\n\nMade by \`validate_checkpoints.py\` (\`run_exp05.sh\` stage 5). The picked checkpoint is the only one run on the test throws.\n" > "$RES/validation.md"
    "$PY" validate_checkpoints.py --nn_dir "$(cat "$WORK/run_dir")nn" --label rl --csv_dir "$VAL" \
        --md "$RES/validation.md" --seeds "$VAL_SEEDS" --workers "$WORKERS" | tee "$WORK/val.txt"
    "$PY" eval_catch.py --exp val_ref --controller classical_multi --seeds "$VAL_SEEDS" --workers "$WORKERS" \
        --out_dir "$VAL" | tail -n 2
    done_stage
fi

# 6. Training curves: training vs validation catch rate, reward, losses, KL, exploration std
if stage curves; then
    "$PY" plot_training.py --run_dir "$(cat "$WORK/run_dir")" --val_dir "$VAL/rl" \
        --ref_json "$VAL/val_ref/classical_multi.json" --out "$RES/training_curves" | tail -n 1
    done_stage
fi

# 7. Freeze the pick (a copy of the checkpoint file, so the test can be rerun from the repo)
if stage freeze; then
    cp "$(tail -n 1 "$WORK/val.txt" | awk '{print $2}')" "$RES/rl_policy.pth"
    tail -n 1 "$WORK/val.txt" | tee "$RES/picked.txt"
    done_stage
fi

# 8. Test once on the test throws
if stage test; then
    eval_test --controller rl --checkpoint "$RES/rl_policy.pth"
    done_stage
fi

# 9. Summary: catch rates, paired test against classical_multi, catch rate by noise level
if stage summary; then
    "$PY" analyze_results.py --exp "$EXP" --ref classical_multi --markdown "$RES/summary.md" > /dev/null
    grep -A 9 -m 1 "catch" "$RES/summary.md" || true
    done_stage
fi

echo "done $(date "+%F %T")" > "$RES/STATUS"
echo "### all done: $RES/summary.md, $RES/training_curves.png"
