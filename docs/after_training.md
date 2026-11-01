# After Training: Testing the HoopCatch Policy

What to do once `hoop_full` finishes (or after you stop it with Ctrl+C).
Design details (states, actions, rewards): [hoop_rl_design.md](hoop_rl_design.md).
All training commands: [../catch_it/RUN_COMMANDS.md](../catch_it/RUN_COMMANDS.md).

Every command below starts from the project folder with the env active:

```bash
cd ~/ece1508/catch_it
conda activate dl-rl
```

## 1. Check that training really stopped

```bash
pgrep -af train_DCMM        # nothing printed = stopped
tail -n 20 hoop_full.log    # should end with "max steps achieved" (or a KeyboardInterrupt if you pressed Ctrl+C)
```

Stopping early is fine because a checkpoint is saved after every epoch.

## 2. Find the checkpoints

```bash
ls outputs/hoop_full/*/*/nn/
```

| File | What it is |
|---|---|
| `best_reward_XX.XX.pth` | Checkpoint with the highest mean episode **reward**. Start here. |
| `last.pth` | Weights from the final epoch. |
| `ep_XXXX_step_XXXXm_reward_XX.XX.pth` | Snapshots saved every few epochs along the way. |

"Best reward" isn't always "best catch rate", because reward also includes shaping terms such as path distance and smoothness. Test both `best_reward` and `last`, and compare them with step 4.

Save the path to a variable so the commands below stay short. Use your own date and time folder:

```bash
RUN=~/ece1508/catch_it/outputs/hoop_full/2026-10-30/23:48:08/nn
ls $RUN
```

> Paths contain `:`, so hydra needs the value wrapped in double quotes inside single quotes:
> `"checkpoint=\"$RUN/best_reward_XX.XX.pth\""`. The commands below already do this.

## 3. Watch it catch (3D viewer)

```bash
python train_DCMM.py task=HoopCatch test=True num_envs=1 device_id=0 viewer=True \
  "checkpoint=\"$RUN/best_reward_XX.XX.pth\""
```

- The MuJoCo window opens and the simulation plays in real time, one throw at a time.
- In test mode the policy uses its **mean action**, with no exploration noise, so it behaves the same way it would on a real robot.
- The terminal prints a summary block after each batch (see step 4). It stops on its own after 10,000 agent steps, about 7 minutes. Press Ctrl+C to stop sooner.
- You need a display for this, so it won't work over plain SSH.

What to look for:
- The hoop moves **before** the ball arrives and meets it on its path, instead of chasing it.
- The hoop opening faces the incoming ball.
- No sudden jerks, and the arm never swings into its own base.

## 4. Measure the catch rate (no window)

```bash
python train_DCMM.py task=HoopCatch test=True num_envs=8 device_id=0 \
  "checkpoint=\"$RUN/best_reward_XX.XX.pth\"" 2>&1 | tee test_best.log
```

This runs 8 sims in parallel and prints a block like this about 20 times:

```
## Sample Length 100 ##
mean_rewards:  ...
mean_lengths:  ...
mean_success: 0.870              <- catch rate (THE number)
mean_min_ball_dist: 0.012
mean_min_path_dist: 0.010
mean_rim_hits: 0.150
mean_ik_fails: 0.300
mean_outcome/catch: 0.870
mean_outcome/ball_floor: 0.090
mean_outcome/ball_robot: 0.020
mean_outcome/collision: 0.000
mean_outcome/timeout: 0.020
```

Read the **last** block, after `Sample Length` reaches 100. The values are averages over the last 100 test episodes:

```bash
grep -A12 "Sample Length 100" test_best.log | tail -n 12
```

### How to judge the number

| `mean_success` | Meaning |
|---|---|
| ~0.08 | Same as a random policy, so it learned nothing. Check the reward curve in TensorBoard. |
| 0.3 – 0.7 | Partly learned. It probably needs more steps, or some throw angles are hard. |
| > 0.8 | Good. |
| ~0.95 | As good as the scripted oracle, which cheats by knowing the throw's aim point. |

To get the oracle reference in the same conditions, run `cd gym_dcmm/envs && python DcmmVecEnv.py --oracle --episodes 200`.

## 5. Compare all checkpoints in one go

```bash
for ck in $RUN/best_reward_*.pth $RUN/last.pth; do
  echo "== $(basename $ck)"
  python train_DCMM.py task=HoopCatch test=True num_envs=8 device_id=0 "checkpoint=\"$ck\"" 2>&1 \
    | grep "mean_success" | tail -n 1
done
```

To include the periodic snapshots, add `$RUN/ep_*.pth` to the list. That also shows whether the catch rate was still rising at the end of training, which would mean more training would help.

## 6. Read the failures (when the catch rate is low)

Use the `mean_outcome/*` lines from step 4:

| Dominant outcome | What it means | First thing to try |
|---|---|---|
| `ball_floor` high, `min_path_dist` large (> 0.1 m) | The hoop never gets to the ball's path. | Train longer, or raise `r_path_progress` / `r_path_near` in `configs/env/DcmmCfg.py`. |
| `ball_floor` high, `min_path_dist` small, `rim_hits` > 0 | It gets there but the ball hits the rim, or the hoop is tilted wrong. | Raise `r_align`, or check the tilt in the viewer. |
| `ball_robot` high | The ball hits the arm or base instead of the hoop. | Watch in the viewer. The hoop is probably blocking with the wrong side. |
| `collision` high | The arm hits itself or the base. | Raise the `r_collision` penalty. |
| `timeout` high | The episode ends before the ball arrives or lands. Rare. | Check `env_time` in the env. |
| `ik_fails` high (> 2 per episode) | The policy pushes the hoop target out of reach. | Shrink `hoop_target_low/high`, or raise `r_constraint`. |

The full debugging table and the reasoning behind each reward is in [hoop_rl_design.md](hoop_rl_design.md).

## 7. Train more from a checkpoint

If the catch rate was still rising at the end, continue from `last.pth`. Only the weights and the observation normalizer are restored. The step counter starts from 0, so set the extra budget yourself:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
systemd-run --user --scope -p MemoryMax=12G \
nice -n 10 python train_DCMM.py task=HoopCatch num_envs=8 device_id=0 output_name=hoop_full_resume \
  "checkpoint=\"$RUN/last.pth\"" train.ppo.max_agent_steps=10000000 \
  2>&1 | tee hoop_full_resume.log
```

If you change a reward weight in `DcmmCfg.py`, start a **fresh** run with a new `output_name` instead of resuming. Old and new reward scales don't mix well, and separate runs are easier to compare in TensorBoard.

## 8. Keep a record

After each test, write down the checkpoint, its `mean_success`, the dominant failure outcome, and any config you changed. A line in a notes file or a table in the report is enough. This makes it easy to see which change actually helped.
