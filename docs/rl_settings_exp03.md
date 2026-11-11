# Settings for training and testing the RL policy against the exp03 baseline

Author: Zwe Min Htet Aung

The classical baseline (`classical_multi`) caught **76.0%** of the throws in exp03 (95% CI 74.8–77.2%). That number is fixed: the baseline doesn't learn, so it doesn't change when you train. To compare the RL policy with it, **test the policy on the same env, with the same settings, on the same 5000 throws**. Its catch rate then sits next to 76.0% as a fair comparison.

This page lists everything that has to match. Where a setting isn't listed, you are free to change it (see the last section).

Source of truth: commit `c403bf3` on `larry-classical`. Full results: `catch_it/results/exp03_harder/README.md`.

## 1. Get the env (easiest way)

`larry-catch` is still at `8510670`, which is an ancestor of `larry-classical`. Merging brings in every env change below, with nothing to copy by hand:

```bash
git fetch origin
git checkout larry-catch
git merge origin/larry-classical      # fast-forward if larry-catch has no new commits
```

If `larry-catch` has new commits, the merge only touches the env, eval and docs files. Your reward weights and PPO settings may conflict in `DcmmCfg.py` or `DcmmPPO.yaml`. If so, keep your reward and PPO values and take everything else from `larry-classical`.

**Check that you have the same env.** Run the baseline on your branch:

```bash
cd catch_it
python eval_catch.py --exp check --controller classical_multi
```

It must give 76.0% again. If it doesn't, something in the env differs from exp03. Find the difference before you trust any RL number.

## 2. Settings that must match

All of these are the defaults in `catch_it/configs/env/DcmmCfg.py` at `c403bf3`, so after the merge you don't have to set anything. Don't override them in training.

### Disturbances, drawn independently for each throw

| Setting | `DcmmCfg` name | exp03 value |
|---|---|---|
| Action delay | `act_delay = {'arm': [1, 2, 3, 4, 5]}` | 0–4 control steps (0–160 ms), equally likely |
| Arm PID gain scale, per joint | `k_arm` | ×0.5–1.5 |
| Wind | `wind_force` | horizontal force 0–0.04 N, random direction, from release on |
| Air drag | `object_damping` | b = 0–0.015 (up to 35% speed lost per second) |
| Ball position noise std | `k_obs_object` | 5–20 mm |
| Ball velocity noise std | `k_obs_object_vel` | 0.1–0.4 m/s |
| Torque noise | `k_act` | each joint's torque × N(1, 0.10), every physics step |
| Throw aim | `aim_with_disturbances = True` | launch velocity solved with the drag and wind, so the ball passes through its target |
| Gravity | `gravity_noise = 0.0` | fixed at 9.81 m/s² |

### Fixed noise and ball

| Setting | `DcmmCfg` name | Value |
|---|---|---|
| Arm joint position and velocity noise std | `k_obs_arm` | 0.001 |
| Hoop position, axis and velocity noise std | `k_obs_hoop` | 0.001 |
| Ball radius | `object_radius` | 35–45 mm, per throw |
| Ball mass | `object_mass` | 35–75 g, per throw |

### Throws

| Setting | `DcmmCfg` name | Value (relative to the arm base) |
|---|---|---|
| Catch zone (where the throw is aimed) | `catch_zone` | r 0.35–0.65 m, azimuth ±45° from +y, z 0.3–0.8 m |
| Launch point | `launch_zone` | r 2.0–3.0 m, azimuth ±60°, height 0.8–1.6 m above the floor |
| Flight time to the target | `flight_time` | 0.6–1.0 s |

### Action interface and timing

| Setting | Value |
|---|---|
| Action | 5 numbers in [−1, 1]: hoop-center step (x, y, z) and hoop-axis tilt (2) |
| Step sizes (`action_scale`) | 0.04 m per step for position, 0.1 rad per step for tilt |
| Target lead (`hoop_target_lead`) | the hoop target stays within 0.15 m of the actual hoop |
| Workspace box (`hoop_target_low` / `high`) | x −0.75 to 0.75, y −0.05 to 0.9, z 0 to 1.1 m |
| IK (`ik_config`) | QP solver, gain 1.0, free rotation about the hoop axis |
| PID gains (`Kp_arm`, `Ki_arm`, `Kd_arm`) and limits | unchanged from `8510670` |
| Control rate | 25 Hz: 40 ms per step, 20 physics substeps (`steps_per_policy=20`) |
| Episode limit | 2.5 s (`env_time=2.5`) |

### Observation (35 numbers)

| Part | Contents |
|---|---|
| hoop | position (3), axis direction (3), velocity (3) |
| arm | joint positions (6), joint velocities (6) |
| object (the ball) | position (3), velocity (3), position relative to the hoop (3) |
| prev_action | the last action (5) |

All positions are in the arm-base frame. The policy gets only this observation. It must not read anything else from the env, such as the throw target, the true ball state, the wind, the drag or the delay. The classical catcher doesn't see them either. (`classical_perfect` and `oracle` do, which is why they are references, not competitors.)

### The catch test and the end of an episode

Unchanged from `8510670`:
- **Catch:** the ball's center crosses the ring plane inside the ring (inner radius 0.125 m).
- **Fail:** the ball touches the floor, the arm or the base, or the arm or hoop hits the base or the floor.
- **Timeout:** the 2.5 s limit is reached.

## 3. How to test

Final numbers come from `eval_catch.py` on the **test seeds 10000–14999**, the same 5000 throws as the baseline:

```bash
cd catch_it
python eval_catch.py --exp exp03_harder --controller rl --checkpoint outputs/<run>/nn/<file>.pth
python analyze_results.py --exp exp03_harder     # puts RL next to the classical rows, with paired tests
```

Rules:
- **Don't use `train_DCMM.py test=True` for reported numbers.** It doesn't fix the throws, so its numbers can't be paired with the baseline.
- **Pick the checkpoint on the validation seeds 0–499:** `--seeds 0:500 --exp val`. Run the test seeds once, on the checkpoint you picked. If you choose checkpoints by test seeds, the comparison isn't fair any more.
- **Commit before the test run.** The `.json` written next to the results records the commit and whether the tree had uncommitted changes. The baseline was run at `c403bf3`. Your run should use a commit whose env files are identical to it.

## 4. What you are free to change

- Reward terms and weights (`reward_weights`, `path_near_sigma`, `align_sigma`).
- PPO settings and network size (`configs/train/DcmmPPO.yaml`).
- **Observation history (recommended).** The ball noise is high: up to 20 mm in position and 0.4 m/s in velocity. The classical catcher averages about 20 readings per throw, so the noise barely affects it. A memoryless MLP sees one frame: at 0.4 m/s velocity noise, its guess of where the ball will be 0.25 s later is off by about 10 cm. Stacking the last few ball observations (for example, the last 4) gives the policy the same chance to average.
  - This doesn't change the env. The policy just keeps its own buffer of past observations.
  - If you add it, update `RLController.act` in `eval_catch.py` to build the same input as `PPO_Hoop.obs2tensor`. The network size is read from the checkpoint automatically.
- Extra randomization used only during training, such as wider ranges. Report it.

Report training time. If time allows, train 2–3 seeds and report the spread.

## 5. The numbers to compare against

exp03, test seeds 10000–14999:

| Controller | Catch rate | 95% CI |
|---|---|---|
| random | 5.0% | 4.5–5.7% |
| classical_single | 21.5% | 20.4–22.7% |
| **classical_multi (the baseline)** | **76.0%** | **74.8–77.2%** |
| classical_perfect (exact prediction, not a competitor) | 81.4% | 80.3–82.5% |
| oracle (knows the target in advance, not a competitor) | 84.4% | 83.3–85.4% |

Where the baseline loses, and so where RL can win:
- **Delay:** 99 / 91 / 76 / 65 / 51% at 0 / 1 / 2 / 3 / 4 steps.
- **Crosswind:** 80 / 75 / 70% at 0–0.01 / 0.01–0.02 / 0.02–0.04 N.

`analyze_results.py` prints these same breakdowns for the RL policy.
