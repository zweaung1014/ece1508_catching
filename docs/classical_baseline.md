# HoopCatch: classical baseline

A non-learning catcher to compare against the PPO policy. It predicts the ball's parabola, picks a reachable catch point on it, and drives the hoop there. It sees the same observation and uses the same action interface as the RL policy, so the comparison measures the method, not the interface.

## 1. Where things live

| What | File |
|---|---|
| Parabola prediction (Campana & Laumond Eqs. 1–3) and the least-squares fit | `catch_it/gym_dcmm/classical/ballistic.py` |
| The catcher: release detection, catch-point choice, servo | `catch_it/gym_dcmm/classical/catcher.py` |
| All catcher settings | `catch_it/configs/env/ClassicalCfg.py` |
| Evaluator for every controller (fixed seeds, one CSV row per episode) | `catch_it/eval_catch.py` |
| Full test run (all controllers into `results/<exp>/`) | `catch_it/run_eval.sh` |
| Results, one folder per experiment with a README of its settings | `catch_it/results/<exp>/` |
| Tables: catch rate, CIs, paired tests, miss reasons, per-bin rates | `catch_it/analyze_results.py` |
| Tests of the predictor (incl. against MuJoCo free flight) | `catch_it/tests/test_ballistic.py` |

## 2. Prediction (Campana & Laumond 2016, Section III-A)

The paper's point-robot takeoff state (c_s, ċ_s) is replaced by the ball's estimated state. Only the unconstrained ballistic equations are used; the friction cones, velocity limits (Eqs. 4–7) and roadmap planner are not.

- Eq. 1: `c(t) = c_s + ċ_s·τ − ½·g·τ²·e_z`, τ = t − t_s. Ball position over time.
- The path lies in the vertical plane π_θ: `θ = atan2(ẏ_s, ẋ_s)`, `Θ = (cos θ, sin θ, 0)`, `x_θ = c·Θ`, `ẋ_θs = ċ_s·Θ`. The horizontal speed is constant, so `t = t_s + (x_θ − x_θs)/ẋ_θs`.
- Eq. 2: `z(x_θ)`, the height along the path.
- Eq. 3: slope `ż/ẋ_θs`, the ball's direction of travel. The hoop axis is set to the opposite direction, so the opening faces the incoming ball.

With drag and wind off, the simulator is exactly ballistic (RK4 integrator): from the true release state, Eq. 1 matches MuJoCo's ball to 3·10⁻¹⁴ m. With them on (exp02 on), the ball's true path is `DragPath` in `ballistic.py`, the closed form of linear drag plus a constant wind force, which matches MuJoCo to ~10⁻⁹ m (`tests/test_ballistic.py`). The catcher's parabola does not model either.

### Estimating the state from the observation

The catcher reads the same noisy observation as the policy (25 Hz; in exp03 each throw draws a ball position σ of 5–20 mm and a velocity σ of 0.1–0.4 m/s). Two details matter:

- **The first read after release has a wrong velocity.** The env's velocity is a finite difference over the last 40 ms, which still includes the hold. Its position is used; its velocity is not.
- **A finite-difference velocity is the velocity 20 ms ago** (the middle of the window). The fit models it that way; ignoring it costs ~0.2 m/s vertically, a 10–18 cm miss.

`fit_parabola` solves a weighted least-squares problem for the state at the latest read: position reads `p_i = c + ċ·τ_i − ½g·τ_i²·e_z`, velocity reads `v_j = ċ − g·(τ_j − Δt/2)·e_z`, weighted by 1/σ. With one position and one velocity read it is exactly determined. In multi-read mode the vertical acceleration is an extra unknown with a prior N(9.81, 0.5²), so it is learned from the data once enough reads exist (exp01 randomized g by ±0.5 m/s²; since exp02 g is fixed, and the fitted value absorbs part of the drag). The weights use the assumed σ in `ClassicalCfg.py`: 12.5 mm and 0.25 m/s for exp03, the middle of its noise ranges (5 mm and 0.05 m/s up to exp02).

## 3. Catcher (`catcher.py`)

Each control step (25 Hz):

1. **Ready (ball held).** Average the ball's resting position. Move the hoop straight ahead to the middle of the reach region (0.5 m out, 0.55 m up) with its axis turned toward the launch point, 45° up. The middle minimizes the worst-case move, since throws often cross from one side to the other (a launch-side ready pose missed those).
2. **Release.** The ball is released when it is > 8 cm from its rest position and its observed speed is > 1.6 m/s (3 cm and 1 m/s up to exp02; with exp03's noise those triggered on ~6% of throws while the ball was still held).
3. **Predict** (section 2), from the next read on (≈ 40–80 ms after release).
4. **Choose the catch point.** Sample the predicted path every 10 ms. Keep points inside the reach region (0.35–0.65 m out, ±45°, 0.3–0.8 m up, from the reachability study in `hoop_rl_design.md` §3) and the hoop target box. For each point, `slack = time until the ball is there − time the hoop needs` (distance at 0.8 m/s, tilt at 2 rad/s, plus 0.1 s settle; below the interface caps of 1 m/s and 2.5 rad/s). Pick the point with slack ≥ 0 needing the least motion (translation + 0.2 m/rad × tilt); if none, the one with the most slack.
5. **Servo.** The env oracle's servo law, aimed at the predicted point and axis instead of the privileged aim point, output as the same 5-D action as the policy. The hoop's own pose comes from the observation (position) and forward kinematics of the observed joint angles (orientation).

Modes:

| Mode | Ball information | Updates |
|---|---|---|
| `single_read` | one read after release (position + velocity) | none: open loop on the ball |
| `multi_read` | every read since release | re-fit and re-plan every step; keeps the chosen catch time, only re-selects if it becomes unreachable, and stops re-selecting 0.15 s before the catch |
| `perfect_state` | true release state, gravity, drag and wind (diagnostic) | none |

### Fairness rules

- The catcher reads only the observation dict. It never reads the throw target, arrival velocity, the episode's gravity, drag or wind (except `perfect_state`, which is a diagnostic).
- Same action interface as RL, so the same limits: 0.04 m and 0.1 rad per step (1 m/s, 2.5 rad/s), 0.15 m target lead, workspace box, IK, PID, action delay.
- The reach region is a fixed setting, separate from the throw sampler's `DcmmCfg.catch_zone`. A much wider assumed region (0.25–0.8 m, ±70°, 0.15–0.95 m) still caught 200/200 validation throws, so the region does not leak the aim zone.
- It never draws from `np.random`, so a seed gives the same throw and the same noise to every controller.
- Settings were tuned on validation seeds 0–499 only (in exp01's env). For exp03 the assumed sensor noise and the release thresholds were set from the new noise ranges and checked on validation seeds. Test seeds 10000–14999 are for reporting.

## 4. Experiment

**Controllers** (all through `env.step` with the same 5-D action):

| Controller | Ball information | Role |
|---|---|---|
| `random` | none (N(0,1) actions) | floor |
| `classical_single` | 1 noisy read | open-loop baseline |
| `classical_multi` | all noisy reads | main classical baseline |
| `rl` | all noisy reads | the method |
| `classical_perfect` | true release state, gravity, drag and wind | diagnostic: planner + arm limits only |
| `oracle` | aim point, known before release | privileged reference |

**Test setting** (exp03, the `DcmmCfg.py` defaults): one setting, with every disturbance drawn at random for each throw:

| Disturbance | Per-throw draw |
|---|---|
| Action delay | the arm follows the command from 0–4 steps ago (0–160 ms), equally likely |
| Arm PID gains | each joint's gains ×0.5–1.5 |
| Wind | constant horizontal force 0–0.04 N in a random direction, from release on (about a 5 m/s breeze on a 4 cm ball) |
| Air drag | damping b 0–0.015 on the ball (up to ~35% speed lost per second for a 35 g ball) |
| Ball position obs noise | Gaussian, std 5–20 mm |
| Ball velocity obs noise | Gaussian, std 0.1–0.4 m/s |
| Torque noise | ×N(1, 0.10) on each physics step |
| Gravity | fixed at 9.81 |

- **Every throw is aimed with its drag and wind** (`DcmmCfg.aim_with_disturbances`), so its true path still passes through the target point: every throw stays catchable, the path is just not a parabola. Aimed as if there were no drag or wind, 26% of exp03's throws would never pass through the catch zone.
- Unchanged: arm/hoop obs noise 0.001, ball radius 35–45 mm and mass 35–75 g, the throw sampler (target, launch point, flight time). A seed gives the same target, launch point, flight time and ball as in exp01 and exp02.
- Each CSV row logs its throw's draws: `pos_noise_std`, `vel_noise_std`, `drag_b`, `drag_rate`, `wind_N`, `crosswind_N`, `tailwind_N`, `act_delay_steps`, `pid_gain_mean`, `pid_gain_min`, and `aim_miss` (~0 when aimed).
- The catcher still assumes no drag, wind or delay; its assumed noise is set from the ranges above (`ClassicalCfg.py`).
- Earlier settings: exp02 (smaller ranges, no wind, aimed without drag) and exp01 (one factor per condition); see the `README.md` in each `catch_it/results/` folder.

**Metrics** (`analyze_results.py`): catch rate with a 95% Wilson interval; paired exact McNemar tests on the same seeds; outcome breakdown; rim hits, IK failures, hoop travel, action smoothness; for the classical modes, prediction error across the true path (drag and wind included) at the planned catch time, hoop tracking error, and the reason for each miss (not reachable in time > prediction > arm tracking > other); catch rate by each per-throw disturbance (position and velocity noise, drag, wind, crosswind, delay, weakest PID gain), flight time, launch azimuth and home-to-target distance. The disturbances are drawn independently, so each bin averages over the others.

## 5. Running it

From `catch_it/` with the env active:

```bash
python tests/test_ballistic.py                                   # predictor, drag, wind, aiming and disturbance checks
python eval_catch.py --exp scratch --controller classical_multi  # 5000 test throws, ~3 min with 16 workers
python eval_catch.py --exp scratch --controller classical_single --seeds 10000:10010 --viewer --workers 1   # watch
python eval_catch.py --exp scratch --controller rl --checkpoint outputs/<run>/nn/<file>.pth
EXP=exp03_harder bash run_eval.sh                                # everything; CHECKPOINT=... adds RL
python analyze_results.py --exp exp03_harder                     # tables; --markdown results/<exp>/summary.md
```

Results go to `catch_it/results/<exp>/<controller>.csv`, with a `.json` next to each (seeds, git commit, full env and catcher settings). Commit the code before a run you keep, so the `.json` pins it. Each experiment folder gets a `README.md` saying what was varied.

## 6. Keeping RL and classical comparable

For whoever trains the RL policy:

- **Must match between the two branches:** the observation and its noise (including the per-throw position- and velocity-noise ranges), the action interface (step sizes, 0.15 m lead, workspace box, IK settings, PID gains and their randomization, the per-throw action delay, torque noise, 25 Hz), the throw sampler (catch zone, launch zone, flight time, fixed gravity, drag and wind ranges, aiming with them, ball randomization), the catch test, the end-of-episode conditions and the 2.5 s limit. Make env changes where both branches pull from, and evaluate both on the same commit (the `.json` records it).
- **Train with the same randomization.** A policy trained without exp03's delay, PID, wind, drag and noise ranges is tested out of distribution. The `DcmmCfg.py` defaults are exp03's, so training picks them up.
- **Give the policy a short history if the sensor noise stays this high.** The multi-read catcher averages ~20 reads per throw, so raising the noise cost it ~1.5 points; a memoryless MLP sees one frame, and at 0.4 m/s velocity noise its extrapolation 0.25 s ahead is off by ~10 cm. Stacking the last few ball observations (e.g. 4) gives it the same chance to average.
- **Free to change:** reward terms and weights, PPO settings, network size, randomization used only during training (report it).
- **Final numbers come from `eval_catch.py`** on test seeds 10000–14999, not from `train_DCMM.py test=True` (that mode does not fix the throws). Pick the checkpoint on validation seeds 0–499.
- `eval_catch.py --controller rl` reads the MLP size from the checkpoint; if the observation vector changes, update the flattening in `RLController.act` to match `PPO_Hoop.obs2tensor`.
- If time allows, train 2–3 seeds and report the spread, plus training time.

## 7. Results

### exp01: one factor per condition

Test seeds 10000–10999 (1000 throws per cell), 2026-11-09, code `ed79c76`. Full tables, confidence intervals, paired tests and per-bin rates: `catch_it/results/exp01_one_factor/summary.md`; settings in its `README.md`.

| Catch rate | nominal | noise ×2 | noise ×4 | tracker noise | gravity fixed | delay +1 | delay +2 |
|---|---|---|---|---|---|---|---|
| random | 4.6% | | | | | | |
| classical_single | 95.7% | 71.5% | 33.7% | 39.0% | 99.2% | 82.6% | 70.2% |
| classical_multi | 99.7% | 99.7% | 99.6% | 99.4% | 99.8% | 97.8% | 89.3% |
| classical_perfect | 99.8% | 99.8% | 99.8% | 99.8% | 99.9% | 98.0% | 88.8% |
| oracle | 94.9% | 94.9% | 94.9% | 94.9% | 95.0% | 94.7% | 88.3% |

What it shows:

- **The multi-read catcher is close to perfect** and is unaffected by observation noise up to ×4: re-fitting averages the noise away (median prediction error 0.3 cm nominal). Its 3 nominal misses are throws it could not reach in time.
- **The single-read catcher is limited by prediction**: 32 of its 43 nominal misses are prediction errors, mostly from the ±0.5 m/s² gravity randomization (99.2% with gravity fixed). It collapses with more noise, because one velocity read carries all the error.
- **Delay is the only condition that hurts the multi-read catcher**, and the perfect-state diagnostic drops just as much, so the loss comes from the arm arriving late, not from prediction.
- **The oracle is below the classical catcher** because it drives to the throw's exact aim point, which is not always the easiest catch point, while the catcher picks the least-motion point from a centered ready pose.
- For the RL comparison: on nominal throws the most RL can do is tie (~99.7%). The conditions with room for RL to win are actuation delay and, once implemented, drag.

Settings used: `ClassicalCfg.py` as committed with these results (also recorded in each `.json`).

### exp02: every disturbance randomized per throw

Test seeds 10000–14999 (5000 throws per controller), 2026-11-10, code `df856c0`. Full tables: `catch_it/results/exp02_randomized/summary.md`; settings in its `README.md`.

| Catch rate | all throws | delay 0 / 1 / 2 steps | vel noise low / mid / high | drag speed loss 0–4 / 4–8 / 8–13% |
|---|---|---|---|---|
| random | 5.9% | | | |
| classical_single | 57.2% | 65 / 57 / 50% | 77 / 55 / 40% | 59 / 57 / 53% |
| classical_multi | 96.1% | 100 / 98 / 91% | 97 / 96 / 96% | 96 / 96 / 96% |
| classical_perfect | 96.2% | 100 / 98 / 91% | 96 / 96 / 96% | 95 / 97 / 97% |
| oracle | 90.8% | 95 / 92 / 85% | 90 / 91 / 91% | 94 / 91 / 79% |

What it shows:

- **Delay is the only disturbance that hurts the multi-read catcher**: it catches everything without delay and 91% at 2 steps. `classical_perfect`, which predicts exactly, gets the same 96.2% and fails on the same pattern, so the multi-read catcher's prediction is not the bottleneck: the arm arriving late is.
- **Drag and velocity noise don't matter to it**: re-fitting every step tracks the real (dragged) path to 0.6 cm median, though it assumes no drag and a 0.05 m/s noise.
- **The single-read catcher collapses with velocity noise** (77% → 40%): one velocity read carries all the error. 1553 of its 2141 misses are prediction errors.
- **The oracle loses with drag** (94% → 79%): it drives to the drag-free aim point, which the ball no longer passes through.
- The "not reachable in time" label on most multi-read misses is a side effect of delay (late re-plans are logged as infeasible; 92% of those throws are still caught), not a reachability problem.
- For the RL comparison: the room to beat the classical catcher is at 1–2 steps of delay, where both classical modes lose 2–9% of throws.

## 8. Env changes the RL branch needs too

The RL branch (`larry-catch`) is at `8510670`, before both exp02 and exp03. It needs:

- `DcmmVecEnv.py` logs, for evaluation only: `self.throw_time` (env time of the release), `self.throw_flight_time`, and the per-throw draws `self.object_damping`, `self.act_delay_steps`, `self.k_obs_object`, `self.k_obs_object_vel`, `self.wind_force`.
- Per-throw disturbances: the `DcmmCfg.py` ranges (`k_obs_object`, `k_obs_object_vel`, `object_damping`, `wind_force`, `act_delay`, `k_arm`, `k_act`, `gravity_noise = 0`, `aim_with_disturbances = True`).
- Draw order in `_reset_simulation()`: ball, gravity, throw, PID, delay, then `random_obs_noise()` (velocity std, then position std) and `random_wind()` last, so a seed's throw, ball and PID draws stay the same. Then `aim_throw()` re-solves the launch velocity with the drag and wind (no random draws). `_get_obs` uses the per-throw stds.
- The wind is applied at release through `data.xfrc_applied` on the ball (`step()`), and `ball_acceleration()` gives gravity + wind / mass.
- `_predict_path()` (reward shaping only) follows the drag-and-wind path, so the shaping points at the ball's real path.
