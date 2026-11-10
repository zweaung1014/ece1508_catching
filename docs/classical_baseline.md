# HoopCatch: classical baseline

A non-learning catcher to compare against the PPO policy. It predicts the ball's parabola, picks a reachable catch point on it, and drives the hoop there. It sees the same observation and uses the same action interface as the RL policy, so the comparison measures the method, not the interface.

## 1. Where things live

| What | File |
|---|---|
| Parabola prediction (Campana & Laumond Eqs. 1–3) and the least-squares fit | `catch_it/gym_dcmm/classical/ballistic.py` |
| The catcher: release detection, catch-point choice, servo | `catch_it/gym_dcmm/classical/catcher.py` |
| All catcher settings | `catch_it/configs/env/ClassicalCfg.py` |
| Evaluator for every controller (fixed seeds, one CSV row per episode) | `catch_it/eval_catch.py` |
| Full test matrix (all controllers × all conditions) | `catch_it/run_eval_matrix.sh` |
| Tables: catch rate, CIs, paired tests, miss reasons, per-bin rates | `catch_it/analyze_results.py` |
| Tests of the predictor (incl. against MuJoCo free flight) | `catch_it/tests/test_ballistic.py` |

## 2. Prediction (Campana & Laumond 2016, Section III-A)

The paper's point-robot takeoff state (c_s, ċ_s) is replaced by the ball's estimated state. Only the unconstrained ballistic equations are used; the friction cones, velocity limits (Eqs. 4–7) and roadmap planner are not.

- Eq. 1: `c(t) = c_s + ċ_s·τ − ½·g·τ²·e_z`, τ = t − t_s. Ball position over time.
- The path lies in the vertical plane π_θ: `θ = atan2(ẏ_s, ẋ_s)`, `Θ = (cos θ, sin θ, 0)`, `x_θ = c·Θ`, `ẋ_θs = ċ_s·Θ`. The horizontal speed is constant, so `t = t_s + (x_θ − x_θs)/ẋ_θs`.
- Eq. 2: `z(x_θ)`, the height along the path.
- Eq. 3: slope `ż/ẋ_θs`, the ball's direction of travel. The hoop axis is set to the opposite direction, so the opening faces the incoming ball.

The simulator is exactly ballistic (no drag, RK4 integrator): from the true release state, Eq. 1 matches MuJoCo's ball to 3·10⁻¹⁴ m (`tests/test_ballistic.py`).

### Estimating the state from the observation

The catcher reads the same noisy observation as the policy (ball position σ 5 mm, velocity σ 0.05 m/s, 25 Hz). Two details matter:

- **The first read after release has a wrong velocity.** The env's velocity is a finite difference over the last 40 ms, which still includes the hold. Its position is used; its velocity is not.
- **A finite-difference velocity is the velocity 20 ms ago** (the middle of the window). The fit models it that way; ignoring it costs ~0.2 m/s vertically, a 10–18 cm miss.

`fit_parabola` solves a weighted least-squares problem for the state at the latest read: position reads `p_i = c + ċ·τ_i − ½g·τ_i²·e_z`, velocity reads `v_j = ċ − g·(τ_j − Δt/2)·e_z`, weighted by 1/σ. With one position and one velocity read it is exactly determined. In multi-read mode the vertical acceleration is an extra unknown with a prior N(9.81, 0.5²), so it is learned from the data once enough reads exist (the env randomizes g by ±0.5 m/s²).

## 3. Catcher (`catcher.py`)

Each control step (25 Hz):

1. **Ready (ball held).** Average the ball's resting position. Move the hoop straight ahead to the middle of the reach region (0.5 m out, 0.55 m up) with its axis turned toward the launch point, 45° up. The middle minimizes the worst-case move, since throws often cross from one side to the other (a launch-side ready pose missed those).
2. **Release.** The ball is released when it is > 3 cm from its rest position and its observed speed is > 1 m/s.
3. **Predict** (section 2), from the next read on (≈ 40–80 ms after release).
4. **Choose the catch point.** Sample the predicted path every 10 ms. Keep points inside the reach region (0.35–0.65 m out, ±45°, 0.3–0.8 m up, from the reachability study in `hoop_rl_design.md` §3) and the hoop target box. For each point, `slack = time until the ball is there − time the hoop needs` (distance at 0.8 m/s, tilt at 2 rad/s, plus 0.1 s settle; below the interface caps of 1 m/s and 2.5 rad/s). Pick the point with slack ≥ 0 needing the least motion (translation + 0.2 m/rad × tilt); if none, the one with the most slack.
5. **Servo.** The env oracle's servo law, aimed at the predicted point and axis instead of the privileged aim point, output as the same 5-D action as the policy. The hoop's own pose comes from the observation (position) and forward kinematics of the observed joint angles (orientation).

Modes:

| Mode | Ball information | Updates |
|---|---|---|
| `single_read` | one read after release (position + velocity) | none: open loop on the ball |
| `multi_read` | every read since release | re-fit and re-plan every step; keeps the chosen catch time, only re-selects if it becomes unreachable, and stops re-selecting 0.15 s before the catch |
| `perfect_state` | true release state and gravity (diagnostic) | none |

### Fairness rules

- The catcher reads only the observation dict. It never reads the throw target, arrival velocity or the episode's gravity (except `perfect_state`, which is a diagnostic).
- Same action interface as RL, so the same limits: 0.04 m and 0.1 rad per step (1 m/s, 2.5 rad/s), 0.15 m target lead, workspace box, IK, PID, action delay.
- The reach region is a fixed setting, separate from the throw sampler's `DcmmCfg.catch_zone`. A much wider assumed region (0.25–0.8 m, ±70°, 0.15–0.95 m) still caught 200/200 validation throws, so the region does not leak the aim zone.
- It never draws from `np.random`, so a seed gives the same throw and the same noise to every controller.
- Settings were tuned on validation seeds 0–499 only. Test seeds 10000–10999 are for reporting.

## 4. Experiment

**Controllers** (all through `env.step` with the same 5-D action):

| Controller | Ball information | Role |
|---|---|---|
| `random` | none (N(0,1) actions) | floor |
| `classical_single` | 1 noisy read | open-loop baseline |
| `classical_multi` | all noisy reads | main classical baseline |
| `rl` | all noisy reads | the method |
| `classical_perfect` | true release state | diagnostic: planner + arm limits only |
| `oracle` | aim point, known before release | privileged reference |

**Conditions** (`eval_catch.py --condition`, simulator changes only; the catcher's assumptions stay fixed):

| Condition | Change |
|---|---|
| `nominal` | none |
| `noise_x2`, `noise_x4` | ball position and velocity noise ×2, ×4 |
| `tracker_noise` | velocity noise 0.18 m/s (a tracker differencing 5 mm positions at 25 Hz) |
| `gravity_fixed` | gravity randomization off |
| `delay_1`, `delay_2` | 1 or 2 extra control steps before a joint target takes effect |

Not yet implemented: ball drag. It needs `random_throw` to aim with drag so throws still land in the catch zone.

**Metrics** (`analyze_results.py`): catch rate with a 95% Wilson interval; paired exact McNemar tests on the same seeds; outcome breakdown; rim hits, IK failures, hoop travel, action smoothness; for the classical modes, prediction error across the true path at the planned catch time, hoop tracking error, and the reason for each miss (not reachable in time > prediction > arm tracking > other); catch rate by flight time, launch azimuth and home-to-target distance.

## 5. Running it

From `catch_it/` with the env active:

```bash
python tests/test_ballistic.py                                   # predictor checks
python eval_catch.py --controller classical_multi                # 1000 test throws, ~30 s with 16 workers
python eval_catch.py --controller classical_single --seeds 10000:10010 --viewer --workers 1   # watch
python eval_catch.py --controller rl --checkpoint outputs/<run>/nn/<file>.pth
bash run_eval_matrix.sh                                          # everything; CHECKPOINT=... adds RL
python analyze_results.py                                        # tables; --markdown results/summary.md
```

Results go to `catch_it/results/<condition>/<controller>.csv`, with a `.json` next to each (seeds, git commit, catcher settings).

## 6. Keeping RL and classical comparable

For whoever trains the RL policy:

- **Must match between the two branches:** the observation and its noise, the action interface (step sizes, 0.15 m lead, workspace box, IK settings, PID gains and their randomization, action delay, 25 Hz), the throw sampler (catch zone, launch zone, flight time, gravity and ball randomization), the catch test, the end-of-episode conditions and the 2.5 s limit. Make env changes where both branches pull from, and evaluate both on the same commit (the `.json` records it).
- **Free to change:** reward terms and weights, PPO settings, network size, randomization used only during training (report it).
- **Final numbers come from `eval_catch.py`** on test seeds 10000–10999, not from `train_DCMM.py test=True` (that mode does not fix the throws). Pick the checkpoint on validation seeds 0–499.
- `eval_catch.py --controller rl` reads the MLP size from the checkpoint; if the observation vector changes, update the flattening in `RLController.act` to match `PPO_Hoop.obs2tensor`.
- If time allows, train 2–3 seeds and report the spread, plus training time.

## 7. Results

Test seeds 10000–10999 (1000 throws per cell), 2026-11-09. Full tables, confidence intervals, paired tests and per-bin rates: `catch_it/results/summary.md` (regenerate with `python analyze_results.py --markdown results/summary.md`). RL rows are added with `CHECKPOINT=... bash run_eval_matrix.sh`.

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

## 8. Env change

`DcmmVecEnv.py` logs two values for evaluation only (no change to the dynamics): `self.throw_time` (env time of the release) and `self.throw_flight_time`. `eval_catch.py` needs them, so the RL branch needs these 4 lines too.
