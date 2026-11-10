# HoopCatch: RL design notes

Reference for the first version of the hoop-catching setup (2026-10-30). The design choices are here so you can trace back why the robot does what it does. Every number below lives in code; the "Where" column says where to change it.

## 1. Where things live

| What | File |
|---|---|
| All tunable numbers: throws, reward weights, action scales, noise, randomization, IK settings | `catch_it/configs/env/DcmmCfg.py` |
| Environment: observation, action → IK, reward, catch/fail detection, throw sampler | `catch_it/gym_dcmm/envs/DcmmVecEnv.py` |
| Robot wrapper: model loading, hoop-pose IK call, PID | `catch_it/gym_dcmm/agents/MujocoDcmm.py` |
| IK solver (QP) | `catch_it/gym_dcmm/utils/ik_pkg/ik_arm.py` |
| PPO trainer, obs flattening order, logging | `catch_it/gym_dcmm/algs/ppo_dcmm/ppo_dcmm_hoop.py` |
| Actor-critic network | `catch_it/gym_dcmm/algs/ppo_dcmm/models_hoop.py` |
| PPO hyperparameters | `catch_it/configs/train/DcmmPPO.yaml` |
| Robot + hoop + ball model | `catch_it/assets/urdf/x1_xarm6_leap_right_object.xml` |
| How to run things | `catch_it/RUN_COMMANDS.md` |

Frame convention: every position the policy sees is **relative to the arm base** (world-aligned axes). The base is welded to the world, so this frame never moves. **+y is the robot's front**, +z is up. The arm base sits 0.40 m above the floor.

## 2. Episode timeline

Control runs at **25 Hz**: one policy step is 20 physics steps of 2 ms. An episode lasts at most 2.5 s, about 62 policy steps.

1. **Reset:** arm goes to the home pose, ball size/mass and gravity are randomized, and a throw is sampled. At home the hoop center is at (0.0, 0.39, 0.32) m and the opening faces forward-up at about 39°.
2. **Hold, 0.5–0.75 s (random):** the ball is held still at the launch point. The policy can already see it and move the hoop.
3. **Throw:** the ball gets its launch velocity and flies ballistically (no drag).
4. **End:** the episode ends at the first of catch, ball hits floor, ball hits robot, arm collision, or 2.5 s timeout.

## 3. Throw sampler (`random_throw`, `DcmmCfg.catch_zone / launch_zone / flight_time`)

Each throw is **aimed** at a target point the arm can reach:

1. Target point `p_T` in the **catch zone**: horizontal distance 0.35–0.65 m from the arm base, azimuth ±45° from +y, height 0.3–0.8 m above the arm base (0.70–1.20 m above the floor).
2. Launch point `p_0`: horizontal distance 2.0–3.0 m, azimuth ±60° from +y, height 0.8–1.6 m above the floor.
3. Flight time `T` between 0.6 and 1.0 s.
4. Launch velocity `v0 = (p_T − p_0)/T − ½·g·T`, using the episode's randomized gravity. The ball passes exactly through `p_T` at time `T`.

The resulting throws:
- Launch speeds of 3.8–5.3 m/s.
- Arrival speeds that are always descending, `v_z` from −3 to −5 m/s.
- Arrival angles 30°–65° below horizontal.

How the zone was chosen: I sampled 200k random arm poses and checked where the hoop can be and which way it can face. Within this zone, about 86% of sampled throws have a reachable hoop pose within 4 cm of `p_T` and within 30° of facing the ball, and 97% within 45°. Sparse sampling underestimates this.

Validation: the **oracle** (a scripted controller that cheats by knowing `p_T`) catches **96.5%** (193/200). So almost every throw is catchable, and the catch detection works. Rerun it whenever you change the throw or zone settings (`RUN_COMMANDS.md` §1).

## 4. State: what the policy sees (35 numbers)

Flattened in this exact order in `obs2tensor`. Changing the order or size breaks old checkpoints.

| # | Item | Dim | Source and noise |
|---|---|---|---|
| 0–2 | Hoop center position | 3 | `hoop_center` site, + N(0, 0.001) |
| 3–5 | Hoop axis: unit vector the opening faces | 3 | link6 z-axis, + N(0, 0.001) |
| 6–8 | Hoop velocity | 3 | Finite difference of the true position × 25 Hz, + N(0, 0.001) |
| 9–14 | Arm joint positions | 6 | + N(0, 0.001) |
| 15–20 | Arm joint velocities | 6 | + N(0, 0.001) |
| 21–23 | Ball position | 3 | + N(0, 0.005), stands in for a camera tracker |
| 24–26 | Ball velocity | 3 | Finite difference of the **true** position × 25 Hz, + N(0, 0.05) |
| 27–29 | Ball minus hoop (from the noisy values above) | 3 | Duplicate information that makes learning easier |
| 30–34 | Previous action, normalized | 5 | |

PPO also normalizes inputs with a running mean and standard deviation (`normalize_input: True`).

Why these choices:
- **Hoop axis instead of a quaternion.** The hoop looks the same however it spins about its own axis, so 3 numbers say everything that matters. A quaternion has the q/−q ambiguity, which confuses networks.
- **No ball history.** The policy is a plain MLP with no memory, so everything it needs must be in the current step: position and velocity of the ball.
- **Ball velocity noise varies per throw.** Velocity is differenced from true positions plus Gaussian noise whose std is drawn per throw from 0.05–0.18 m/s. 0.05 is an easy tracker; 0.18 is what a real tracker differencing its own 5 mm-noise positions at 25 Hz would give. Knobs: `k_obs_object`, `k_obs_object_vel` (a range).

## 5. Action: what the policy outputs (5 numbers in [−1, 1])

| # | Meaning | Scale (`DcmmCfg.action_scale`) | Max speed at 25 Hz |
|---|---|---|---|
| 0–2 | Hoop center move Δx, Δy, Δz (arm-base frame) | 0.04 m | 1.0 m/s |
| 3–4 | Hoop tilt about its own x and y axes | 0.1 rad | 2.5 rad/s |

Each step (`_step_mujoco_simulation`):
1. Start from the **previous target** hoop pose, not the measured one, and add the scaled action.
2. Clip the target center to at most **0.15 m** from the actual hoop center (`hoop_target_lead`), so the target can't run away from a lagging arm.
3. Clip it to the workspace box: x ±0.75, y −0.05 to 0.9, z 0 to 1.1 m (`hoop_target_low/high`).
4. Inverse kinematics turns the target pose into 6 joint targets. A PID controller per joint (randomized gains) tracks them.

IK details (`ik_config`):
- `free_ee_z_rot: True`: the IK ignores spin about the hoop axis. Joint 6 only spins the hoop about that axis, so the arm has exactly 5 useful degrees of freedom, which match the 5 actions.
- `gain: 1.0`: the original solver used 2.0. It overshot, oscillated, and failed about 70% of hoop moves; 1.0 fails about 13% from random poses and rarely during oracle episodes.
- The Jacobian is now taken at the link6 frame origin, where the error is measured. The original used the link's center of mass.
- If the IK fails, the target stays where it was and the step costs −1.

## 6. Reward (`compute_reward`, weights in `DcmmCfg.reward_weights`)

Definitions:
- `d_path`: shortest distance between the hoop center and the ball's **predicted ballistic path** (next 1.5 s, cut at the floor).
- `v_c`: the ball's predicted velocity at that closest point.
- `n`: the hoop axis.
- `a`: the normalized action.

| Term | Formula (per policy step) | Weight | When |
|---|---|---|---|
| Path progress | `w·(d_path_prev − d_path)` | 5.0 | After the throw, until the ball touches anything |
| Path near | `w·exp(−(d_path/0.1)²)` | 0.5 | Same |
| Alignment | `w·max(0, −n·v̂_c)·exp(−(d_path/0.2)²)` | 0.5 | Same, and only if \|v_c\| > 0.5 m/s |
| **Catch** | `+w`, episode ends | **100** | Ball center crosses the ring plane (see §7) |
| Rim | `+w` | −2 | Each policy step with ball–rim contact |
| Ball fail | `+w`, episode ends | −10 | Ball touches the floor, arm links, or base |
| Collision | `+w`, episode ends | −10 | Arm or hoop touches the base or floor |
| Smoothness | `−w·‖a_t − a_{t−1}‖²` | 0.02 | Always |
| IK failure | `−w` | 1.0 | When the IK fails |

PPO multiplies all rewards by `reward_scale_value: 0.1` before learning.

Why it's built this way:
- **Shaping uses the ball's path, not its current position.** In free flight the predicted path only changes when the hoop moves. So "progress" rewards exactly what the policy controls: getting the hoop onto the ball's path, early. Raw hoop-to-ball distance would be dominated by the ball flying toward the robot, and would only become informative in the last ~0.2 s.
- **The reward cheats, the policy doesn't.** The reward reads the true ball state. The policy only sees noisy observations. Using privileged information in the reward is standard and is fine at test time, because the reward isn't used then.
- **Signed alignment.** The hoop opening must face *against* the ball's motion. The original used `abs(cos)`, which would also reward facing away.
- **Shaping turns off once the ball touches anything.** After a bounce the ballistic prediction is wrong.

Approximate size per episode:
- Catch: 100.
- Path near: up to about 10.
- Alignment: up to about 10.
- Progress: up to about 2.
- Smoothness: about −9 for a random policy (σ = 1), near 0 for a smooth one.

A random policy already catches about 8% (smoke test), so PPO sees successes from the start.

## 7. Success and failure detection

Checked after **every 2 ms physics step**, not every policy step. At 5 m/s the ball moves 20 cm per policy step, which is more than the hoop is deep.

- **Catch:** `s = n·(p_ball − c)` is the ball center's signed distance to the ring plane, and `r` is its distance to the hoop axis. A catch is `s` going from > 0 to ≤ 0 while `r < 0.125 m`, the ring's inner radius. Rim physics decides whether a marginal ball actually gets in. The episode ends immediately on a catch; we don't check that the ball stays in the net.
- **Contacts:** classified by body. Rim and net are separate bodies (`hoop_rim`, `hoop_net`).
  - Ball on rim: penalty, the episode continues.
  - Ball on net or hoop: allowed.
  - Ball on floor, arm links, or base: failure.
  - Arm or hoop on base or floor: collision.
- `info["outcome"]`: 1 catch, 2 ball_floor, 3 ball_robot, 4 collision, 5 timeout. The trainer logs the fraction of each.

## 8. Domain randomization (per episode)

| What | Range | Where |
|---|---|---|
| Ball radius | 0.035–0.045 m | `object_radius` |
| Ball mass | 0.035–0.075 kg (hollow-sphere inertia) | `object_mass` |
| Air drag (damping b on the ball's free joint, force −b·v) | b 0–0.005 (a 35 g ball loses up to ~13% of its speed per second) | `object_damping`. Throws are aimed as if there were no drag, so the ball falls a bit short. |
| Gravity | fixed at −9.81 m/s² | `gravity_noise` (0) |
| Ball velocity noise std | 0.05–0.18 m/s | `k_obs_object_vel` |
| Action delay | arm follows the command from 0, 1 or 2 steps ago (0/40/80 ms) | `act_delay` (buffer length 1–3) |
| Arm PID gain scale | ×0.75–1.25 per joint | `k_arm` |
| Torque noise | ×N(1, 0.025) per physics step | `k_act` |
| Hold time before throw | 0.5–0.75 s | `object_static` |

The reward's privileged path prediction (`_predict_path`) includes the drag, so the shaping points at the ball's real path.

## 9. PPO

Unchanged from the original repo (`DcmmPPO.yaml`):
- MLP [256, 128] with ELU, separate value network.
- γ 0.99, GAE τ 0.95, lr 5e-4 with KL-adaptive schedule, clip 0.2.
- Horizon 64, minibatch 512, 6 epochs, entropy 0.
- Initial action std 1 (log σ = 0).

There is no time-out bootstrapping, but timeouts basically never happen because every ball ends on the floor or in the hoop within about 2 s.

## 10. Bugs fixed while building this

- **Ball fell at about 72% of gravity.** Changing `body_mass` at runtime needs `mujoco.mj_setConst`; without it the oracle caught only 21%. Fixed in `_reset_simulation`.
- **IK gain 2 oscillated** (see §5).
- **Ball mass never varied** (`uniform(mass[0], mass[0])`). Ball size and mass were also only set once per env process; they are now set every reset.
- **Ball damping 5e-3 to 2e-2** acted like roughly 10–35% speed loss over a flight. Now 0.
- **`configs/env/` was gitignored** by the root `env/` rule. Now tracked.
- **Every step rendered a camera image** that the policy never used. It's now skipped. With the smaller robot model, training runs at about 500 FPS with 8 envs, versus 214–276 FPS for the old Tracking task.

## 11. When it doesn't learn: what to look at

Watch in TensorBoard: `metrics/episode_success_per_step`, `episode/min_path_dist`, `episode/outcome/*`, `episode/ik_fails`, `episode/rim_hits`.

| Symptom | Likely cause | Try |
|---|---|---|
| Success stuck near the random 5–8% | Shaping too weak relative to noise or smoothness | Raise `r_path_near` / `r_path_progress`; lower `r_smooth`; check `min_path_dist` is falling |
| `min_path_dist` falls but success doesn't | Hoop on the path but facing the wrong way | Raise `r_align`, or `align_sigma` so alignment matters earlier |
| Many `ik_fails` | Policy pushes the target out of reach | Raise `r_constraint`; shrink `hoop_target_low/high` |
| Lots of `collision` | Hoop swinging into the base | Raise `r_collision`; raise the box minimum z |
| High `rim_hits` | Hoop slightly off-center | Lower `path_near_sigma` (sharper centering) |
| Jittery motion in the viewer | Smoothness too weak | Raise `r_smooth` |
| Arm too slow for the ball | Action scale | Raise `action_scale["pos"]` or `["tilt"]` |
| Everything fails after editing the throw settings | Throws no longer reachable | Rerun the oracle: it should stay above 90% |

Ideas for later versions:
- **Curriculum:** start with a narrow azimuth and slow throws, widen as success rises.
- **Realistic tracker:** compute ball velocity from the noisy positions, or stack the last 2–3 positions.
- **Baseline:** fit a parabola to the observed ball and send the hoop to the intercept point; the oracle code is a starting point.
- **Fixed-seed evaluation set:** `env.reset(seed=...)` is supported.
- **Stricter success:** require the ball to stay in the net for about 0.3 s.
