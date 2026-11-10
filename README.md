# HoopCatch: Learning to Catch a Thrown Ball with a Hoop

**Author:** Zwe Min Htet Aung · ECE1508 course project

A fixed-base xArm6 with a hoop (ring + net) on its last link learns, with PPO in MuJoCo, to catch balls thrown at it. Built on the [Catch It](https://github.com/hang0610/Catch_It) codebase (Zhang et al., ICRA 2025, MIT License). The original README is in [catch_it/README.md](catch_it/README.md).

## Repository layout

| Path | Contents |
|---|---|
| [catch_it/](catch_it/) | Code: environment, PPO trainer, configs, robot model |
| [catch_it/configs/env/DcmmCfg.py](catch_it/configs/env/DcmmCfg.py) | Settings you can tune: throws, reward weights, action scales, noise, randomization |
| [catch_it/gym_dcmm/envs/DcmmVecEnv.py](catch_it/gym_dcmm/envs/DcmmVecEnv.py) | Environment: observation, action → IK, reward, catch detection |
| [catch_it/configs/train/DcmmPPO.yaml](catch_it/configs/train/DcmmPPO.yaml) | PPO hyperparameters |
| [catch_it/RUN_COMMANDS.md](catch_it/RUN_COMMANDS.md) | Full command reference |
| [docs/hoop_rl_design.md](docs/hoop_rl_design.md) | Full RL design notes and the reasoning behind them |
| [docs/after_training.md](docs/after_training.md) | How to evaluate and diagnose a trained policy |
| [catch_it/gym_dcmm/classical/](catch_it/gym_dcmm/classical/) | Classical baseline: parabola prediction (Campana & Laumond) and the catcher |
| [catch_it/eval_catch.py](catch_it/eval_catch.py) | Evaluates any controller (RL, classical, oracle, random) on fixed seeds |
| [docs/classical_baseline.md](docs/classical_baseline.md) | Classical baseline design, experiment, and how to keep RL and classical comparable |

## Setup

```bash
conda create -n dl-rl python=3.8
conda activate dl-rl
pip install torch torchvision torchaudio
cd catch_it
pip install -e .
pip install -r requirements.txt
```

## Essential commands

Run everything from `catch_it/` with the env active.

**Check the environment.** The oracle is a scripted controller that knows where the throw is aimed. It should catch about 95% of throws. If it doesn't, the environment is broken.

```bash
cd gym_dcmm/envs
python DcmmVecEnv.py --oracle --episodes 100   # headless, prints each episode's outcome
python DcmmVecEnv.py --oracle --viewer         # watch the oracle
python DcmmVecEnv.py --viewer                  # drive the hoop with the keyboard
cd ../..
```

**Train.** Start with a smoke test (about 1 minute), then do the full run:

```bash
python train_DCMM.py task=HoopCatch num_envs=8 device_id=0 train.ppo.max_agent_steps=30000 output_name=smoke_test
python train_DCMM.py task=HoopCatch num_envs=8 device_id=0 output_name=hoop_full 2>&1 | tee hoop_full.log
```

Checkpoints are saved every epoch, so you can stop with Ctrl+C at any point. `64 * num_envs` must be divisible by `train.ppo.minibatch_size` (512), so use 8, 16, 24, ... envs. Use `device_id=-1` to run on the CPU.

**Monitor training:**

```bash
tensorboard --logdir outputs   # http://localhost:6006
```

Watch `metrics/episode_success_per_step` (catch rate), `episode/min_path_dist`, and `episode/outcome/*`.

**Test a policy.** Checkpoints are in `outputs/<output_name>/<date>/<time>/nn/`. Paths contain `:`, so wrap them as shown:

```bash
# watch in the 3D viewer
python train_DCMM.py task=HoopCatch test=True num_envs=1 device_id=0 viewer=True \
  'checkpoint="outputs/hoop_full/<date>/<time>/nn/best_reward_XX.XX.pth"'

# measure the catch rate (read mean_success)
python train_DCMM.py task=HoopCatch test=True num_envs=8 device_id=0 \
  'checkpoint="outputs/hoop_full/<date>/<time>/nn/best_reward_XX.XX.pth"'
```

Memory-limited runs, resuming, and gotchas are covered in [RUN_COMMANDS.md](catch_it/RUN_COMMANDS.md).

**Compare against the classical baseline.** Every controller runs on the same 5000 test throws (seeds 10000–14999). Each throw draws its own velocity noise, air drag and action delay. Details: [docs/classical_baseline.md](docs/classical_baseline.md).

```bash
python eval_catch.py --exp scratch --controller classical_multi                        # classical catcher, re-plans every step
python eval_catch.py --exp scratch --controller rl --checkpoint outputs/<run>/nn/<file>.pth
EXP=exp02_randomized CHECKPOINT=outputs/<run>/nn/<file>.pth bash run_eval.sh          # all controllers
python analyze_results.py --exp exp02_randomized                                      # catch rates, CIs, paired tests
```

## MDP definition

Control runs at 25 Hz: each policy step is 20 physics steps of 2 ms. An episode lasts at most 2.5 s, about 62 steps.

1. The ball is held still at its launch point for 0.5–0.75 s.
2. It is thrown along an aimed ballistic arc that passes through a reachable point in front of the robot.
3. The episode ends on the first of: catch, ball hits the floor, ball hits the robot, arm collision, or timeout.

All positions are in the **arm-base frame**: +y is the robot's front and +z is up.

### Throws

Each throw is aimed at a target point `p_T`:

- **Catch zone:** 0.35–0.65 m from the arm base, within ±45° of straight ahead, 0.3–0.8 m above the arm base.
- **Launch point:** 2–3 m away, within ±60° of straight ahead, 0.8–1.6 m above the floor.
- **Flight time:** `T` is 0.6–1.0 s.
- **Launch velocity:** `v0 = (p_T − p_0)/T − ½·g·T`.

The resulting launch speeds are 3.8–5.3 m/s, and the ball is always descending when it arrives.

### State (35-dim)

| Index | Observation | Dim | Noise |
|---|---|---|---|
| 0–2 | Hoop center position | 3 | N(0, 0.001) |
| 3–5 | Hoop axis (unit vector the opening faces) | 3 | N(0, 0.001) |
| 6–8 | Hoop linear velocity | 3 | N(0, 0.001) |
| 9–14 | Arm joint positions | 6 | N(0, 0.001) |
| 15–20 | Arm joint velocities | 6 | N(0, 0.001) |
| 21–23 | Ball position | 3 | N(0, 0.005) |
| 24–26 | Ball linear velocity | 3 | N(0, 0.05) |
| 27–29 | Ball position minus hoop position | 3 | from the noisy values |
| 30–34 | Previous action | 5 | none |

The hoop's orientation is given as its axis rather than a quaternion. The hoop is symmetric about that axis, so 3 numbers are enough, and the axis avoids the q/−q ambiguity of quaternions. The policy is a memoryless MLP, so the ball's velocity is included directly. PPO normalizes inputs with a running mean and standard deviation.

### Action (5-dim, each in [−1, 1])

| Index | Action | Scale | Max rate |
|---|---|---|---|
| 0–2 | Change in hoop-center target Δx, Δy, Δz | 0.04 m | 1.0 m/s |
| 3–4 | Hoop tilt about its own x and y axes | 0.1 rad | 2.5 rad/s |

The action moves a **target hoop pose**, not the joints directly. Each step:

1. The scaled action is added to the previous target pose.
2. The target is kept within 0.15 m of the actual hoop and inside a workspace box.
3. A QP inverse-kinematics solver turns the target into 6 joint targets, with spin about the hoop axis left free.
4. A per-joint PID controller tracks the joint targets.

There is no spin action, because spinning a symmetric hoop changes nothing.

### Reward

Definitions:

- `d_path`: shortest distance from the hoop center to the ball's **predicted ballistic path** over the next 1.5 s.
- `v̂_c`: direction of the ball's predicted velocity at that closest point.
- `n`: the hoop axis.
- `a`: the normalized action.

| Term | Formula (per step) | Weight | Active |
|---|---|---|---|
| Path progress | `w·(d_path_prev − d_path)` | 5.0 | After the throw, until the ball first touches anything |
| Path near | `w·exp(−(d_path/0.1)²)` | 0.5 | Same |
| Alignment | `w·max(0, −n·v̂_c)·exp(−(d_path/0.2)²)` | 0.5 | Same |
| **Catch** | `+w`, episode ends | **100** | Ball center crosses the ring plane inside the ring |
| Rim hit | `+w` | −2 | Each step with ball–rim contact |
| Ball fail | `+w`, episode ends | −10 | Ball touches the floor, arm, or base |
| Collision | `+w`, episode ends | −10 | Arm or hoop touches the base or floor |
| Smoothness | `−w·‖a_t − a_{t−1}‖²` | 0.02 | Always |
| IK failure | `−w` | 1.0 | When the IK solver fails |

PPO scales all rewards by 0.1.

- The shaping terms use the ball's predicted path, not its current position, so they reward getting the hoop onto the path early. That is the part the policy controls.
- The reward uses the true ball state, while the policy sees only noisy observations.
- Catches are checked at every 2 ms physics step, because the ball can travel about 20 cm in one policy step.

### Domain randomization (per episode)

- Ball radius: 0.035–0.045 m.
- Ball mass: 0.035–0.075 kg.
- Gravity: −9.81 ± 0.5 m/s².
- Arm PID gains: ×0.75–1.25 per joint.
- Torque noise: ×N(1, 0.025) on each physics step.

For the full design and tuning notes, see [docs/hoop_rl_design.md](docs/hoop_rl_design.md).

## Acknowledgement

This project builds on [Catch It! Learning to Catch in Flight with Mobile Dexterous Hands](https://github.com/hang0610/Catch_It) by Y. Zhang, T. Liang, Z. Chen, Y. Ze, and H. Xu (ICRA 2025), released under the MIT License ([catch_it/LICENSE](catch_it/LICENSE)).
