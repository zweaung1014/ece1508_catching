# exp05: Pengze's PPO with ball sensor noise only, steadier PPO settings

Author: Zwe Min Htet Aung

**Status:** done (2026-11-19). The two sections below, Results and Where to find everything, were added after the run. Every section from Why onward was written, and the protocol fixed, before any exp05 result was seen; those sections are unchanged.

## Results (added 2026-11-19, after the run)

**RL tied the classical catcher. Under the decision rule, it does not beat it.** Test throws 10000–14999, each controller run once:

| Controller | Catch | 95% CI |
|---|---|---|
| **rl** (snapshot at 14M steps, picked on validation) | **99.8%** (4990/5000) | 99.6–99.9% |
| classical_multi | 99.8% (4989/5000) | 99.6–99.9% |
| classical_perfect (true ball state) | 99.8% (4990/5000) | 99.6–99.9% |
| oracle | 95.7% | 95.1–96.2% |
| classical_single (one reading) | 31.8% | 30.6–33.1% |
| random | 4.8% | 4.2–5.4% |

- **Paired McNemar test against classical_multi:** 10 throws only RL caught, 9 only classical_multi caught, p = 1. There is no difference.
- **The task leaves no room to win.** `classical_perfect`, which reads the true ball state, also catches 99.8%.
- **The two controllers miss different throws.** Only seed 14149 is missed by both (`figures/misses_map.png`).
  - classical_multi's 11 misses are throws it could not reach in time.
  - 9 of RL's 10 misses were aimed more than 0.5 m from the arm base, where 49% of all throws land.
  - 6 of RL's 10 misses had velocity noise std above 0.35 m/s, where 16% of all throws fall.
- **One noisy frame is enough for RL.** With a single frame RL matches classical_multi at every noise level. classical_single, which also uses one reading, drops from 62% to 16% as the velocity noise std goes from 0.1 to 0.4 m/s (`figures/catch_vs_noise.png`).
- **Secondary measures,** which are not part of the decision rule (`figures/how_they_move.png`). Compared with classical_multi, RL has:
  - far fewer IK failures: 0.02 vs 0.48 per throw;
  - less hoop travel: 0.88 vs 1.00 m;
  - smaller action changes: 0.68 vs 0.92.
- **Training** (`training_curves.png`, `figures/learning_curve.png`):
  - **No under-fitting:** 98.4% on validation by 5M steps.
  - **No over-fitting:** validation stays at or above the training catch rate throughout.
  - **Stable:** 99.2–100% on validation from 6M to 25M steps, unlike exp03 and exp04, which rose and then fell.
  - **To watch:** late in the run the KL climbs to the early-stop line (0.032) as the exploration std falls to 0.1. A decaying learning rate would be the next change.
- **The sanity gate passed:** 100% with the aim point visible, after 2M steps (`sanity.txt`).
- **Fixed after the run, in plotting and the summary only:**
  - `24b17bc`: the curves plot crashed at 500/500.
  - `5c58387`: `summary.md` listed the curve CSVs as controllers.

  Training, the checkpoint pick and the test run were not touched.

## Where to find everything

- **In git** (branch `larry-ball-noise`, folder `catch_it/results/exp05_ball_noise/`):
  - this README;
  - `summary.md`, the full tables;
  - `figures/`, the report figures;
  - `training_curves.png`, every training signal;
  - `validation.md` and `picked.txt`;
  - `rl_policy.pth`;
  - one CSV and one JSON per controller.
- **Only on the training machine** (git-ignored, under `catch_it/`):
  - `outputs/rl_exp05/2026-11-19/08-48-56/`: the run, with all snapshots in `nn/`, TensorBoard in `tb/`, `config.yaml` and `git.txt`;
  - `outputs/rl_exp05/train.log`;
  - `outputs/exp05_val/`: validation results per snapshot, plus classical_multi on the validation throws;
  - `pipeline.log` in this folder.
- **Redraw the figures:**
  - `python plot_results.py --exp exp05_ball_noise --val_ref outputs/exp05_val/val_ref/classical_multi.json`
  - `python plot_training.py ...` (see `run_exp05.sh` stage 6)

## Why

- **Building on Pengze's RL setup.** This continues the team's noise-free RL setup (branch `pengze-catch`, commit 70a1589):
  - plain PPO; one observation frame (35 numbers); a 5-number step action (hoop x, y, z and two tilts);
  - no disturbances.
- **What changes:**
  - **Noise:** the ball readings get exp03's sensor noise. Nothing else is disturbed.
  - **PPO settings:** steadier ones.
- **Why the settings change.** On `larry-catch`, exp03 and exp04 showed that the old settings lose what PPO learns:
  - The learning rate was KL-adaptive and sat at its 1e-3 cap.
  - Each update used only 1024 steps, about 25 throws.
  - Even with the throw's aim point handed to the policy, the catch rate peaked near 60% and ended at 31.5%.

## The setting

| | exp05 | Pengze (`pengze-catch`) | exp03 (`larry-catch`) |
|---|---|---|---|
| Ball position noise std, drawn per throw | 5–20 mm | 0 | 5–20 mm |
| Ball velocity noise std, drawn per throw | 0.1–0.4 m/s | 0 | 0.1–0.4 m/s |
| Arm joint / hoop reading noise | 0 | 0 | 0.001 rad / 1 mm |
| Torque noise | 0 | 0 | 10% |
| Action delay | none | none | 0–4 steps (0–160 ms) |
| Air drag, wind | none | none | up to 35% speed loss/s, 0.04 N |
| PID gain scale per joint | 0.75–1.25 | 0.75–1.25 | 0.5–1.5 |
| Ball size and mass, throws, rewards, IK | Pengze's | | |

`tests/test_ball_noise.py` checks that with the ball noise set to zero, this env reproduces Pengze's env step for step.

## PPO settings (`configs/train/DcmmPPO.yaml`)

| | exp05 | Pengze |
|---|---|---|
| Learning rate | 3e-4, fixed | 5e-4, KL-adaptive |
| Steps per update | 512 × 16 envs = 8192 (~200 throws) | 64 × 16 = 1024 |
| Minibatch, epochs | 2048, 5 | 512, 6 |
| Training steps | 25M | 25M |

Unchanged from Pengze:
- the trainer fixes in commit 70a1589;
- an update stops after an epoch whose KL exceeds 0.032;
- the network, the rewards, and the one-frame input.

## Expectation (written before the run)

In exp03, `classical_multi` caught 98.6% of the 944 test throws that had no delay. With no delay here, it will likely catch about 99%.
- RL can at best tie it, so a win under the rule below is very unlikely.
- What exp05 measures is how close Pengze's PPO gets once the ball readings are noisy.
- A single noisy frame cannot be averaged: `classical_single`, which uses one reading, caught 23% in exp03. So the one-frame policy may struggle.

## Protocol

1. **Smoke test:** 30K steps.
2. **Sanity gate** (`tools/sanity_aim_obs.py`): the policy's `object.rel_pos3d` input is replaced by the throw's aim point, so the task is easy.
   - Train 2M steps with the exp05 settings, then score the final policy on validation throws 0–199.
   - The pipeline stops below 50%. That means the settings are still unstable, and the 10 h run would be wasted.
   - Results: `sanity.txt`, `sanity_curves.png`.
3. **Baselines** on the test throws (seeds 10000–14999): `random`, `oracle`, `classical_single`, `classical_multi`, `classical_perfect`.
4. **Train:** 16 envs, 25M steps, seed 1 (`outputs/rl_exp05/`).
5. **Validate:**
   - every snapshot (~1M steps apart, plus `best_reward_*` and `last`) on validation throws 0–499 → `validation.md`;
   - `classical_multi` on the same throws, as the reference line in the curves.
   - The highest catch rate is picked; a tie goes to the earlier snapshot.
6. **Training curves:** `training_curves.png`, with the numbers in `training_curves_train.csv` and `training_curves_val.csv`.
7. **Freeze** the pick to `rl_policy.pth`, then **test once** on the test throws → `rl.csv`.
8. **Summary** (`summary.md`): catch rates with 95% CIs, the paired McNemar test against `classical_multi`, and catch rate by noise level.

**Decision rule, fixed now:** RL beats the classical catcher only if `rl`'s test catch rate is above `classical_multi`'s and the paired test gives p < 0.05.

## Reading the training curves (`training_curves.png`)

Panel 1 compares two catch rates:
- **Training:** the stochastic policy on its last 200 training throws.
- **Validation:** the mean action on 500 held-out throws, at every snapshot.

How to read it:
- **Both low and flat:** under-fitting. The policy has not learned the task.
- **Training clearly above validation:** the policy depends on its exploration noise or on the training throws (over-fitting). Every training episode draws a new random throw, so this gap is usually small in this task.
- **Both rise, then fall:** unstable training, not over-fitting. This is what happened in exp03 and exp04.

The other panels show:
- episode reward;
- value and policy losses;
- the KL of each update, with the early-stop line;
- the exploration std.

## Files

| File | What |
|---|---|
| `STATUS`, `pipeline.log` | pipeline progress |
| `sanity.txt`, `sanity_curves.png` | the settings check (stage 2) |
| `validation.md`, `picked.txt` | every checkpoint's validation catch rate, and the pick |
| `training_curves.png`, `training_curves_*.csv` | training curves (stage 6) |
| `rl_policy.pth` | the frozen pick. `*.pth` is git-ignored; commit with `git add -f`. |
| `<controller>.csv` / `.json` | one row per test throw / settings and commit |
| `summary.md` | all tables |
| `figures/test_catch_rates.png` | test catch rate of every controller (made by `plot_results.py`, after the run) |
| `figures/catch_vs_noise.png` | catch rate by position / velocity noise level |
| `figures/misses_map.png` | where the missed throws were aimed, and their velocity noise |
| `figures/how_they_move.png` | IK failures, hoop travel, action change (secondary) |
| `figures/learning_curve.png` | RL catch rate during training: training vs validation |

## Reproduce

```bash
cd catch_it
systemd-run --user --unit=exp05 -p MemoryMax=11G -p MemorySwapMax=0 --nice=10 \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --working-directory=$PWD systemd-inhibit --what=sleep:idle --why=exp05 bash run_exp05.sh
```

Test only, from the frozen policy:

```bash
python eval_catch.py --exp exp05_ball_noise --controller rl --checkpoint results/exp05_ball_noise/rl_policy.pth
```
