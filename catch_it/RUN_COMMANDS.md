# Run Commands (local setup: ThinkPad P14s Gen 5, RTX 500 Ada)

Task: `HoopCatch` — fixed-base xArm6 catches a thrown ball with the hoop.
Design notes (states, actions, rewards, throws, what to tune): `../docs/hoop_rl_design.md`.

Always start from the project folder with the conda env active:

```bash
cd ~/ece1508/catch_it
conda activate dl-rl
```

## 0. Sanity checks

```bash
nvidia-smi                                   # GPU visible?
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pgrep -af train_DCMM                         # leftover processes? kill with: pkill -f train_DCMM
```

## 1. Check the environment (no learning)

Run from `gym_dcmm/envs/`. The oracle is a privileged scripted controller that drives the hoop to the
throw's aim point; it should catch ~95% of the throws. If it doesn't, the env is broken, not the policy.

```bash
cd gym_dcmm/envs
python DcmmVecEnv.py --oracle --episodes 100            # headless, prints the outcome of every episode
python DcmmVecEnv.py --oracle --viewer                  # watch it (real time)
python DcmmVecEnv.py --viewer                           # drive the hoop with the keyboard
python DcmmVecEnv.py --oracle --episodes 3 --print_reward   # print every reward term each step
cd ../..
```

Keyboard: arrows = hoop x/y, numpad `+`/`-` = z, numpad `7`/`9` = tilt about hoop x, `4`/`6` = tilt about hoop y.

## 2. Train

Smoke test (~1 min, checks everything works):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
systemd-run --user --scope -p MemoryMax=12G \
nice -n 10 python train_DCMM.py task=HoopCatch num_envs=8 device_id=0 \
  train.ppo.max_agent_steps=30000 output_name=smoke_test
```

Full run (auto-stops after 10 h; remove `timeout --signal=INT 10h` to run the full 25M steps):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
systemd-run --user --scope -p MemoryMax=12G \
nice -n 10 timeout --signal=INT 10h \
python train_DCMM.py task=HoopCatch num_envs=8 device_id=0 output_name=hoop_full \
  2>&1 | tee hoop_full.log
```

- Stop any time with **Ctrl+C** — checkpoints are saved every epoch.
- Time estimate: `max_agent_steps / FPS` seconds (≈500 FPS with 8 envs here → 25M steps ≈ 14 h).
- No camera rendering happens during training, so `MUJOCO_GL=egl` is not needed anymore.

## 3. Monitor training

```bash
tensorboard --logdir ~/ece1508/catch_it/outputs
```

Open http://localhost:6006 and watch:

- `metrics/episode_success_per_step` — catch rate (the number that matters)
- `metrics/episode_rewards_per_step` — total reward per episode
- `episode/min_path_dist` — closest the hoop got to the ball's predicted path (should shrink towards 0)
- `episode/outcome/*` — fraction of episodes ending in catch / ball_floor / ball_robot / collision / timeout
- `episode/rim_hits`, `episode/ik_fails` — per-episode counts

## 4. Find the trained policy

```bash
ls outputs/hoop_full/*/*/nn/
```

- `best_reward_XX.XX.pth` — best checkpoint (use this)
- `last.pth` — most recent epoch

## 5. Test a policy

Full post-training walkthrough (reading the results, comparing checkpoints, diagnosing failures): `../docs/after_training.md`.

> Test mode stops by itself after `train.ppo.max_test_steps` (10000) agent steps; Ctrl+C also works.
> Paths containing `:` must be wrapped as `'checkpoint="..."'`.

Watch in the 3D viewer:

```bash
python train_DCMM.py task=HoopCatch test=True num_envs=1 device_id=0 viewer=True \
  'checkpoint="/home/zweminhtetaung/ece1508/catch_it/outputs/hoop_full/<date>/<time>/nn/best_reward_XX.XX.pth"'
```

Measure the catch rate (no window, 8 parallel sims; read `mean_success` once `Sample Length` hits 100):

```bash
python train_DCMM.py task=HoopCatch test=True num_envs=8 device_id=0 \
  'checkpoint="/home/zweminhtetaung/ece1508/catch_it/outputs/hoop_full/<date>/<time>/nn/best_reward_XX.XX.pth"'
```

## 6. Resume an interrupted run

Only weights + obs normalizer are restored; step counter, optimizer, and best-reward reset. Set the remaining budget yourself.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
systemd-run --user --scope -p MemoryMax=12G \
nice -n 10 python train_DCMM.py task=HoopCatch num_envs=8 device_id=0 output_name=hoop_full_resume \
  'checkpoint="/path/to/nn/last.pth"' \
  train.ppo.max_agent_steps=<remaining_steps>
```

## exp05: ball noise, steadier PPO settings (whole pipeline, ~14 h)

What and why: `results/exp05_ball_noise/README.md`. One background job runs everything: smoke test, settings check,
baselines, training, checkpoint choice on the validation throws, training curves, test, summary.
It survives closing the terminal, is memory-capped (11G), and keeps the laptop from sleeping (keep it plugged in).

```bash
systemd-run --user --unit=exp05 -p MemoryMax=11G -p MemorySwapMax=0 --nice=10 \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --working-directory=$PWD systemd-inhibit --what=sleep:idle --why=exp05 bash run_exp05.sh

cat results/exp05_ball_noise/STATUS                        # running <stage> / done / failed <stage>
tail -f results/exp05_ball_noise/pipeline.log
grep "Agent Steps" outputs/rl_exp05/train.log | tail -n 3  # training progress (catch rate, LR, exploration std)
systemctl --user stop exp05                                 # stop; rerunning skips the stages already done
```

Results: `results/exp05_ball_noise/` (`summary.md`, `validation.md`, `training_curves.png`, `sanity.txt`).
Training curves of any run: `python plot_training.py --run_dir outputs/<name>/<date>/<time> --out <file>`.

## Gotchas

- `horizon_length * num_envs` (512 * num_envs) must be divisible by `train.ppo.minibatch_size` (2048) → use `num_envs` = 4, 8, 12, 16…
- Too many envs freezes the laptop (RAM/CPU). 8 is safe here; try 16 only after a short FPS test.
- No GPU / driver issue → use `device_id=-1` (CPU; training speed barely changes since simulation is CPU-bound).
- Checkpoints from the old Tracking/Catching tasks do not load (different observation/action sizes).
- Output folder timestamps use Asia/Shanghai time, not local time.
