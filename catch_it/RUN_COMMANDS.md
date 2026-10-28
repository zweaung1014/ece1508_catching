# Run Commands (local setup: ThinkPad P14s Gen 5, RTX 500 Ada)

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

## 1. Train stage 1 — Tracking (base + arm reach the object)

Smoke test (~3 min, checks everything works):

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
systemd-run --user --scope -p MemoryMax=12G \
nice -n 10 python train_DCMM.py task=Tracking num_envs=8 device_id=0 \
  train.ppo.max_agent_steps=50000 output_name=smoke_test
```

Full run (auto-stops after 10 h; remove `timeout --signal=INT 10h` to run the full 25M steps):

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
systemd-run --user --scope -p MemoryMax=12G \
nice -n 10 timeout --signal=INT 10h \
python train_DCMM.py task=Tracking num_envs=8 device_id=0 output_name=track_full \
  2>&1 | tee track_full.log
```

- Stop any time with **Ctrl+C** — checkpoints are saved every epoch.
- Time estimate: `max_agent_steps / FPS` seconds (≈214–276 FPS with 8 envs here).

## 2. Monitor training

```bash
tensorboard --logdir ~/ece1508/catch_it/outputs
```

Open http://localhost:6006 → watch `metrics/episode_success_per_step` and `metrics/episode_rewards_per_step`.

## 3. Find the trained policy

```bash
ls outputs/track_full/*/*/nn/
```

- `best_reward_XX.XX.pth` — best checkpoint (use this)
- `last.pth` — most recent epoch

## 4. Test a tracking policy

> Test mode **never stops by itself** — press **Ctrl+C** when done.
> Paths containing `:` must be wrapped as `'checkpoint_tracking="..."'`.

Watch in the 3D viewer (no `MUJOCO_GL=egl` here):

```bash
python train_DCMM.py task=Tracking test=True num_envs=1 device_id=0 viewer=True \
  'checkpoint_tracking="/home/zweminhtetaung/ece1508/catch_it/outputs/track_full/2026-10-28/02:05:43/nn/best_reward_25.67.pth"'
```

Measure success rate (no window, 8 parallel sims; read `mean_success` once `Sample Length` hits 100):

```bash
MUJOCO_GL=egl python train_DCMM.py task=Tracking test=True num_envs=8 device_id=0 \
  'checkpoint_tracking="/home/zweminhtetaung/ece1508/catch_it/outputs/track_full/2026-10-28/02:05:43/nn/best_reward_25.67.pth"'
```

Pretrained baseline for comparison:

```bash
MUJOCO_GL=egl python train_DCMM.py task=Tracking test=True num_envs=8 device_id=0 \
  checkpoint_tracking=assets/models/track.pth
```

## 5. Train stage 2 — Catching (adds hand grasping, starts from tracking policy)

Leave `checkpoint_catching` empty, otherwise the tracking weights are not loaded.

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
systemd-run --user --scope -p MemoryMax=12G \
nice -n 10 timeout --signal=INT 10h \
python train_DCMM.py task=Catching_TwoStage num_envs=8 device_id=0 output_name=catch_full \
  'checkpoint_tracking="/home/zweminhtetaung/ece1508/catch_it/outputs/track_full/2026-10-28/02:05:43/nn/best_reward_25.67.pth"' \
  2>&1 | tee catch_full.log
```

## 6. Test a catching policy

Replace `<date>/<time>/...` with your path from `ls outputs/catch_full/*/*/nn/`.

```bash
python train_DCMM.py task=Catching_TwoStage test=True num_envs=1 device_id=0 viewer=True \
  'checkpoint_catching="/home/zweminhtetaung/ece1508/catch_it/outputs/catch_full/<date>/<time>/nn/best_reward_XX.XX.pth"'
```

Pretrained catching baseline:

```bash
python train_DCMM.py task=Catching_TwoStage test=True num_envs=1 device_id=0 viewer=True \
  checkpoint_catching=assets/models/catch_two_stage.pth
```

## 7. Resume an interrupted run

Only weights + obs normalizer are restored; step counter, optimizer, and best-reward reset. Set the remaining budget yourself.

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
systemd-run --user --scope -p MemoryMax=12G \
nice -n 10 python train_DCMM.py task=Tracking num_envs=8 device_id=0 output_name=track_full_resume \
  'checkpoint_tracking="/path/to/nn/last.pth"' \
  train.ppo.max_agent_steps=<remaining_steps>
```

## Gotchas

- `64 * num_envs` must be divisible by `train.ppo.minibatch_size` (default 512) → use `num_envs` = 8, 16, 24, 32…
  For `num_envs=4` add `train.ppo.minibatch_size=128`.
- Too many envs freezes the laptop (RAM/CPU). 8 is safe here; try 16 only after a short FPS test.
- No GPU / driver issue → use `device_id=-1` (CPU; training speed barely changes since simulation is CPU-bound).
- If EGL fails at startup, drop `MUJOCO_GL=egl` or force Intel EGL:
  `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json MUJOCO_GL=egl ...`
- Output folder timestamps use Asia/Shanghai time, not local time.
