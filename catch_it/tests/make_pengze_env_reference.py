"""
Author: Zwe Min Htet Aung
ABOUT: records a reference rollout of Pengze's original env (commit 70a1589 on pengze-catch, every
disturbance off), so tests/test_ball_noise.py can check that the ball-noise env is the same env when its
noise ranges are set to zero. The actions come from their own generator (not np.random), so they do not
depend on the env's random draws.

Run from catch_it/ at commit 70a1589 only (git switch --detach 70a1589):
  python tests/make_pengze_env_reference.py
"""
import os, sys, io, contextlib, subprocess
CATCH_IT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [CATCH_IT, os.path.join(CATCH_IT, "gym_dcmm")]
import numpy as np

SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.join(CATCH_IT, "tests", "data", "pengze_env_reference.npz")


def flat(obs):
    return np.concatenate((obs["hoop"]["pos3d"], obs["hoop"]["axis3d"], obs["hoop"]["v_lin_3d"],
                           obs["arm"]["joint_pos"], obs["arm"]["joint_vel"],
                           obs["object"]["pos3d"], obs["object"]["v_lin_3d"], obs["object"]["rel_pos3d"],
                           obs["prev_action"]))


def rollout(env, seed):
    """Observations, rewards and outcome of one episode under a fixed random action sequence."""
    rng = np.random.default_rng(1000 + seed)
    obs, _ = env.reset(seed=seed)
    frames, rewards, done = [flat(obs)], [], False
    while not done:
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 5))
        frames.append(flat(obs)); rewards.append(r)
        done = term or trunc
    return np.array(frames), np.array(rewards), info["outcome"]


def make_env():
    os.chdir(CATCH_IT)
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    with contextlib.redirect_stdout(io.StringIO()):
        return DcmmVecEnv(task="HoopCatch", viewer=False, imshow_cam=False, env_time=2.5, steps_per_policy=20)


if __name__ == "__main__":
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=CATCH_IT, text=True).strip()
    assert commit == "70a1589", "run this at Pengze's commit 70a1589, not " + commit
    env = make_env()
    data = {}
    for s in SEEDS:
        f, r, o = rollout(env, s)
        data["obs_%d" % s], data["rew_%d" % s], data["outcome_%d" % s] = f, r, o
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **data)
    print("saved", OUT, {s: (len(data["rew_%d" % s]), int(data["outcome_%d" % s])) for s in SEEDS})
