"""
Author: Zwe Min Htet Aung
ABOUT: a check of the PPO settings alone, not an experiment. The policy's object.rel_pos3d (3 of its 35 inputs)
is replaced by the throw's aim point, the point the ball will pass through, so the right policy is easy to
learn: move the hoop there. If PPO and its settings work, the catch rate climbs well above parking the hoop
(~12%) and stays up. On larry-catch with the old settings (KL-adaptive learning rate at its 1e-3 cap,
1024 steps per update) it peaked near 60% and ended at 31.5%.
Everything else (env, noise, rewards, trainer) is exactly as in training; train_DCMM.py is not edited.

Usage (from catch_it/):
  python tools/sanity_aim_obs.py train train.ppo.max_agent_steps=2000000 output_name=exp05_sanity   # Hydra overrides
  python tools/sanity_aim_obs.py eval --checkpoint outputs/exp05_sanity/<date>/<time>/nn/last.pth --min 0.5
eval scores the policy's mean action on validation throws (default seeds 0-199) with the aim point
in its input, prints the catch rate, and exits with status 1 if it is below --min.
"""
import os, sys
CATCH_IT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [CATCH_IT, os.path.join(CATCH_IT, "gym_dcmm")]
os.chdir(CATCH_IT)
import numpy as np
import gymnasium as gym


def with_aim_point(obs, env):
    """The observation with object.rel_pos3d replaced by the throw's aim point (arm-base frame)."""
    obs = dict(obs)
    obj = dict(obs["object"])
    obj["rel_pos3d"] = env.throw_target.astype(np.float32).copy()
    obs["object"] = obj
    return obs


class AimPointObs(gym.ObservationWrapper):
    def observation(self, obs):
        return with_aim_point(obs, self.env.unwrapped)


def train(overrides):
    import train_DCMM
    make_vec = gym.make_vec

    def make_vec_with_aim_point(*args, **kwargs):
        kwargs["wrappers"] = list(kwargs.get("wrappers") or []) + [AimPointObs]
        return make_vec(*args, **kwargs)

    gym.make_vec = make_vec_with_aim_point    # train_DCMM calls gym.make_vec
    sys.argv = [sys.argv[0], "--config-path=" + os.path.join(CATCH_IT, "configs")] + overrides
    train_DCMM.main()


_W = {}


def _init(checkpoint):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
        import eval_catch
        _W["env"] = DcmmVecEnv(task="HoopCatch", viewer=False, imshow_cam=False, env_time=2.5, steps_per_policy=20)
        _W["ctrl"] = eval_catch.RLController(checkpoint)


def _episode(seed):
    env, ctrl = _W["env"], _W["ctrl"]
    obs, _ = env.reset(seed=seed)
    ctrl.reset(env, seed)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(ctrl.act(with_aim_point(obs, env), env))
        done = term or trunc
    return int(info["outcome"] == 1)


def evaluate(argv):
    import argparse
    import multiprocessing as mp
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seeds", default="0:200")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min", type=float, default=0.0, help="exit with status 1 below this catch rate")
    a = p.parse_args(argv)
    lo, hi = (int(v) for v in a.seeds.split(":"))
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(os.path.abspath(a.checkpoint),)) as pool:
        caught = pool.map(_episode, range(lo, hi), chunksize=5)
    rate = float(np.mean(caught))
    ok = rate >= a.min
    print("sanity (aim point visible): catch {:.3f} on {} validation throws (seeds {}) | need >= {:.2f}: {}".format(
        rate, len(caught), a.seeds, a.min, "PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval"):
        sys.exit(__doc__)
    if sys.argv[1] == "train":
        train(sys.argv[2:])
    else:
        evaluate(sys.argv[2:])
