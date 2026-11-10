"""
Author: Zwe Min Htet Aung
ABOUT: evaluate any HoopCatch controller on a fixed list of seeds, so every controller faces exactly the
same throws (same seed -> same throw, ball, gravity, PID gains and noise sequence).
Writes one CSV row per episode, plus a JSON file with the run settings.

Controllers:
  random            untrained-policy stand-in: N(0, 1) actions, clipped
  oracle            the env's privileged scripted controller (knows the aim point before release)
  rl                a PPO checkpoint, mean action (needs --checkpoint)
  classical_single  parabola from one read after release, open loop on the ball
  classical_multi   parabola re-fitted to every read, re-planned every step
  classical_perfect diagnostic: parabola from the true release state and gravity

Conditions (change the simulator only, see CONDITIONS): nominal, noise_x2, noise_x4, tracker_noise,
gravity_fixed, delay_1, delay_2.

Usage (from catch_it/, env active):
  python eval_catch.py --controller classical_multi                       # test seeds 10000-10999
  python eval_catch.py --controller rl --checkpoint outputs/<run>/nn/best.pth
  python eval_catch.py --controller oracle --condition noise_x2 --seeds 0:500
  python eval_catch.py --controller classical_single --seeds 10000:10010 --viewer --workers 1
Output: results/<condition>/<controller>[_<tag>].csv and .json
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
CATCH_IT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [CATCH_IT, os.path.join(CATCH_IT, "gym_dcmm")]
import argparse
import csv
import json
import math
import multiprocessing as mp
import subprocess
import time
import numpy as np
import configs.env.DcmmCfg as DcmmCfg
import configs.env.ClassicalCfg as ClassicalCfg

CONTROLLERS = ["random", "oracle", "rl", "classical_single", "classical_multi", "classical_perfect"]
# Simulator changes per test condition (DcmmCfg attribute -> value). Values are relative to the defaults.
CONDITIONS = {
    "nominal": {},
    "noise_x2": {"k_obs_object": 2 * DcmmCfg.k_obs_object, "k_obs_object_vel": 2 * DcmmCfg.k_obs_object_vel},
    "noise_x4": {"k_obs_object": 4 * DcmmCfg.k_obs_object, "k_obs_object_vel": 4 * DcmmCfg.k_obs_object_vel},
    # Velocity noise of a tracker that differences its own 5 mm-noise positions at 25 Hz (~0.18 m/s)
    "tracker_noise": {"k_obs_object_vel": 0.18},
    "gravity_fixed": {"gravity_noise": 0.0},
    # Extra control steps (40 ms each) before a new joint target reaches the PID; nominal is [1] = none
    "delay_1": {"act_delay": {"arm": [2]}},
    "delay_2": {"act_delay": {"arm": [3]}},
}
TEST_SEEDS = (10000, 11000)
OUTCOME_NAMES = ["none", "catch", "ball_floor", "ball_robot", "collision", "timeout"]


# ---------------------------------------------------------------------------- controllers
class RandomController:
    def reset(self, env, seed):
        # Own generator: drawing from np.random would shift the env's noise sequence
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        return np.clip(self.rng.normal(0.0, 1.0, 5), -1, 1)


class OracleController:
    def reset(self, env, seed):
        pass

    def act(self, obs, env):
        return env.oracle_action()


class RLController:
    """Mean action of a PPO checkpoint (same input pipeline as PPO_Hoop.obs2tensor + running_mean_std)."""
    def __init__(self, checkpoint):
        import torch
        from gym_dcmm.algs.ppo_dcmm.models_hoop import ActorCritic
        from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
        self.torch = torch
        ckpt = torch.load(checkpoint, map_location="cpu")
        sd = ckpt["model"]
        # Read the network shape from the checkpoint, so a changed MLP size still loads
        layers = sorted(int(k.split(".")[2]) for k in sd if k.startswith("actor_mlp.mlp.") and k.endswith(".weight"))
        units = [sd["actor_mlp.mlp.{}.weight".format(i)].shape[0] for i in layers]
        obs_dim = sd["actor_mlp.mlp.{}.weight".format(layers[0])].shape[1]
        self.model = ActorCritic({"actor_units": units, "actions_num": sd["mu.weight"].shape[0],
                                  "input_shape": (obs_dim,),
                                  "separate_value_mlp": any(k.startswith("value_mlp.") for k in sd)})
        self.model.load_state_dict(sd)
        self.model.eval()
        self.rms = RunningMeanStd((obs_dim,))
        self.rms.load_state_dict(ckpt["running_mean_std"])
        self.rms.eval()
        self.obs_dim = obs_dim

    def reset(self, env, seed):
        pass

    def act(self, obs, env):
        # Must match PPO_Hoop.obs2tensor (gym_dcmm/algs/ppo_dcmm/ppo_dcmm_hoop.py)
        x = np.concatenate((obs["hoop"]["pos3d"], obs["hoop"]["axis3d"], obs["hoop"]["v_lin_3d"],
                            obs["arm"]["joint_pos"], obs["arm"]["joint_vel"],
                            obs["object"]["pos3d"], obs["object"]["v_lin_3d"], obs["object"]["rel_pos3d"],
                            obs["prev_action"]))
        assert x.shape[0] == self.obs_dim, "observation size {} != checkpoint input {}".format(x.shape[0], self.obs_dim)
        with self.torch.no_grad():
            xt = self.torch.tensor(x[None], dtype=self.torch.float32)
            a = self.model.act_inference({"obs": self.rms(xt)})
        return np.clip(a[0].numpy().astype(np.float64), -1, 1)


class ClassicalController:
    def __init__(self, mode):
        self.mode = mode
        self.catcher = None

    def reset(self, env, seed):
        from gym_dcmm.classical.catcher import ClassicalCatcher
        if self.catcher is None:
            self.catcher = ClassicalCatcher(self.mode, env.Dcmm.model_arm, 1.0 / env.fps)
        self.catcher.reset()

    def act(self, obs, env):
        privileged = None
        if self.mode == "perfect_state" and env.throw_time is not None:
            privileged = {"p0": env.object_pos3d - env.arm_base_pos, "v0": env.object_vel6d[:3].copy(),
                          "g": -env.Dcmm.model.opt.gravity[2], "t_throw": env.throw_time}
        return self.catcher.act(obs, privileged)


def make_controller(name, checkpoint):
    if name == "random":
        return RandomController()
    if name == "oracle":
        return OracleController()
    if name == "rl":
        return RLController(checkpoint)
    return ClassicalController({"classical_single": "single_read", "classical_multi": "multi_read",
                                "classical_perfect": "perfect_state"}[name])


# ---------------------------------------------------------------------------- episodes
def apply_condition(condition):
    for k, v in CONDITIONS[condition].items():
        setattr(DcmmCfg, k, v)


def run_episode(env, ctrl, seed, controller_name, condition):
    obs, info = env.reset(seed=seed)
    ctrl.reset(env, seed)
    dt = 1.0 / env.fps
    hoop = [env._get_hoop_pos3d().copy()]
    axis = [env._get_hoop_axis().copy()]
    prev_a = np.zeros(5)
    smooth, ret, steps, done = 0.0, 0.0, 0, False
    while not done:
        a = np.asarray(ctrl.act(obs, env), dtype=np.float64)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        steps += 1
        ret += r
        smooth += float(np.sum((np.clip(a, -1, 1) - prev_a) ** 2))
        prev_a = np.clip(a, -1, 1)
        hoop.append(env._get_hoop_pos3d().copy())
        axis.append(env._get_hoop_axis().copy())
    hoop = np.array(hoop)

    # Throw (privileged, for analysis only)
    launch = env.object_pos3d - env.arm_base_pos
    v0 = env.object_vel6d[:3].copy()
    g = -env.Dcmm.model.opt.gravity[2]
    target = env.throw_target
    v_arr = env.throw_arrival_vel
    row = {
        "seed": seed, "controller": controller_name, "condition": condition,
        "outcome": OUTCOME_NAMES[info["outcome"]], "success": int(info["success"]),
        "steps": steps, "episode_time": round(info["env_time"], 4), "return": ret,
        "min_ball_dist": info["min_ball_dist"], "min_path_dist": info["min_path_dist"],
        "rim_hits": info["rim_hits"], "ik_fails": info["ik_fails"],
        "flight_time": env.throw_flight_time, "gravity": g, "throw_time": env.throw_time if env.throw_time is not None else np.nan,
        "static_time": env.object_static_time, "ball_radius": env.object_radius, "ball_mass": env.object_mass,
        "launch_r": math.hypot(launch[0], launch[1]), "launch_az_deg": math.degrees(math.atan2(launch[0], launch[1])),
        "launch_z": launch[2],
        "target_r": math.hypot(target[0], target[1]), "target_az_deg": math.degrees(math.atan2(target[0], target[1])),
        "target_z": target[2],
        "launch_speed": float(np.linalg.norm(v0)), "arrival_speed": float(np.linalg.norm(v_arr)),
        "arrival_angle_deg": math.degrees(math.atan2(-v_arr[2], math.hypot(v_arr[0], v_arr[1]))),
        "home_to_target": float(np.linalg.norm(target - hoop[0])),
        "hoop_travel": float(np.sum(np.linalg.norm(np.diff(hoop, axis=0), axis=1))),
        "action_smooth": smooth / max(steps, 1),
    }

    # Classical diagnostics: prediction error, tracking error, timing
    log = getattr(getattr(ctrl, "catcher", None), "log", None)
    keys = ["release_detect_delay", "decision_delay", "n_reads", "n_reselect", "t_star_after_release", "slack",
            "feasible", "in_region", "g_hat", "pred_err", "pred_err_perp", "hoop_err_at_tstar", "axis_err_at_tstar_deg"]
    row.update({k: np.nan for k in keys})
    if log is not None and env.throw_time is not None and not np.isnan(log["t_star"]):
        from gym_dcmm.classical.ballistic import Parabola
        truth = Parabola(launch, v0, env.throw_time, g)
        t_star = log["t_star"]
        p_true = truth.position(t_star)
        d_true = truth.direction(t_star)
        err = log["p_star"] - p_true
        k = int(np.clip(round(t_star / dt), 0, len(hoop) - 1))   # hoop pose at the planned catch time (or the last one)
        row.update({
            "release_detect_delay": log["release_detect_time"] - env.throw_time,
            "decision_delay": log["decision_time"] - env.throw_time,
            "n_reads": log["n_reads"], "n_reselect": log["n_reselect"],
            "t_star_after_release": t_star - env.throw_time, "slack": log["slack"],
            "feasible": int(log["feasible"]), "in_region": int(log["in_region"]), "g_hat": log["g_hat"],
            "pred_err": float(np.linalg.norm(err)),
            "pred_err_perp": float(np.linalg.norm(err - np.dot(err, d_true) * d_true)),  # miss across the ball's path
            "hoop_err_at_tstar": float(np.linalg.norm(hoop[k] - log["p_star"])),
            "axis_err_at_tstar_deg": math.degrees(math.acos(np.clip(np.dot(axis[k], log["n_star"]), -1, 1))),
        })
    return row


# ---------------------------------------------------------------------------- workers
_W = {}


def _init_worker(controller, condition, checkpoint, viewer):
    import contextlib, io
    apply_condition(condition)
    with contextlib.redirect_stdout(io.StringIO()):   # the env and the normalizer print on construction
        from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
        _W["env"] = DcmmVecEnv(task="HoopCatch", object_name="object", render_per_step=False, viewer=viewer,
                               imshow_cam=False, camera_name=["top"], render_mode="rgb_array",
                               env_time=2.5, steps_per_policy=20)
        _W["ctrl"] = make_controller(controller, checkpoint)
    _W["controller"], _W["condition"] = controller, condition


def _run_chunk(seeds):
    return [run_episode(_W["env"], _W["ctrl"], s, _W["controller"], _W["condition"]) for s in seeds]


def parse_seeds(text):
    if ":" in text:
        a, b = text.split(":")
        return list(range(int(a), int(b)))
    return [int(s) for s in text.split(",")]


def git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CATCH_IT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=CATCH_IT) != 0
        return commit, dirty
    except Exception:
        return "unknown", True


def main():
    parser = argparse.ArgumentParser(description="Evaluate a HoopCatch controller on fixed seeds")
    parser.add_argument("--controller", required=True, choices=CONTROLLERS)
    parser.add_argument("--condition", default="nominal", choices=list(CONDITIONS))
    parser.add_argument("--seeds", default="{}:{}".format(*TEST_SEEDS),
                        help="a:b (range) or a,b,c. Test set 10000:11000, validation 0:500")
    parser.add_argument("--checkpoint", default="", help="PPO checkpoint (.pth) for --controller rl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--viewer", action="store_true", help="watch in the MuJoCo viewer (forces --workers 1)")
    parser.add_argument("--tag", default="", help="suffix of the output file name")
    parser.add_argument("--out_dir", default=os.path.join(CATCH_IT, "results"))
    args = parser.parse_args()
    if args.controller == "rl" and not args.checkpoint:
        parser.error("--controller rl needs --checkpoint")
    if args.checkpoint:
        args.checkpoint = os.path.abspath(args.checkpoint)
    seeds = parse_seeds(args.seeds)
    workers = 1 if args.viewer else max(1, min(args.workers, len(seeds)))

    t0 = time.time()
    rows = []
    if workers == 1:
        _init_worker(args.controller, args.condition, args.checkpoint, args.viewer)
        for i, s in enumerate(seeds):
            rows.append(run_episode(_W["env"], _W["ctrl"], s, args.controller, args.condition))
            print("seed {}: {} | catch rate {:.3f} ({}/{})".format(
                s, rows[-1]["outcome"], np.mean([r["success"] for r in rows]), i + 1, len(seeds)), flush=True)
    else:
        n_chunks = min(len(seeds), workers * 8)
        chunks = [list(c) for c in np.array_split(seeds, n_chunks) if len(c)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(args.controller, args.condition, args.checkpoint, False)) as pool:
            for chunk_rows in pool.imap_unordered(_run_chunk, chunks):
                rows.extend(chunk_rows)
                print("{}/{} episodes | catch rate {:.3f} | {:.0f} s".format(
                    len(rows), len(seeds), np.mean([r["success"] for r in rows]), time.time() - t0), flush=True)
    rows.sort(key=lambda r: r["seed"])

    out_dir = os.path.join(args.out_dir, args.condition)
    os.makedirs(out_dir, exist_ok=True)
    name = args.controller + ("_" + args.tag if args.tag else "")
    csv_path = os.path.join(out_dir, name + ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    commit, dirty = git_info()
    meta = {
        "controller": args.controller, "condition": args.condition, "overrides": str(CONDITIONS[args.condition]),
        "seeds": args.seeds, "episodes": len(rows), "checkpoint": args.checkpoint,
        "git_commit": commit, "git_dirty": dirty, "wall_time_s": round(time.time() - t0, 1),
        "catch_rate": float(np.mean([r["success"] for r in rows])),
        "classical_cfg": {k: (v.tolist() if isinstance(v, np.ndarray) else
                              {kk: vv.tolist() for kk, vv in v.items()} if isinstance(v, dict) else v)
                          for k, v in vars(ClassicalCfg).items()
                          if not k.startswith("_") and isinstance(v, (int, float, bool, np.ndarray, dict))},
    }
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    outcomes = {n: sum(r["outcome"] == n for r in rows) for n in OUTCOME_NAMES[1:]}
    print("\n{} | {} | {} episodes | catch rate {:.3f} | {}".format(
        args.controller, args.condition, len(rows), meta["catch_rate"], outcomes))
    print("saved", csv_path)


if __name__ == "__main__":
    main()
