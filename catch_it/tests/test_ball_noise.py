"""
Author: Zwe Min Htet Aung
ABOUT: checks of the exp05 env (Pengze's env plus per-throw ball sensor noise) and of the RL evaluation.
1. The ball noise std of every throw is inside the DcmmCfg ranges, is the same for the same seed, and
   changes between throws. Arm and hoop readings stay exact.
2. With the ball noise ranges set to zero, the env is Pengze's original env: a fixed-action rollout gives the
   same observations, rewards and outcome as the reference recorded at his commit 70a1589
   (tests/make_pengze_env_reference.py).
3. eval_catch.RLController, loading a checkpoint saved by the trainer, takes the same action as the
   trainer's own model_act(inference=True) on a real rollout.

Run from catch_it/:  python tests/test_ball_noise.py     (pytest also works if installed)
"""
import os, sys, tempfile
CATCH_IT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [CATCH_IT, os.path.join(CATCH_IT, "gym_dcmm")]
import numpy as np
import configs.env.DcmmCfg as DcmmCfg
sys.path.insert(0, os.path.join(CATCH_IT, "tests"))
from make_pengze_env_reference import OUT as REFERENCE, SEEDS, flat, make_env, rollout


def test_noise_draws():
    env = make_env()
    draws = []
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        assert DcmmCfg.k_obs_object[0] <= env.k_obs_object <= DcmmCfg.k_obs_object[1]
        assert DcmmCfg.k_obs_object_vel[0] <= env.k_obs_object_vel <= DcmmCfg.k_obs_object_vel[1]
        draws.append((env.k_obs_object, env.k_obs_object_vel))
        # Arm joint readings are exact (k_obs_arm = 0); the ball reading is not
        assert np.array_equal(obs["arm"]["joint_pos"], env.Dcmm.data.qpos[env.Dcmm.arm_qpos_adr])
        assert not np.allclose(obs["object"]["pos3d"], env._get_object_pos3d())
    env.reset(seed=3)
    assert (env.k_obs_object, env.k_obs_object_vel) == draws[3], "same seed, different noise std"
    assert len(set(draws)) == len(draws), "the noise std should change from throw to throw"
    print("noise draws ok: pos std {:.4f}-{:.4f} m, vel std {:.3f}-{:.3f} m/s over 20 throws".format(
        min(d[0] for d in draws), max(d[0] for d in draws), min(d[1] for d in draws), max(d[1] for d in draws)))


def test_zero_noise_is_pengze_env():
    ref = np.load(REFERENCE)
    saved = DcmmCfg.k_obs_object, DcmmCfg.k_obs_object_vel
    DcmmCfg.k_obs_object, DcmmCfg.k_obs_object_vel = np.zeros(2), np.zeros(2)
    try:
        env = make_env()
        for s in SEEDS:
            frames, rewards, outcome = rollout(env, s)
            assert frames.shape == ref["obs_%d" % s].shape, "seed {}: episode length changed".format(s)
            assert np.allclose(frames, ref["obs_%d" % s], atol=1e-9), "seed {}: observations differ".format(s)
            assert np.allclose(rewards, ref["rew_%d" % s], atol=1e-9), "seed {}: rewards differ".format(s)
            assert outcome == int(ref["outcome_%d" % s])
    finally:
        DcmmCfg.k_obs_object, DcmmCfg.k_obs_object_vel = saved
    print("zero noise = Pengze's env: {} reference episodes identical".format(len(SEEDS)))


def test_eval_matches_trainer():
    import torch
    from omegaconf import OmegaConf
    from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_hoop import PPO_Hoop
    import eval_catch
    cfg = OmegaConf.load(os.path.join(CATCH_IT, "configs", "train", "DcmmPPO.yaml"))
    torch.manual_seed(0)
    agent = PPO_Hoop.__new__(PPO_Hoop)       # only the parts model_act and save need
    from gym_dcmm.algs.ppo_dcmm.models_hoop import ActorCritic
    from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
    agent.device = "cpu"
    agent.normalize_input = agent.normalize_value = True
    agent.model = ActorCritic({"actor_units": list(cfg.network.mlp.units), "actions_num": 5, "input_shape": (35,),
                               "separate_value_mlp": True, "log_std_min": cfg.network.log_std_min,
                               "log_std_max": cfg.network.log_std_max,
                               "raw_mean_soft_bound": cfg.network.raw_mean_soft_bound})
    agent.running_mean_std = RunningMeanStd((35,))
    agent.value_mean_std = RunningMeanStd((1,))
    # Non-trivial normalization statistics, as after training
    agent.running_mean_std.update(torch.randn(256, 35) * 3 + 1)
    agent.running_mean_std.eval(); agent.value_mean_std.eval(); agent.model.eval()
    agent.optimizer = torch.optim.Adam(agent.model.parameters())
    agent.agent_steps, agent.epoch_num, agent.best_rewards, agent.last_lr = 0, 0, 0.0, 3e-4
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck")
        agent.save(path)
        ctrl = eval_catch.RLController(path + ".pth")
    env = make_env()
    steps = 0
    for seed in [0, 1]:
        obs, _ = env.reset(seed=seed); ctrl.reset(env, seed); done = False
        while not done:
            batch = {k: ({kk: vv[None] for kk, vv in v.items()} if isinstance(v, dict) else v[None]) for k, v in obs.items()}
            a_train = agent.model_act({"obs": agent.obs2tensor(batch)}, inference=True)["actions"][0].numpy()
            a_eval = ctrl.act(obs, env)
            assert np.allclose(np.clip(a_train, -1, 1), a_eval, atol=1e-6), "train and eval actions differ"
            obs, _, term, trunc, _ = env.step(a_eval)
            done = term or trunc; steps += 1
    print("eval = trainer: identical actions over {} steps".format(steps))


if __name__ == "__main__":
    for t in [test_noise_draws, test_zero_noise_is_pengze_env, test_eval_matches_trainer]:
        t()
    print("all ball-noise tests passed")
