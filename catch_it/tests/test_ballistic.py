"""
Author: Zwe Min Htet Aung
ABOUT: checks of the ballistic predictor (gym_dcmm/classical/ballistic.py).
1. Eqs. (2)-(3) of Campana & Laumond describe the same path as Eq. (1).
2. The least-squares fit recovers an exact state (single read, multi read, fitted g).
3. The prediction matches the ball's free flight in the MuJoCo env (no drag, RK4) to < 1 mm.

Run from catch_it/:  python tests/test_ballistic.py     (pytest also works if installed)
"""
import os, sys
CATCH_IT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [CATCH_IT, os.path.join(CATCH_IT, "gym_dcmm")]
import numpy as np
from gym_dcmm.classical.ballistic import Parabola, fit_parabola


def test_plane_equations_match_eq1():
    rng = np.random.default_rng(0)
    for _ in range(50):
        par = Parabola(rng.uniform(-2, 2, 3), rng.uniform(-5, 5, 3), rng.uniform(0, 1), rng.uniform(9.3, 10.3))
        t = par.t_s + np.linspace(0.0, 1.2, 61)
        c = par.position(t)
        x = par.x_theta(c)
        assert np.allclose(par.z_of_x(x), c[:, 2], atol=1e-9)                       # Eq. (2)
        assert np.allclose(par.time_of_x(x), t, atol=1e-9)
        v = par.velocity(t)
        assert np.allclose(par.direction(t), v / np.linalg.norm(v, axis=1, keepdims=True), atol=1e-9)  # Eq. (3)
        assert np.allclose(par.direction(t[5]), v[5] / np.linalg.norm(v[5]), atol=1e-9)


def test_fit_recovers_exact_state():
    dt = 0.04
    true = Parabola([0.3, 2.0, 0.9], [-0.5, -3.5, 1.2], t_s=0.62, g=9.55)
    t = 0.62 + dt * np.arange(1, 9)
    pos = true.position(t)
    vel = true.velocity(t - dt / 2)          # finite difference over the last dt = velocity at the midpoint
    # Single read (one position + one velocity), g known: exactly determined
    est = fit_parabola(t[:1], pos[:1], t[:1], vel[:1], t_ref=t[0], vel_window=dt, g=9.55)
    assert np.allclose(est.position(t), pos, atol=1e-9)
    # Many reads, g known
    est = fit_parabola(t, pos, t[1:], vel[1:], t_ref=t[-1], vel_window=dt, g=9.55)
    assert np.allclose(est.position(t), pos, atol=1e-9)
    # Many reads, g fitted from a wrong prior (weak prior so the data decides)
    est = fit_parabola(t, pos, t[1:], vel[1:], t_ref=t[-1], vel_window=dt, g=9.81, fit_g=True, g_prior_sigma=1e3)
    assert abs(est.g - 9.55) < 1e-6
    assert np.allclose(est.position(t), pos, atol=1e-7)


def test_matches_mujoco_flight(seeds=(0, 1, 2, 3, 4)):
    os.chdir(CATCH_IT)
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    env = DcmmVecEnv(task="HoopCatch", viewer=False, imshow_cam=False, env_time=2.5, steps_per_policy=20)
    worst = 0.0
    for seed in seeds:
        env.reset(seed=seed)
        dt = 1.0 / env.fps
        k, done, par = 0, False, None
        while not done:
            _, _, term, trunc, _ = env.step(np.zeros(5))
            k += 1
            done = term or trunc
            if env.throw_time is None:
                continue
            if par is None:
                launch = env.object_pos3d - env.arm_base_pos
                par = Parabola(launch, env.object_vel6d[:3], env.throw_time, -env.Dcmm.model.opt.gravity[2])
            if env.ball_touched or done:
                break
            err = np.linalg.norm(env._get_object_pos3d() - par.position(k * dt))
            worst = max(worst, err)
    env.close()
    print("worst free-flight prediction error: {:.2e} m".format(worst))
    assert worst < 1e-3


if __name__ == "__main__":
    test_plane_equations_match_eq1()
    test_fit_recovers_exact_state()
    test_matches_mujoco_flight()
    print("all ballistic tests passed")
