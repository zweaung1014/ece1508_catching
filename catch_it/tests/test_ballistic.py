"""
Author: Zwe Min Htet Aung
ABOUT: checks of the ballistic predictor (gym_dcmm/classical/ballistic.py) and the per-throw disturbances.
1. Eqs. (2)-(3) of Campana & Laumond describe the same path as Eq. (1).
2. The least-squares fit recovers an exact state (single read, multi read, fitted g).
3. DragPath equals the parabola at k = 0 and matches a fine numerical integration for k > 0.
4. DragPath matches the ball's free flight in the MuJoCo env (random drag, RK4) to < 1 mm, and at the
   strongest drag (b = 0.005, 35 g ball) the ball loses ~13% of its speed per second.
5. The per-throw disturbances are the same for the same seed and stay inside their ranges.

Run from catch_it/:  python tests/test_ballistic.py     (pytest also works if installed)
"""
import os, sys
CATCH_IT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [CATCH_IT, os.path.join(CATCH_IT, "gym_dcmm")]
import numpy as np
import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.classical.ballistic import Parabola, DragPath, fit_parabola


def make_env():
    os.chdir(CATCH_IT)
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    return DcmmVecEnv(task="HoopCatch", viewer=False, imshow_cam=False, env_time=2.5, steps_per_policy=20)


def fly(env, seed):
    """Release the ball (hoop held still) and record its true path until it touches anything.
    Returns the true DragPath and a list of (time, position, velocity) in flight."""
    env.reset(seed=seed)
    dt = 1.0 / env.fps
    k, done, path, samples = 0, False, None, []
    while not done:
        _, _, term, trunc, _ = env.step(np.zeros(5))
        k += 1
        done = term or trunc
        if env.throw_time is None:
            continue
        if path is None:
            path = DragPath(env.object_pos3d - env.arm_base_pos, env.object_vel6d[:3], env.throw_time,
                            -env.Dcmm.model.opt.gravity[2], env.object_damping / env.object_mass)
        if env.ball_touched or done:
            break
        samples.append((k * dt, env._get_object_pos3d(), env._get_object_v_lin_3d()))
    return path, samples


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


def test_drag_path():
    c0, v0, t_s, g = np.array([0.3, 2.0, 0.9]), np.array([-0.5, -3.5, 1.2]), 0.62, 9.81
    t = t_s + np.linspace(0.0, 1.2, 61)
    # k = 0 is the parabola
    par, drag0 = Parabola(c0, v0, t_s, g), DragPath(c0, v0, t_s, g, 0.0)
    assert np.allclose(drag0.position(t), par.position(t), atol=1e-12)
    assert np.allclose(drag0.direction(t), par.direction(t), atol=1e-9)
    # Tiny k is continuous with k = 0
    assert np.allclose(DragPath(c0, v0, t_s, g, 1e-6).position(t), par.position(t), atol=1e-5)
    # k > 0 matches a fine RK4 integration of dv/dt = -g e_z - k v
    k, h = 0.143, 1e-4
    path = DragPath(c0, v0, t_s, g, k)
    acc = lambda v: np.array([0.0, 0.0, -g]) - k * v
    p, v = c0.copy(), v0.copy()
    for _ in range(int(round(1.2 / h))):
        k1v, k1p = acc(v), v
        k2v, k2p = acc(v + 0.5 * h * k1v), v + 0.5 * h * k1v
        k3v, k3p = acc(v + 0.5 * h * k2v), v + 0.5 * h * k2v
        k4v, k4p = acc(v + h * k3v), v + h * k3v
        p = p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    assert np.allclose(path.position(t_s + 1.2), p, atol=1e-9)
    assert np.allclose(path.velocity(t_s + 1.2), v, atol=1e-9)


def test_matches_mujoco_flight(seeds=(0, 1, 2, 3, 4, 5, 6, 7)):
    env = make_env()
    worst = 0.0
    for seed in seeds:
        path, samples = fly(env, seed)
        for t, p, _ in samples:
            worst = max(worst, np.linalg.norm(p - path.position(t)))
    env.close()
    print("worst free-flight prediction error (random drag): {:.2e} m".format(worst))
    assert worst < 1e-3


def test_strongest_drag():
    saved = DcmmCfg.object_damping, DcmmCfg.object_mass
    DcmmCfg.object_damping, DcmmCfg.object_mass = np.array([0.005, 0.005]), np.array([0.035, 0.035])
    try:
        env = make_env()
        worst, k_meas = 0.0, []
        for seed in (0, 1, 2):
            path, samples = fly(env, seed)
            for t, p, _ in samples:
                worst = max(worst, np.linalg.norm(p - path.position(t)))
            # Horizontal speed decays as exp(-k t) (gravity is vertical)
            (t0, _, v0), (t1, _, v1) = samples[0], samples[-1]
            k_meas.append(np.log(np.linalg.norm(v0[:2]) / np.linalg.norm(v1[:2])) / (t1 - t0))
        env.close()
    finally:
        DcmmCfg.object_damping, DcmmCfg.object_mass = saved
    k = float(np.mean(k_meas))
    print("strongest drag: path error {:.2e} m, k = {:.4f} 1/s (expected {:.4f}), speed lost over 1 s = {:.1%}".format(
        worst, k, 0.005 / 0.035, 1 - np.exp(-k)))
    assert worst < 1e-3
    assert abs(k - 0.005 / 0.035) < 0.01 * 0.005 / 0.035
    assert 0.12 < 1 - np.exp(-k) < 0.14


def test_per_throw_disturbances():
    env = make_env()
    draws = []
    for seed in range(300):
        env.reset(seed=seed)
        draws.append((env.k_obs_object_vel, env.object_damping, env.act_delay_steps, env.Dcmm.model.opt.gravity[2]))
    draws = np.array(draws)
    assert np.all((draws[:, 0] >= 0.05) & (draws[:, 0] <= 0.18))
    assert np.all((draws[:, 1] >= 0.0) & (draws[:, 1] <= 0.005))
    assert set(draws[:, 2].astype(int)) == {0, 1, 2}
    assert np.allclose(draws[:, 3], -9.81)
    # Same seed, same throw and disturbances
    env.reset(seed=7)
    a = (env.k_obs_object_vel, env.object_damping, env.act_delay_steps, env.throw_target.copy())
    env.reset(seed=7)
    b = (env.k_obs_object_vel, env.object_damping, env.act_delay_steps, env.throw_target.copy())
    assert a[:3] == b[:3] and np.allclose(a[3], b[3])
    # Same seed, same throw geometry and ball as exp01 (the new draws come after them)
    ref = os.path.join(CATCH_IT, "results", "exp01_one_factor", "nominal", "oracle.csv")
    if os.path.exists(ref):
        import pandas as pd
        old = pd.read_csv(ref).set_index("seed")
        for seed in range(10000, 10020):
            env.reset(seed=seed)
            t = env.throw_target
            assert np.isclose(np.hypot(t[0], t[1]), old.loc[seed, "target_r"])
            assert np.isclose(t[2], old.loc[seed, "target_z"])
            assert np.isclose(env.throw_flight_time, old.loc[seed, "flight_time"])
            assert np.isclose(env.object_mass, old.loc[seed, "ball_mass"])
    env.close()


if __name__ == "__main__":
    test_plane_equations_match_eq1()
    test_fit_recovers_exact_state()
    test_drag_path()
    test_matches_mujoco_flight()
    test_strongest_drag()
    test_per_throw_disturbances()
    print("all ballistic tests passed")
