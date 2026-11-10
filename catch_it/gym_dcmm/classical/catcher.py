"""
Author: Zwe Min Htet Aung
ABOUT: classical (non-learning) catcher for the HoopCatch env: predict the ball's parabola, pick a
reachable catch point on it, and drive the hoop there.

It reads the same observation dict as the RL policy and returns the same 5-D action in [-1, 1]
(hoop-center delta and tilt), so the env's speed limits, IK and PID apply to it exactly as to RL.
It never reads the env's throw target, arrival velocity or gravity (except mode "perfect_state",
a diagnostic that is handed the true release state by the evaluator).

Modes:
- "single_read":   one parabola from the first clean read after release, never updated (open loop on the ball).
- "multi_read":    re-fit the parabola to every read since release, every control step (closed loop).
- "perfect_state": like single_read, but the parabola comes from the true release state and gravity.

Prediction: Eqs. (1)-(3) of Campana & Laumond (2016), see ballistic.py.
Settings: configs/env/ClassicalCfg.py.
"""
import math
import numpy as np
import mujoco
import configs.env.DcmmCfg as DcmmCfg
import configs.env.ClassicalCfg as CCfg
from gym_dcmm.classical.ballistic import Parabola, fit_parabola

MODES = ["single_read", "multi_read", "perfect_state"]


def _angle_between(a, b):
    return math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))


class ClassicalCatcher:
    """
    Args:
    - mode: one of MODES
    - model_arm: the arm-only MuJoCo model (env.Dcmm.model_arm), used for forward kinematics of the
      observed joint angles (the hoop's full orientation is needed to express tilts in its own frame)
    - dt: control period of the env (s)
    """
    def __init__(self, mode, model_arm, dt):
        if mode not in MODES:
            raise ValueError("Invalid mode: {}".format(mode))
        self.mode = mode
        self.dt = dt
        self.model_arm = model_arm
        self.data_arm = mujoco.MjData(model_arm)
        self.reset()

    # ------------------------------------------------------------------ episode state
    def reset(self):
        self.k = 0                  # index of the current observation (time = k * dt)
        self.rest_sum = np.zeros(3)
        self.rest_n = 0
        self.released = False
        self.t_pos, self.p_reads = [], []
        self.t_vel, self.v_reads = [], []
        self.parabola = None
        self.plan = None            # dict(t, p, n, slack, feasible)
        self.ready = None           # (p, n) of the ready pose
        # Diagnostics (read by the evaluator)
        self.log = {
            "release_detect_time": np.nan, "decision_time": np.nan, "n_reads": 0, "n_reselect": 0,
            "t_star": np.nan, "p_star": np.full(3, np.nan), "n_star": np.full(3, np.nan),
            "slack": np.nan, "feasible": False, "in_region": False, "g_hat": np.nan,
        }

    # ------------------------------------------------------------------ main entry
    def act(self, obs, privileged=None):
        """
        Map one observation to an action.
        privileged: only for mode "perfect_state", dict(p0, v0, g, t_throw) in the arm-base frame.
        """
        t = self.k * self.dt
        self.k += 1
        p_obs = np.asarray(obs["object"]["pos3d"], dtype=np.float64)
        v_obs = np.asarray(obs["object"]["v_lin_3d"], dtype=np.float64)
        hoop_pos = np.asarray(obs["hoop"]["pos3d"], dtype=np.float64)
        R_hoop = self._hoop_rotation(obs["arm"]["joint_pos"])

        if not self.released:
            rest = self.rest_sum / max(self.rest_n, 1)
            if self.rest_n > 0 and np.linalg.norm(p_obs - rest) > CCfg.release_disp \
                    and np.linalg.norm(v_obs) > CCfg.release_speed:
                # First read after release: the position is valid, the velocity is not
                # (it is a finite difference across the hold)
                self.released = True
                self.log["release_detect_time"] = t
                self.t_pos.append(t); self.p_reads.append(p_obs)
            else:
                self.rest_sum += p_obs
                self.rest_n += 1
                if CCfg.ready_pose:
                    self.ready = self._ready_pose(self.rest_sum / self.rest_n)
        else:
            self.t_pos.append(t); self.p_reads.append(p_obs)
            self.t_vel.append(t); self.v_reads.append(v_obs)
            self._update_plan(t, hoop_pos, R_hoop[:, 2], privileged)

        if self.plan is not None:
            return self._servo(hoop_pos, R_hoop, self.plan["p"], self.plan["n"])
        if self.ready is not None:
            return self._servo(hoop_pos, R_hoop, *self.ready)
        return np.zeros(5)

    # ------------------------------------------------------------------ prediction + planning
    def _update_plan(self, t, hoop_pos, hoop_axis, privileged):
        if self.mode in ["single_read", "perfect_state"] and self.parabola is not None:
            return  # open loop on the ball: the first plan is final
        if self.mode == "perfect_state":
            self.parabola = Parabola(privileged["p0"], privileged["v0"], privileged["t_throw"], privileged["g"])
        elif self.mode == "single_read":
            self.parabola = fit_parabola(self.t_pos[-1:], self.p_reads[-1:], self.t_vel[-1:], self.v_reads[-1:],
                                         t_ref=t, vel_window=self.dt, g=CCfg.g_nominal, fit_g=False,
                                         pos_sigma=CCfg.pos_sigma, vel_sigma=CCfg.vel_sigma)
        else:
            self.parabola = fit_parabola(self.t_pos, self.p_reads, self.t_vel, self.v_reads,
                                         t_ref=t, vel_window=self.dt, g=CCfg.g_nominal, fit_g=CCfg.fit_gravity,
                                         g_prior_sigma=CCfg.g_prior_sigma,
                                         pos_sigma=CCfg.pos_sigma, vel_sigma=CCfg.vel_sigma)
        if np.isnan(self.log["decision_time"]):
            self.log["decision_time"] = t
        self.log["n_reads"] = len(self.t_pos)
        self.log["g_hat"] = self.parabola.g

        keep_t = self.plan["t"] if self.plan is not None else None
        plan = self._choose_catch_point(self.parabola, t, hoop_pos, hoop_axis, keep_t)
        if self.plan is not None and plan["t"] != self.plan["t"]:
            self.log["n_reselect"] += 1
        self.plan = plan
        self.log.update({"t_star": plan["t"], "p_star": plan["p"].copy(), "n_star": plan["n"].copy(),
                         "slack": plan["slack"], "feasible": plan["feasible"], "in_region": plan["in_region"]})

    def _choose_catch_point(self, par, t_now, hoop_pos, hoop_axis, keep_t=None):
        """
        Sample the predicted path, keep the points the hoop can catch at (inside the reach region and
        the hoop target box), and pick the one needing the least hoop motion among those it can reach
        in time. slack = (time until the ball is there) - (time the hoop needs to get there).
        """
        def evaluate(ts):
            P = par.position(ts)
            N = -par.direction(ts)                   # hoop axis faces the incoming ball (Eq. 3)
            in_region = self._in_region(P)
            dist = np.linalg.norm(P - hoop_pos, axis=1)
            tilt = np.arctan2(np.linalg.norm(np.cross(hoop_axis, N), axis=1), N @ hoop_axis)
            t_need = np.maximum(dist / CCfg.hoop_speed, tilt / CCfg.hoop_tilt_rate) + CCfg.settle_time
            return P, N, in_region, dist, tilt, (ts - t_now) - t_need

        def pack(ts, P, N, in_region, slack, i):
            return {"t": float(ts[i]), "p": P[i], "n": N[i], "slack": float(slack[i]),
                    "feasible": bool(in_region[i] and slack[i] >= 0), "in_region": bool(in_region[i])}

        # Multi-read: keep the current catch time unless it became unusable (avoids target jumps)
        if keep_t is not None and self.mode == "multi_read":
            if keep_t <= t_now:
                return self.plan                     # the ball is due now: hold still
            ts = np.array([keep_t])
            P, N, in_region, _, _, slack = evaluate(ts)
            if keep_t - t_now < CCfg.commit_time or (in_region[0] and slack[0] >= CCfg.replan_slack):
                refined = pack(ts, P, N, in_region, slack, 0)
                # Slack shrinks as the deadline nears; report the values from when this catch time was chosen
                refined["slack"], refined["feasible"] = self.plan["slack"], self.plan["feasible"]
                return refined

        ts = t_now + np.arange(CCfg.min_lead, CCfg.horizon, CCfg.candidate_dt)
        P, N, in_region, dist, tilt, slack = evaluate(ts)
        pack_i = lambda i: pack(ts, P, N, in_region, slack, i)
        feasible = in_region & (slack >= 0)
        if feasible.any():
            cost = dist + CCfg.tilt_cost * tilt
            i = int(np.argmin(np.where(feasible, cost, np.inf)))
        elif in_region.any():
            i = int(np.argmax(np.where(in_region, slack, -np.inf)))   # best effort: the most slack
        else:
            i = int(np.argmin(dist))                                    # path misses the region: closest point
        return pack_i(i)

    @staticmethod
    def _in_region(P):
        reg = CCfg.reach_region
        r = np.hypot(P[:, 0], P[:, 1])
        az = np.arctan2(P[:, 0], P[:, 1])            # azimuth from +y (the robot's front)
        ok = (r >= reg["r"][0]) & (r <= reg["r"][1]) & (az >= reg["azimuth"][0]) & (az <= reg["azimuth"][1]) \
            & (P[:, 2] >= reg["z"][0]) & (P[:, 2] <= reg["z"][1])
        ok &= np.all((P >= DcmmCfg.hoop_target_low) & (P <= DcmmCfg.hoop_target_high), axis=1)
        return ok

    @staticmethod
    def _ready_pose(rest):
        """Hoop straight ahead in the middle of the reach region, axis pointing back up toward the launch point."""
        p = np.array([0.0, CCfg.ready_r, CCfg.ready_z])
        h = rest[0:2] - p[0:2]
        h = h / max(np.linalg.norm(h), 1e-6)
        e = CCfg.ready_elevation
        n = np.array([math.cos(e) * h[0], math.cos(e) * h[1], math.sin(e)])
        return p, n

    # ------------------------------------------------------------------ control
    def _hoop_rotation(self, joint_pos):
        """Forward kinematics of the observed joint angles: link6 rotation (z column = hoop axis)."""
        self.data_arm.qpos[0:6] = joint_pos
        mujoco.mj_kinematics(self.model_arm, self.data_arm)
        return self.data_arm.body("link6").xmat.reshape(3, 3).copy()

    @staticmethod
    def _servo(hoop_pos, R_hoop, p_target, n_target):
        """
        Same servo law as the env's oracle: move the hoop center toward p_target and tilt the hoop
        axis toward n_target, expressed in the 5-D normalized action (the env clips and rate-limits it).
        """
        action = np.zeros(5)
        action[0:3] = (p_target - hoop_pos) / DcmmCfg.action_scale["pos"]
        axis = R_hoop[:, 2]
        rot_vec = np.cross(axis, n_target)
        sin_angle = np.linalg.norm(rot_vec)
        if sin_angle > 1e-6:
            angle = math.atan2(sin_angle, np.dot(axis, n_target))
            rot_local = R_hoop.T @ (rot_vec / sin_angle * angle)
            action[3:5] = rot_local[0:2] / DcmmCfg.action_scale["tilt"]
        return np.clip(action, -1, 1)
