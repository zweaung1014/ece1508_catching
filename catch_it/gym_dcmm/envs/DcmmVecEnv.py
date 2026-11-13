"""
Author: Zwe Min Htet Aung
Version@2026-10-30
ABOUT: this file contains the RL environment for catching a thrown ball with a hoop on a fixed-base xArm6 (task "HoopCatch").
The full design (states, actions, rewards, throws) is documented in docs/hoop_rl_design.md.
Adapted from the Catch It codebase (MIT License, see LICENSE).
"""
import os, sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./gym_dcmm/'))
import argparse
import math
import time
import configs.env.DcmmCfg as DcmmCfg
import cv2 as cv
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from gym_dcmm.agents.MujocoDcmm import MJ_DCMM
import copy
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from utils.util import *
from scipy.spatial.transform import Rotation as R

# os.environ['MUJOCO_GL'] = 'egl'
np.set_printoptions(precision=8)

# Episode outcomes (info["outcome"])
OUTCOME_NONE = 0        # still running
OUTCOME_CATCH = 1       # ball passed through the ring
OUTCOME_BALL_FLOOR = 2  # ball hit the floor
OUTCOME_BALL_ROBOT = 3  # ball hit the arm links or the base
OUTCOME_COLLISION = 4   # arm/hoop hit the base or the floor
OUTCOME_TIMEOUT = 5     # env_time exceeded
OUTCOME_NAMES = ["none", "catch", "ball_floor", "ball_robot", "collision", "timeout"]

# Keyboard control (one action pulse per key press)
key_action = np.zeros(5)

def env_key_callback(keycode):
  global key_action
  if keycode == 265: # AKA: up
    key_action[1] = 1.0
  if keycode == 264: # AKA: down
    key_action[1] = -1.0
  if keycode == 263: # AKA: left
    key_action[0] = -1.0
  if keycode == 262: # AKA: right
    key_action[0] = 1.0
  if keycode == 334: # AKA + (on the numpad)
    key_action[2] = 1.0
  if keycode == 333: # AKA - (on the numpad)
    key_action[2] = -1.0
  if keycode == 327: # AKA 7 (on the numpad)
    key_action[3] = 1.0
  if keycode == 329: # AKA 9 (on the numpad)
    key_action[3] = -1.0
  if keycode == 52: # AKA: 4
    key_action[4] = 1.0
  if keycode == 54: # AKA: 6
    key_action[4] = -1.0

class DcmmVecEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "depth_array"]}
    """
    Args:
        render_mode: str
            The mode of rendering, including "rgb_array", "depth_array".
        render_per_step: bool
            Whether to render the mujoco model per simulation step.
        viewer: bool
            Whether to show the mujoco viewer.
        imshow_cam: bool
            Whether to show the camera image.
        camera_name: str
            The name of the camera.
        object_name: str
            The name of the object.
        env_time: float
            The maximum time of the environment.
        steps_per_policy: int
            The number of steps per action.
        img_size: tuple
            The size of the image.
    """
    def __init__(
        self,
        task="HoopCatch",
        render_mode="rgb_array",
        render_per_step=False,
        viewer=False,
        imshow_cam=False,
        camera_name=["top"],
        object_name="object",
        env_time=2.5,
        steps_per_policy=20,
        img_size=(480, 640),
        device='cuda:0',
        print_obs=False,
        print_reward=False,
        print_ctrl=False,
        print_info=False,
        print_contacts=False,
    ):
        if task not in ["HoopCatch"]:
            raise ValueError("Invalid task: {}".format(task))
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.object_name = object_name
        self.imshow_cam = imshow_cam
        self.task = task
        self.img_size = img_size
        self.device = device
        self.steps_per_policy = steps_per_policy
        self.render_per_step = render_per_step
        # Print Settings
        self.print_obs = print_obs
        self.print_reward = print_reward
        self.print_ctrl = print_ctrl
        self.print_info = print_info
        self.print_contacts = print_contacts
        # Initialize the environment
        self.Dcmm = MJ_DCMM(viewer=viewer, object_name=object_name)
        # self.Dcmm.show_model_info()
        self.fps = 1 / (self.steps_per_policy * self.Dcmm.model.opt.timestep)
        self.last_sync_time = None
        self.arm_base_pos = self.Dcmm.data.body("arm_base").xpos.copy()

        # Get the geom/body ids used to classify the contacts
        model = self.Dcmm.model
        self.floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)
        self.object_body_id = model.body(self.object_name).id
        self.rim_body_id = model.body("hoop_rim").id
        self.hoop_body_ids = np.array([model.body(name).id for name in ["hoop", "hoop_net", "hoop_rim"]])
        self.arm_body_ids = np.array([model.body("link{}".format(i)).id for i in range(1, 7)])
        # The base is welded to the world, so all of its bodies have weldid 0
        self.base_body_ids = np.array([i for i in range(1, model.nbody) if model.body_weldid[i] == 0])

        # Ball properties (randomized in random_object)
        self.object_radius = model.geom_size[self.object_id][0]
        self.object_mass = model.body_mass[self.object_body_id]

        # Set the camera configuration
        self.Dcmm.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]
        self.Dcmm.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        self.mujoco_renderer = MujocoRenderer(
            self.Dcmm.model, self.Dcmm.data
        )
        if self.Dcmm.open_viewer:
            if self.Dcmm.viewer:
                print("Close the previous viewer")
                self.Dcmm.viewer.close()
            self.Dcmm.viewer = mujoco.viewer.launch_passive(self.Dcmm.model, self.Dcmm.data, key_callback=env_key_callback)
            # Modify the view position and orientation
            self.Dcmm.viewer.cam.lookat[0:3] = [0, 1, 0.8]
            self.Dcmm.viewer.cam.distance = 4.0
            self.Dcmm.viewer.cam.azimuth = 150
            self.Dcmm.viewer.cam.elevation = -20
        else: self.Dcmm.viewer = None

        # Observations are dictionaries with the hoop's, the arm's and the ball's state. (dim = 35)
        # All positions are relative to the arm base (the base is fixed, so this frame never moves).
        self.observation_space = spaces.Dict(
            {
                "hoop": spaces.Dict({
                    "pos3d": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    "axis3d": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    "v_lin_3d": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                }),
                "arm": spaces.Dict({
                    "joint_pos": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                    "joint_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                }),
                "object": spaces.Dict({
                    "pos3d": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    "v_lin_3d": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    "rel_pos3d": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                }),
                "prev_action": spaces.Box(-1, 1, shape=(5,), dtype=np.float32),
            }
        )
        # Actions (dim = 5), normalized to [-1, 1]:
        # [0:3] delta hoop center, [3:5] tilt of the hoop axis (see DcmmCfg.action_scale)
        self.action_space = spaces.Box(-1, 1, shape=(5,), dtype=np.float32)
        self.action_buffer = {
            "arm": DynamicDelayBuffer(maxlen=2),
        }

        self.obs_dim = get_total_dimension(self.observation_space)
        self.act_dim = self.action_space.shape[0]
        print("##### HoopCatch Task \n obs_dim: {}, act_dim: {}\n".format(self.obs_dim, self.act_dim))

        # Time grid of the ballistic path prediction used by the reward
        self.path_tau = np.arange(0.0, DcmmCfg.path_horizon, DcmmCfg.path_dt)

        # Init env params
        self.arm_limit = True
        self.terminated = False
        self.start_time = self.Dcmm.data.time
        self.env_time = env_time
        self.steps = 0
        self.object_throw = False
        self.object_static_time = 0.75
        self.object_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.object_pos3d = np.array([0.0, 2.5, 1.2])
        self.object_vel6d = np.zeros(6)
        self.throw_target = np.zeros(3)
        self.throw_arrival_vel = np.zeros(3)
        self.prev_action = np.zeros(self.act_dim)
        self.init_pos = True
        self.prev_hoop_pos3d = np.zeros(3)
        self.prev_obj_pos3d = np.zeros(3)
        self._reset_episode_state()

        # Random PID Params
        self.k_arm = np.ones(6)
        # Per-throw disturbances (drawn in _reset_simulation)
        self.object_damping = 0.0
        self.act_delay_steps = 0
        self.k_obs_object = float(DcmmCfg.k_obs_object[0])
        self.k_obs_object_vel = float(DcmmCfg.k_obs_object_vel[0])
        self.wind_force = np.zeros(3)   # N, world frame, applied to the ball after release

    def _reset_episode_state(self):
        self.outcome = OUTCOME_NONE
        self.step_catch = False
        self.step_rim = False
        self.ball_touched = False
        self.prev_d_path = None
        self.prev_ring_side = 0.0
        # Env time (s since reset) at which the ball was released; logged for evaluation only
        self.throw_time = None
        # Episode statistics (logged at the end of every episode)
        self.min_ball_dist = 10.0
        self.min_path_dist = 10.0
        self.rim_hits = 0
        self.ik_fails = 0

    def update_render_state(self, render_per_step):
        self.render_per_step = render_per_step

    def _get_hoop_pos3d(self):
        # Hoop center w.r.t. the arm base
        return self.Dcmm.data.site("hoop_center").xpos - self.arm_base_pos

    def _get_hoop_axis(self):
        # Unit vector the hoop opening faces (z axis of the hoop / link6)
        return self.Dcmm.data.site("hoop_center").xmat.reshape(3, 3)[:, 2].copy()

    def _get_object_pos3d(self):
        # Ball center w.r.t. the arm base
        return self.Dcmm.data.body(self.Dcmm.object_name).xpos - self.arm_base_pos

    def _get_object_v_lin_3d(self):
        # True linear velocity of the ball (world frame)
        adr = self.Dcmm.object_qvel_adr
        return self.Dcmm.data.qvel[adr:adr+3].copy()

    def _get_obs(self):
        hoop_pos3d = self._get_hoop_pos3d()
        obj_pos3d = self._get_object_pos3d()
        if self.init_pos:
            self.prev_hoop_pos3d[:] = hoop_pos3d[:]
            self.prev_obj_pos3d[:] = obj_pos3d[:]
            self.init_pos = False
        # Add Obs Noise (Additive DcmmCfg.k_obs_*)
        hoop_pos_obs = hoop_pos3d + np.random.normal(0, DcmmCfg.k_obs_hoop, 3)
        obj_pos_obs = obj_pos3d + np.random.normal(0, self.k_obs_object, 3)
        obs = {
            "hoop": {
                "pos3d": hoop_pos_obs,
                "axis3d": self._get_hoop_axis() + np.random.normal(0, DcmmCfg.k_obs_hoop, 3),
                "v_lin_3d": (hoop_pos3d - self.prev_hoop_pos3d)*self.fps + np.random.normal(0, DcmmCfg.k_obs_hoop, 3),
            },
            "arm": {
                "joint_pos": self.Dcmm.data.qpos[self.Dcmm.arm_qpos_adr] + np.random.normal(0, DcmmCfg.k_obs_arm, 6),
                "joint_vel": self.Dcmm.data.qvel[self.Dcmm.arm_qvel_adr] + np.random.normal(0, DcmmCfg.k_obs_arm, 6),
            },
            "object": {
                "pos3d": obj_pos_obs,
                "v_lin_3d": (obj_pos3d - self.prev_obj_pos3d)*self.fps + np.random.normal(0, self.k_obs_object_vel, 3),
                "rel_pos3d": obj_pos_obs - hoop_pos_obs,
            },
            "prev_action": self.prev_action.copy(),
        }
        self.prev_hoop_pos3d = hoop_pos3d
        self.prev_obj_pos3d = obj_pos3d
        if self.print_obs:
            print("##### print obs: \n", obs)
        return obs

    def _get_info(self):
        # Time of the Mujoco environment
        env_time = self.Dcmm.data.time - self.start_time
        ball_dist = np.linalg.norm(self._get_object_pos3d() - self._get_hoop_pos3d())
        self.min_ball_dist = min(self.min_ball_dist, ball_dist)
        if self.print_info:
            print("##### print info")
            print("env_time: ", env_time)
            print("ball_dist: ", ball_dist)
        return {
            "env_time": env_time,
            "success": self.outcome == OUTCOME_CATCH,
            "outcome": self.outcome,
            "ball_dist": ball_dist,
            "min_ball_dist": self.min_ball_dist,
            "min_path_dist": self.min_path_dist,
            "rim_hits": self.rim_hits,
            "ik_fails": self.ik_fails,
        }

    def update_target_ctrl(self):
        self.action_buffer["arm"].append(copy.deepcopy(self.Dcmm.target_arm_qpos[:]))

    def _get_ctrl(self):
        # Map the target joint positions to the arm torques
        mv_arm = self.Dcmm.arm_pid.update(self.action_buffer["arm"][0], self.Dcmm.data.qpos[self.Dcmm.arm_qpos_adr], self.Dcmm.data.time)
        # Add Action Noise (Scale with DcmmCfg.k_act)
        ctrl = mv_arm * np.random.normal(1, DcmmCfg.k_act, 6)
        if self.print_ctrl:
            print("##### ctrl: ", ctrl)
        return ctrl

    def random_object(self):
        # Random ball size, mass and damping (changed in the compiled model, no xml reload needed)
        model = self.Dcmm.model
        self.object_radius = np.random.uniform(DcmmCfg.object_radius[0], DcmmCfg.object_radius[1])
        self.object_mass = np.random.uniform(DcmmCfg.object_mass[0], DcmmCfg.object_mass[1])
        model.geom_size[self.object_id][0] = self.object_radius
        model.geom_rbound[self.object_id] = self.object_radius
        model.geom_aabb[self.object_id][3:6] = self.object_radius
        model.body_mass[self.object_body_id] = self.object_mass
        # Thin hollow sphere (like a tennis ball)
        model.body_inertia[self.object_body_id] = 2.0 / 3.0 * self.object_mass * self.object_radius**2
        adr = self.Dcmm.object_qvel_adr
        # Damping b on the free joint = linear air drag -b*v (logged for evaluation)
        self.object_damping = np.random.uniform(DcmmCfg.object_damping[0], DcmmCfg.object_damping[1])
        model.dof_damping[adr:adr+6] = self.object_damping

    def random_throw(self):
        # Target point inside the catch zone (w.r.t. the arm base)
        zone = DcmmCfg.catch_zone
        azimuth = np.random.uniform(zone["azimuth"][0], zone["azimuth"][1])
        r = np.random.uniform(zone["r"][0], zone["r"][1])
        z = np.random.uniform(zone["z"][0], zone["z"][1])
        self.throw_target = np.array([r * math.sin(azimuth), r * math.cos(azimuth), z])
        target = self.arm_base_pos + self.throw_target
        # Launch point (world frame), azimuth measured from +y (the front of the robot)
        launch = DcmmCfg.launch_zone
        azimuth = np.random.uniform(launch["azimuth"][0], launch["azimuth"][1])
        r = np.random.uniform(launch["r"][0], launch["r"][1])
        height = np.random.uniform(launch["z"][0], launch["z"][1])
        launch_pos = np.array([self.arm_base_pos[0] + r * math.sin(azimuth),
                               self.arm_base_pos[1] + r * math.cos(azimuth),
                               height])
        # Ballistic velocity that hits the target after the flight time T: p_T = p_0 + v_0 T + 0.5 g T^2
        T = np.random.uniform(DcmmCfg.flight_time[0], DcmmCfg.flight_time[1])
        self.throw_flight_time = T  # logged for evaluation only
        g = self.Dcmm.model.opt.gravity.copy()
        v0 = (target - launch_pos) / T - 0.5 * g * T
        self.throw_arrival_vel = v0 + g * T
        self.object_pos3d = launch_pos
        self.object_vel6d = np.concatenate((v0, np.zeros(3)))
        # Random Static Time
        self.object_static_time = np.random.uniform(DcmmCfg.object_static[0], DcmmCfg.object_static[1])

    def random_PID(self):
        # Random the PID Controller Params of the arm
        self.k_arm = np.random.uniform(0, 1, size=6)
        # Reset the PID Controller
        self.Dcmm.arm_pid.reset(self.k_arm*(DcmmCfg.k_arm[1]-DcmmCfg.k_arm[0])+DcmmCfg.k_arm[0])

    def random_delay(self):
        # Random the Delay Buffer Params
        self.action_buffer["arm"].set_maxlen(np.random.choice(DcmmCfg.act_delay['arm']))
        # Buffer length 1 = no delay; the arm follows the command from (length - 1) steps ago
        self.act_delay_steps = int(self.action_buffer["arm"].maxlen) - 1
        # Clear Buffer
        self.action_buffer["arm"].clear()

    def random_obs_noise(self):
        # Std of the ball velocity noise, then of the ball position noise, for this throw
        self.k_obs_object_vel = np.random.uniform(DcmmCfg.k_obs_object_vel[0], DcmmCfg.k_obs_object_vel[1])
        self.k_obs_object = np.random.uniform(DcmmCfg.k_obs_object[0], DcmmCfg.k_obs_object[1])

    def random_wind(self):
        # Constant horizontal force on the ball after release (applied in step), uniform direction
        magnitude = np.random.uniform(DcmmCfg.wind_force[0], DcmmCfg.wind_force[1])
        heading = np.random.uniform(-np.pi, np.pi)
        self.wind_force = np.array([magnitude * math.cos(heading), magnitude * math.sin(heading), 0.0])

    def ball_acceleration(self):
        # Acceleration of the ball in flight apart from drag: gravity plus wind (world frame)
        return self.Dcmm.model.opt.gravity.copy() + self.wind_force / self.object_mass

    def aim_throw(self):
        '''
        Re-solve the launch velocity so the ball's true path (drag and wind included) passes through the
        target point after the flight time T. With dv/dt = a - k v (k = damping / mass):
          p(T) = p0 + (a/k) T + (v0 - a/k) (1 - exp(-k T)) / k   ->   linear in v0
        '''
        if not DcmmCfg.aim_with_disturbances:
            return
        a = self.ball_acceleration()
        k = self.object_damping / self.object_mass
        T = self.throw_flight_time
        dp = (self.arm_base_pos + self.throw_target) - self.object_pos3d
        if k > 1e-9:
            decay = -math.expm1(-k * T) / k                  # (1 - exp(-k T)) / k
            v0 = a / k + (dp - (a / k) * T) / decay
            self.throw_arrival_vel = a / k + (v0 - a / k) * math.exp(-k * T)
        else:
            v0 = dp / T - 0.5 * a * T
            self.throw_arrival_vel = v0 + a * T
        self.object_vel6d = np.concatenate((v0, np.zeros(3)))

    def _reset_simulation(self):
        # Reset the data in Mujoco Simulation
        mujoco.mj_resetData(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_resetData(self.Dcmm.model_arm, self.Dcmm.data_arm)
        if self.Dcmm.model.na == 0:
            self.Dcmm.data.act[:] = None
        if self.Dcmm.model_arm.na == 0:
            self.Dcmm.data_arm.act[:] = None
        self.Dcmm.data.ctrl = np.zeros(self.Dcmm.model.nu)
        self.Dcmm.data_arm.ctrl = np.zeros(self.Dcmm.model_arm.nu)
        # Random Ball
        self.random_object()
        # Recompute the derived model constants after the mass change
        # (without this the ball falls with ~mass_new/mass_xml of the gravity)
        mujoco.mj_setConst(self.Dcmm.model, self.Dcmm.data)
        self.Dcmm.data.qpos[self.Dcmm.arm_qpos_adr] = DcmmCfg.arm_joints[:]
        self.Dcmm.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]
        # Random Gravity
        self.Dcmm.model.opt.gravity[2] = -9.81 + DcmmCfg.gravity_noise*np.random.uniform(-1, 1)
        # Random Throw (uses the gravity above)
        self.random_throw()
        self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                    velocity=np.zeros(6))
        # Random PID
        self.random_PID()
        # Random Delay
        self.random_delay()
        # Random ball noise and wind (drawn last, so the throw, ball and PID draws above stay the same per seed)
        self.random_obs_noise()
        self.random_wind()
        # Aim the throw with the ball's drag and wind (no random draws)
        self.aim_throw()
        # Forward Kinematics
        mujoco.mj_forward(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_forward(self.Dcmm.model_arm, self.Dcmm.data_arm)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        # Reset the basic simulation
        self._reset_simulation()
        self.init_pos = True
        self.object_throw = False
        self.steps = 0
        # Reset the time
        self.start_time = self.Dcmm.data.time

        ## Reset the target joint positions of the arm
        self.Dcmm.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
        ## Reset the episode state
        self.terminated = False
        self.arm_limit = True
        self.prev_action = np.zeros(self.act_dim)
        self._reset_episode_state()
        self.prev_ring_side = self._get_ring_side()[0]

        # Get the observation and info
        observation = self._get_obs()
        info = self._get_info()
        self._render_and_sync()

        return observation, info

    def _get_ring_side(self):
        '''
        Signed distance of the ball center to the ring plane (positive on the opening side)
        and its distance to the hoop axis.
        '''
        rel = self.Dcmm.data.body(self.Dcmm.object_name).xpos - self.Dcmm.data.site("hoop_center").xpos
        axis = self._get_hoop_axis()
        side = np.dot(rel, axis)
        radial = np.linalg.norm(rel - side * axis)
        return side, radial

    def _check_catch(self):
        # Catch: the ball center crosses the ring plane from the opening side, inside the ring
        side, radial = self._get_ring_side()
        if self.object_throw and self.prev_ring_side > 0 and side <= 0 and radial < DcmmCfg.hoop_ring_radius:
            self.outcome = OUTCOME_CATCH
            self.step_catch = True
        self.prev_ring_side = side

    def _check_contacts(self):
        '''
        Classify the contacts of this physics step:
        - ball vs rim: rim hit (penalty, the episode continues)
        - ball vs net/hoop body: allowed
        - ball vs floor: OUTCOME_BALL_FLOOR
        - ball vs arm links or base: OUTCOME_BALL_ROBOT
        - arm/hoop vs base or floor: OUTCOME_COLLISION
        '''
        data = self.Dcmm.data
        model = self.Dcmm.model
        for i in range(data.ncon):
            geom1 = data.contact.geom1[i]
            geom2 = data.contact.geom2[i]
            body1 = model.geom_bodyid[geom1]
            body2 = model.geom_bodyid[geom2]
            if self.print_contacts:
                print("contact: {} - {}".format(model.body(body1).name, model.body(body2).name))
            if geom1 == self.object_id or geom2 == self.object_id:
                other_geom, other_body = (geom2, body2) if geom1 == self.object_id else (geom1, body1)
                if other_body == self.rim_body_id:
                    self.step_rim = True
                    self.ball_touched = True
                elif other_body in self.hoop_body_ids:
                    self.ball_touched = True
                elif other_geom == self.floor_id:
                    self.outcome = OUTCOME_BALL_FLOOR
                elif other_body in self.arm_body_ids or other_body in self.base_body_ids:
                    self.outcome = OUTCOME_BALL_ROBOT
            else:
                moving1 = body1 in self.arm_body_ids or body1 in self.hoop_body_ids
                moving2 = body2 in self.arm_body_ids or body2 in self.hoop_body_ids
                static1 = geom1 == self.floor_id or body1 in self.base_body_ids
                static2 = geom2 == self.floor_id or body2 in self.base_body_ids
                if (moving1 and static2) or (moving2 and static1):
                    self.outcome = OUTCOME_COLLISION

    def _predict_path(self):
        '''
        Closest approach between the hoop center and the ball's predicted ballistic path
        (the reward uses this privileged prediction; the policy never sees it).

        Return:
        - d_path: closest distance (m)
        - v_closest: predicted ball velocity at the closest point
        '''
        p = self.Dcmm.data.body(self.Dcmm.object_name).xpos.copy()
        v = self._get_object_v_lin_3d()
        g = self.ball_acceleration()                         # gravity + wind / mass (only called after release)
        tau = self.path_tau[:, None]
        # Linear drag dv/dt = g - k v (k = damping / mass); k = 0 is the plain parabola
        k = self.object_damping / self.object_mass
        if k > 1e-9:
            decay = -np.expm1(-k * tau) / k                  # (1 - exp(-k tau)) / k
            path = p + (g / k) * tau + (v - g / k) * decay
            v_path = g / k + (v - g / k) * np.exp(-k * tau)
        else:
            path = p + tau * v + 0.5 * tau**2 * g
            v_path = v + g * tau
        # Cut the path where the ball would reach the floor
        below = np.nonzero(path[:, 2] < self.object_radius)[0]
        n = max(1, below[0]) if below.size else len(path)
        dist = np.linalg.norm(path[:n] - self.Dcmm.data.site("hoop_center").xpos, axis=1)
        i = np.argmin(dist)
        return dist[i], v_path[i]

    def compute_reward(self, action):
        '''
        Rewards (weights in DcmmCfg.reward_weights):
        - Path Progress / Path Near / Alignment Reward (dense, after the throw, before the ball touches anything)
        - Catch Reward
        - Rim Penalty
        - Ball Fail Penalty
        - Collision Penalty
        - Smoothness Penalty
        - Constraint (IK failure) Penalty
        '''
        w = DcmmCfg.reward_weights
        reward_path_progress = 0.0
        reward_path_near = 0.0
        reward_align = 0.0
        if self.object_throw and not self.ball_touched:
            d_path, v_closest = self._predict_path()
            self.min_path_dist = min(self.min_path_dist, d_path)
            ## Path Progress Reward
            # The path only changes when the hoop moves, so this rewards moving the hoop onto the ball's path
            if self.prev_d_path is not None:
                reward_path_progress = (self.prev_d_path - d_path) * w["r_path_progress"]
            self.prev_d_path = d_path
            ## Path Near Reward
            reward_path_near = math.exp(-(d_path / DcmmCfg.path_near_sigma)**2) * w["r_path_near"]
            ## Alignment Reward
            # The hoop opening should face the incoming ball: hoop_axis . ball_dir = -1
            speed = np.linalg.norm(v_closest)
            if speed > 0.5:
                cos_align = np.dot(self._get_hoop_axis(), v_closest) / speed
                reward_align = max(0.0, -cos_align) * math.exp(-(d_path / DcmmCfg.align_sigma)**2) * w["r_align"]
        else:
            self.prev_d_path = None

        ## Event Rewards
        reward_catch = w["r_catch"] if self.step_catch else 0.0
        reward_rim = w["r_rim"] if self.step_rim else 0.0
        reward_ball_fail = w["r_ball_fail"] if self.outcome in [OUTCOME_BALL_FLOOR, OUTCOME_BALL_ROBOT] else 0.0
        reward_collision = w["r_collision"] if self.outcome == OUTCOME_COLLISION else 0.0

        ## Smoothness Penalty
        reward_smooth = - np.sum((action - self.prev_action)**2) * w["r_smooth"]

        ## Constraint Penalty
        # Compute the Penalty when the IK cannot reach the commanded hoop pose
        reward_constraint = 0 if self.arm_limit else -w["r_constraint"]

        rewards = reward_path_progress + reward_path_near + reward_align + reward_catch + reward_rim \
                + reward_ball_fail + reward_collision + reward_smooth + reward_constraint
        if self.print_reward:
            print("##### print reward")
            print("path_progress: {:.3f}, path_near: {:.3f}, align: {:.3f}, catch: {:.3f}, rim: {:.3f}, \n".format(
                reward_path_progress, reward_path_near, reward_align, reward_catch, reward_rim
            ) + "ball_fail: {:.3f}, collision: {:.3f}, smooth: {:.3f}, constraint: {:.3f}".format(
                reward_ball_fail, reward_collision, reward_smooth, reward_constraint
            ))
            print("total reward: {:.3f}\n".format(rewards))
        return rewards

    def _step_mujoco_simulation(self, action):
        ## Map the normalized action to a new hoop target
        delta_pos = action[0:3] * DcmmCfg.action_scale["pos"]
        delta_tilt = action[3:5] * DcmmCfg.action_scale["tilt"]
        target_center, target_rot = self.Dcmm.get_target_hoop_pose()
        hoop_pos3d = self._get_hoop_pos3d()
        # Keep the target close to the actual hoop and inside the workspace box
        target_center = hoop_pos3d + clip_norm(target_center + delta_pos - hoop_pos3d, DcmmCfg.hoop_target_lead)
        target_center = np.clip(target_center, DcmmCfg.hoop_target_low, DcmmCfg.hoop_target_high)
        # Tilt about the hoop's own x and y axes (spin about the hoop axis is left to the IK)
        target_rot = target_rot @ R.from_euler('xy', delta_tilt).as_matrix()
        result_QP = self.Dcmm.move_hoop_pose(target_center, target_rot)
        if result_QP[1]:
            self.arm_limit = True
            self.Dcmm.target_arm_qpos[:] = result_QP[0]
        else:
            # print("IK Failed!!!")
            self.arm_limit = False
            self.ik_fails += 1
        # Add Target Action to the Buffer
        self.update_target_ctrl()
        # Reset the per-step events
        self.step_catch = False
        self.step_rim = False
        for _ in range(self.steps_per_policy):
            # Update the control command according to the latest policy output
            self.Dcmm.data.ctrl[self.Dcmm.arm_ctrl_ids] = self._get_ctrl()
            if self.render_per_step:
                # Rendering
                img = self.render()
            # Hold the ball still during the static time, then throw it
            if self.Dcmm.data.time - self.start_time < self.object_static_time:
                self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                            velocity=np.zeros(6))
                self.Dcmm.data.ctrl[self.Dcmm.object_ctrl_id] = self.object_mass * -self.Dcmm.model.opt.gravity[2]
            elif not self.object_throw:
                self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                            velocity=self.object_vel6d[:])
                self.Dcmm.data.ctrl[self.Dcmm.object_ctrl_id] = 0.0
                # Wind acts from the release on (mj_resetData clears it at the next reset)
                self.Dcmm.data.xfrc_applied[self.object_body_id, 0:3] = self.wind_force
                self.object_throw = True
                self.throw_time = self.Dcmm.data.time - self.start_time

            mujoco.mj_step(self.Dcmm.model, self.Dcmm.data)

            # Update the contact information and the catch criterion
            self._check_contacts()
            if self.outcome == OUTCOME_NONE:
                self._check_catch()
            # End the episode in advance on catch / ball fail / collision
            if self.outcome != OUTCOME_NONE:
                self.terminated = True
                break
        if self.step_rim:
            self.rim_hits += 1

    def step(self, action):
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self._step_mujoco_simulation(action)
        # Design the reward function
        reward = self.compute_reward(action)
        self.prev_action = action.copy()
        # The episode is truncated if the env_time is larger than the predefined time
        truncated = False
        if not self.terminated and self.Dcmm.data.time - self.start_time > self.env_time:
            truncated = True
            self.outcome = OUTCOME_TIMEOUT
        # Get the obs and info
        obs = self._get_obs()
        info = self._get_info()
        self._render_and_sync()
        terminated = self.terminated
        return obs, reward, terminated, truncated, info

    def _render_and_sync(self):
        # Rendering is only needed to look at the camera images, the policy uses states only
        if self.imshow_cam:
            self.render()
        # Sync the viewer (if exists) with the data
        if self.Dcmm.viewer != None:
            # Run in real time when watching
            if self.last_sync_time is not None:
                time.sleep(max(0.0, 1.0 / self.fps - (time.time() - self.last_sync_time)))
            self.Dcmm.viewer.sync()
            self.last_sync_time = time.time()

    def render(self):
        imgs = np.zeros((0, self.img_size[0], self.img_size[1]))
        for camera_name in self.camera_name:
            img = self.mujoco_renderer.render(
                self.render_mode, camera_name = camera_name
            )
            if self.imshow_cam and self.render_mode == "rgb_array":
                cv.imshow(camera_name, cv.cvtColor(img, cv.COLOR_BGR2RGB))
                cv.waitKey(1)
            # Converts the depth array valued from 0-1 to real meters
            elif self.render_mode == "depth_array":
                img = self.Dcmm.depth_2_meters(img)
                if self.imshow_cam:
                    depth_norm = np.zeros(img.shape, dtype=np.uint8)
                    cv.convertScaleAbs(img, depth_norm, alpha=(255.0/img.max()))
                    cv.imshow(camera_name+"_depth", depth_norm)
                    cv.waitKey(1)
        return imgs

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        if self.Dcmm.viewer != None: self.Dcmm.viewer.close()

    def oracle_action(self):
        '''
        Privileged scripted controller (for testing the env, not a policy):
        drive the hoop to the throw's target point, facing the incoming ball.
        '''
        action = np.zeros(self.act_dim)
        action[0:3] = (self.throw_target - self._get_hoop_pos3d()) / DcmmCfg.action_scale["pos"]
        axis = self._get_hoop_axis()
        axis_des = -self.throw_arrival_vel / np.linalg.norm(self.throw_arrival_vel)
        rot_vec = np.cross(axis, axis_des)
        sin_angle = np.linalg.norm(rot_vec)
        if sin_angle > 1e-6:
            angle = math.atan2(sin_angle, np.dot(axis, axis_des))
            rot_mat = self.Dcmm.data.site("hoop_center").xmat.reshape(3, 3)
            rot_local = rot_mat.T @ (rot_vec / sin_angle * angle)
            action[3:5] = rot_local[0:2] / DcmmCfg.action_scale["tilt"]
        return np.clip(action, -1, 1)

    def run_test(self, oracle=False, episodes=0):
        global key_action
        self.reset()
        num_episodes = 0
        outcomes = np.zeros(len(OUTCOME_NAMES), dtype=int)
        while episodes <= 0 or num_episodes < episodes:
            t_start = time.time()
            if oracle:
                action = self.oracle_action()
            else:
                # Keyboard control
                action = key_action.copy()
                key_action[:] = 0.0
            observation, reward, terminated, truncated, info = self.step(action)
            if terminated or truncated:
                num_episodes += 1
                outcomes[info["outcome"]] += 1
                print("episode {}: {} | min ball dist {:.3f} | min path dist {:.3f} | rim hits {} | ik fails {} | catch rate {:.2f}".format(
                    num_episodes, OUTCOME_NAMES[info["outcome"]], info["min_ball_dist"], info["min_path_dist"],
                    info["rim_hits"], info["ik_fails"], outcomes[OUTCOME_CATCH] / num_episodes))
                self.reset()
            if self.Dcmm.viewer != None:
                # Run in real time when watching
                time.sleep(max(0.0, 1.0 / self.fps - (time.time() - t_start)))
        print("outcomes: ", dict(zip(OUTCOME_NAMES, outcomes)))

if __name__ == "__main__":
    os.chdir('../../')
    parser = argparse.ArgumentParser(description="Args for DcmmVecEnv")
    parser.add_argument('--viewer', action='store_true', help="open the mujoco.viewer or not")
    parser.add_argument('--imshow_cam', action='store_true', help="imshow the camera image or not")
    parser.add_argument('--oracle', action='store_true', help="drive the hoop with the privileged scripted controller")
    parser.add_argument('--episodes', type=int, default=0, help="number of episodes to run (0: forever)")
    parser.add_argument('--print_reward', action='store_true', help="print the reward terms every step")
    args = parser.parse_args()
    print("args: ", args)
    env = DcmmVecEnv(task='HoopCatch', object_name='object', render_per_step=False,
                    print_reward=args.print_reward, print_info=False,
                    print_contacts=False, print_ctrl=False,
                    print_obs=False, camera_name = ["top"],
                    render_mode="rgb_array", imshow_cam=args.imshow_cam,
                    viewer = args.viewer,
                    env_time = 2.5, steps_per_policy=20)
    env.run_test(oracle=args.oracle, episodes=args.episodes)