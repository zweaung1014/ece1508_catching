import os
import numpy as np
from pathlib import Path

## Define the model path
path = os.path.realpath(__file__)
root = str(Path(path).parent)
ASSET_PATH = os.path.join(root, "../../assets")
# print("ASSET_PATH: ", ASSET_PATH)
# Fixed-base xArm6 with the hoop end-effector
XML_DCMM_HOOP_OBJECT_PATH = "urdf/x1_xarm6_leap_right_object.xml"
XML_ARM_PATH = "urdf/xarm6_right.xml"
## Weight Saved Path
WEIGHT_PATH = os.path.join(ASSET_PATH, "weights")

## Define the initial (home) joint positions of the arm
# At home the hoop center is ~(0.0, 0.39, 0.32) m from the arm base, opening faces forward-up
arm_joints = np.array([
   0.0, 0.0, -0.0, 3.07, 2.25, -1.5
])

## Hoop geometry (must match the hoop in the MJCF)
# Inner radius of the orange ring (ring radius 0.133 - capsule radius 0.008)
hoop_ring_radius = 0.125

## Define the action
# The policy outputs 5 numbers in [-1, 1]:
#   [0:3] delta of the hoop center (x, y, z in the arm-base frame), scaled by action_scale["pos"] (m)
#   [3:5] tilt of the hoop axis about the hoop's own x and y axes, scaled by action_scale["tilt"] (rad)
# Spinning about the hoop axis is useless for a symmetric hoop, so it is not an action.
action_scale = {
    "pos": 0.04,
    "tilt": 0.1,
}
# Bounds of the hoop-center target (arm-base frame, m) so the IK target cannot run off
hoop_target_low = np.array([-0.75, -0.05, 0.0])
hoop_target_high = np.array([0.75, 0.9, 1.1])
# Max distance the hoop-center target may lead the actual hoop center (m)
hoop_target_lead = 0.15

## Define the reward weights
reward_weights = {
    # Dense shaping, only after the ball is thrown.
    # d_path = closest distance between the hoop center and the ball's predicted ballistic path
    "r_path_progress": 5.0,     # * (d_path_prev - d_path)
    "r_path_near": 0.5,         # * exp(-(d_path / path_near_sigma)^2)
    "r_align": 0.5,             # * max(0, -hoop_axis . ball_dir_at_closest_point) * exp(-(d_path / align_sigma)^2)
    # Events
    "r_catch": 100.0,           # ball center crosses the ring plane inside the ring (episode ends)
    "r_rim": -2.0,              # per policy step with ball-rim contact
    "r_ball_fail": -10.0,       # ball touches the floor, arm links or the base (episode ends)
    "r_collision": -10.0,       # arm/hoop touches the base or the floor (episode ends)
    # Regularization
    "r_smooth": 0.02,           # * -||a_t - a_{t-1}||^2 (normalized actions)
    "r_constraint": 1.0,        # * -1 when the IK fails
}
path_near_sigma = 0.1
align_sigma = 0.2
# Horizon (s) and resolution (s) of the ballistic path prediction used by the reward
path_horizon = 1.5
path_dt = 0.01

## Define the throw sampler
# Every throw is aimed so the ball passes through a target point inside the catch zone.
# Catch zone, relative to the arm base: horizontal distance r, azimuth from +y (forward), height z
catch_zone = {
    "r": np.array([0.35, 0.65]),
    "azimuth": np.radians(np.array([-45.0, 45.0])),
    "z": np.array([0.3, 0.8]),
}
# Launch point: horizontal distance r from the arm base, azimuth from +y, height above the floor (world z)
launch_zone = {
    "r": np.array([2.0, 3.0]),
    "azimuth": np.radians(np.array([-60.0, 60.0])),
    "z": np.array([0.8, 1.6]),
}
# Time of flight from the launch point to the target point (s)
flight_time = np.array([0.6, 1.0])

## Define the camera params for the MujocoRenderer.
cam_config = {
    "name": "top",
    "width": 640,
    "height": 480,
}

## Define IK
ik_config = {
    "solver_type": "QP",
    "ps": 0.001,
    "λΣ": 12.5,
    "ilimit": 100,
    "ee_tol": 1e-4,
    # Ignore the rotation about the end-effector z axis (= hoop axis, = joint6 axis)
    "free_ee_z_rot": True,
    # Step gain of the QP IK (the original 2.0 oscillated and failed ~70% of the hoop moves; 1.0 fails ~13%)
    "gain": 1.0,
}

# Define the Randomization Params (drawn per throw; ranges of exp03, see results/exp03_harder/README.md)
## Arm Joints: each joint's PID gains are scaled by a factor in this range (payload / tuning differences)
k_arm = np.array([0.5, 1.5])
## Ball (sphere)
object_radius = np.array([0.035, 0.045])
object_mass = np.array([0.035, 0.075])
# Air drag: damping b on the ball's free joint gives the force -b*v, so the speed decays as exp(-(b/m) t).
# At 0.015 a 35 g ball loses ~35% of its speed over 1 s (a 75 g ball ~18%), about a light foam ball.
object_damping = np.array([0.0, 0.015])
## Wind: constant horizontal force on the ball after release, magnitude drawn from this range (N),
# direction uniform. 0.04 N is about what a 5 m/s breeze puts on a 4 cm ball (0.5 rho Cd A v^2).
wind_force = np.array([0.0, 0.04])
## Throw aim: True = the launch velocity accounts for the ball's drag and wind, so the true path still
# passes through the target point (every throw stays catchable; the path is just not a parabola).
# False = aimed as if there were no drag or wind (exp02), so the ball misses the aim point.
aim_with_disturbances = True
object_static = np.array([0.5, 0.75])
## Gravity: -9.81 +/- gravity_noise (0 = fixed at 9.81)
gravity_noise = 0.0
## Observation Noise (Gaussian std)
k_obs_arm = 0.001
k_obs_hoop = 0.001
# Ball position noise: the std is drawn per throw from this range (m); a good to mediocre depth camera at 2-3 m.
k_obs_object = np.array([0.005, 0.02])
# Ball velocity noise: the std is drawn per throw from this range (m/s); lightly filtered differences
# of those positions (unfiltered differences at 25 Hz would be ~0.18-0.7 m/s).
k_obs_object_vel = np.array([0.1, 0.4])
## Actions Noise: the torque of each joint is scaled by N(1, k_act) on every physics step
k_act = 0.10
## Action Delay
# Length of the joint-target buffer, drawn per throw: 1 = the arm follows the latest command,
# n = the command from n - 1 control steps (40 ms each) ago; up to 4 steps (160 ms).
act_delay = {
    'arm': [1, 2, 3, 4, 5],
}

## Define PID params for the arm.
Kp_arm = np.array([300.0, 400.0, 400.0, 50.0, 200.0, 20.0])
Ki_arm = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3])
Kd_arm = np.array([40.0, 40.0, 40.0, 5.0, 10.0, 1])
llim_arm = np.array([-300.0, -300.0, -300.0, -50.0, -50.0, -20.0])
ulim_arm = np.array([300.0, 300.0, 300.0, 50.0, 50.0, 20.0])