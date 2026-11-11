"""
Author: Zwe Min Htet Aung
ABOUT: settings of the classical (non-learning) catcher, gym_dcmm/classical/catcher.py.
The catcher sees the same observation and uses the same 5-D action interface as the RL policy.
Everything here is what the catcher *assumes*; the env (DcmmCfg) sets what really happens.
"""
import numpy as np

## Sensor model the catcher assumes (tracker spec).
# Set from the env's noise ranges, the way an engineer sets a filter from the sensor's data sheet:
# the middle of DcmmCfg.k_obs_object (5-20 mm) and DcmmCfg.k_obs_object_vel (0.1-0.4 m/s).
# exp01/exp02 used 0.005 and 0.05 (their noise levels).
pos_sigma = 0.0125      # std of the observed ball position (m)
vel_sigma = 0.25        # std of the observed ball velocity (m/s)

## Gravity model
g_nominal = 9.81        # m/s^2, the parabola assumes this unless the vertical acceleration is fitted
# Multi-read only: also estimate the vertical acceleration, with a Gaussian prior around g_nominal.
# With few reads the prior dominates; with many reads the data does.
fit_gravity = True
g_prior_sigma = 0.5     # m/s^2

## Release detection: the ball has left the launch point when BOTH hold
# About 4x the largest noise std, so noise alone does not trigger it (with exp02's 0.03 / 1.0, the
# exp03 noise triggered it on ~6% of throws while the ball was still held). A thrown ball moves
# 15-20 cm per control step, so the real release is still seen at the first read after it.
release_disp = 0.08     # observed position is this far (m) from the averaged rest position
release_speed = 1.6     # observed speed (m/s) is above this

## Region where the hoop can catch (arm-base frame, azimuth measured from +y).
# From the hoop reachability study (docs/hoop_rl_design.md section 3). Kept separate from
# DcmmCfg.catch_zone on purpose: the catcher must not learn where the throw sampler aims.
reach_region = {
    "r": np.array([0.35, 0.65]),
    "azimuth": np.radians(np.array([-45.0, 45.0])),
    "z": np.array([0.3, 0.8]),
}

## Hoop motion model, used only to decide whether a catch point can be reached in time.
# Kept below the interface caps (1.0 m/s and 2.5 rad/s) to leave room for the PID lag.
hoop_speed = 0.8        # m/s
hoop_tilt_rate = 2.0    # rad/s
settle_time = 0.1       # s, added to every move
tilt_cost = 0.2         # m per rad: how a tilt compares to a translation when choosing the catch point

## Catch point search along the predicted path
candidate_dt = 0.01     # s between candidate points
min_lead = 0.04         # s, candidates must be at least one control step ahead
horizon = 1.5           # s of predicted path to search

## Multi-read re-planning
commit_time = 0.15      # s before the planned catch time: stop re-selecting, only refine the point
replan_slack = -0.05    # s, keep the current catch time while its slack stays above this

## Ready pose while the ball is held (uses only the observed rest position of the ball)
# The hoop waits straight ahead in the middle of the reach region (the shortest worst-case move:
# throws often cross from one side to the other), with its axis turned toward the launch point.
ready_pose = True
ready_r = 0.5           # m from the arm base, straight ahead (+y)
ready_z = 0.55          # m above the arm base
ready_elevation = np.radians(45.0)  # hoop axis elevation; the axis points back toward the launch point
