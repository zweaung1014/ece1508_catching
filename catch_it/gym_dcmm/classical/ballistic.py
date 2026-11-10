"""
Author: Zwe Min Htet Aung
ABOUT: ballistic (parabola) prediction of the thrown ball for the classical catcher.

Implements Eqs. (1)-(3) of M. Campana and J.-P. Laumond, "Ballistic motion planning", IROS 2016
(docs/Campana and Laumond - 2016 - Ballistic motion planning.pdf, Section III-A). The paper's
point-robot takeoff state (c_s, c_dot_s) is replaced by the ball's estimated state. The friction
cones, velocity limits (Eqs. 4-7) and the roadmap planner are not used.

All vectors are in the arm-base frame (world-aligned axes, +z up). g is the magnitude of gravity,
so the acceleration is -g e_z.
"""
import numpy as np

E_Z = np.array([0.0, 0.0, 1.0])


class Parabola:
    """
    Ballistic path of a point that is at c_s with velocity c_dot_s at time t_s.

    Args:
    - c_s: position at t_s (3,)
    - c_dot_s: velocity at t_s (3,)
    - t_s: time of that state (s)
    - g: gravity magnitude (m/s^2)
    """
    def __init__(self, c_s, c_dot_s, t_s=0.0, g=9.81):
        self.c_s = np.asarray(c_s, dtype=np.float64).copy()
        self.c_dot_s = np.asarray(c_dot_s, dtype=np.float64).copy()
        self.t_s = float(t_s)
        self.g = float(g)
        # The path lies in the vertical plane pi_theta through c_s (paper Fig. 2)
        self.theta = np.arctan2(self.c_dot_s[1], self.c_dot_s[0])
        self.Theta = np.array([np.cos(self.theta), np.sin(self.theta), 0.0])
        self.x_theta_s = self.c_s @ self.Theta              # x_theta_s = c_s . Theta
        self.x_dot_theta_s = self.c_dot_s @ self.Theta      # horizontal speed, constant in flight
        self.z_s = self.c_s[2]
        self.z_dot_s = self.c_dot_s[2]
        self.alpha_s = np.arctan2(self.z_dot_s, self.x_dot_theta_s)  # takeoff angle

    # ---- Eq. (1): position and velocity over time ----
    def position(self, t):
        """c(t) = -g/2 tau^2 e_z + c_dot_s tau + c_s, tau = t - t_s. t can be a scalar or an array (N,)."""
        tau = np.asarray(t, dtype=np.float64) - self.t_s
        return self.c_s + np.multiply.outer(tau, self.c_dot_s) - 0.5 * self.g * np.multiply.outer(tau**2, E_Z)

    def velocity(self, t):
        """Time derivative of Eq. (1)."""
        tau = np.asarray(t, dtype=np.float64) - self.t_s
        return self.c_dot_s - self.g * np.multiply.outer(tau, E_Z)

    # ---- Eqs. (2)-(3): the same path written in the vertical plane pi_theta ----
    def x_theta(self, c):
        """Horizontal coordinate in pi_theta: x_theta = c . Theta."""
        return np.asarray(c) @ self.Theta

    def z_of_x(self, x_theta):
        """Eq. (2): z = -g/2 (x - x_s)^2 / x_dot_s^2 + (z_dot_s / x_dot_s)(x - x_s) + z_s."""
        dx = np.asarray(x_theta) - self.x_theta_s
        return -0.5 * self.g * dx**2 / self.x_dot_theta_s**2 + self.z_dot_s / self.x_dot_theta_s * dx + self.z_s

    def slope(self, x_theta):
        """Eq. (3): dz/dx_theta = z_dot / x_dot_s = -g (x - x_s) / x_dot_s^2 + z_dot_s / x_dot_s."""
        dx = np.asarray(x_theta) - self.x_theta_s
        return -self.g * dx / self.x_dot_theta_s**2 + self.z_dot_s / self.x_dot_theta_s

    def time_of_x(self, x_theta):
        """The horizontal speed is constant, so t = t_s + (x - x_s) / x_dot_s."""
        return self.t_s + (np.asarray(x_theta) - self.x_theta_s) / self.x_dot_theta_s

    def direction(self, t):
        """
        Unit direction of travel at time t, from the slope of Eq. (3): (Theta + slope e_z) / norm.
        Falls back to the velocity of Eq. (1) for (nearly) vertical throws.
        """
        if abs(self.x_dot_theta_s) < 1e-6:
            v = np.atleast_2d(self.velocity(t))
            return np.squeeze(v / np.linalg.norm(v, axis=-1, keepdims=True))
        x = self.x_theta(self.position(t))
        s = np.asarray(self.slope(x))
        d = np.multiply.outer(np.ones_like(s), self.Theta) + np.multiply.outer(s, E_Z)
        d = d * np.sign(self.x_dot_theta_s)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


def fit_parabola(t_pos, positions, t_vel, velocities, t_ref, vel_window,
                 g=9.81, fit_g=False, g_prior_sigma=0.5, pos_sigma=0.005, vel_sigma=0.05):
    """
    Weighted least-squares estimate of the ball state at t_ref from noisy reads.

    Position reads:  p_i = c + c_dot tau_i - g/2 tau_i^2 e_z            (+ noise pos_sigma)
    Velocity reads:  v_j = c_dot - g (tau_j - vel_window/2) e_z          (+ noise vel_sigma)
    tau = t - t_ref. A velocity read is a finite difference over the last vel_window seconds, so it
    equals the velocity at the middle of that window, not at the read time.

    With one position and one velocity read and fit_g=False the system is exactly determined
    (the single-read estimate). With fit_g=True the vertical acceleration is an extra unknown with a
    Gaussian prior N(g, g_prior_sigma^2).

    Return: Parabola with the estimated state at t_ref (and the fitted g)
    """
    t_pos = np.asarray(t_pos, dtype=np.float64).reshape(-1)
    t_vel = np.asarray(t_vel, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
    n_unknown = 7 if fit_g else 6   # c (3), c_dot (3), [g]
    rows, rhs = [], []
    wp, wv = 1.0 / pos_sigma, 1.0 / vel_sigma
    for tau, p in zip(t_pos - t_ref, positions):
        for k in range(3):
            a = np.zeros(n_unknown)
            a[k], a[3 + k] = 1.0, tau
            b = p[k]
            if k == 2:
                if fit_g:
                    a[6] = -0.5 * tau**2
                else:
                    b += 0.5 * g * tau**2
            rows.append(a * wp); rhs.append(b * wp)
    for tau, v in zip(t_vel - t_ref, velocities):
        tau_mid = tau - 0.5 * vel_window
        for k in range(3):
            a = np.zeros(n_unknown)
            a[3 + k] = 1.0
            b = v[k]
            if k == 2:
                if fit_g:
                    a[6] = -tau_mid
                else:
                    b += g * tau_mid
            rows.append(a * wv); rhs.append(b * wv)
    if fit_g:
        a = np.zeros(n_unknown)
        a[6] = 1.0 / g_prior_sigma
        rows.append(a); rhs.append(g / g_prior_sigma)
    sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    g_hat = sol[6] if fit_g else g
    return Parabola(sol[0:3], sol[3:6], t_ref, g_hat)
