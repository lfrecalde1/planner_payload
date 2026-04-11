#!/usr/bin/env python3
"""ROS 2 node that publishes a pendulum-like tether simulation using
a classic softplus implicit-complementarity solver.

This node replaces the explicit IPM variables (s, gamma) and the equation
    s * gamma = mu
with the softplus implicit mapping

    gamma = b_mu(w)
    s     = b_mu(-w)

where
    b_mu(w) = 0.5 * (w + sqrt(w^2 + 4*mu))

The inner unknown is:
    z = [v_next(6), w]

Published topics:
- /carrier/odom
- /payload/odom
- /cable/tension
- /constrained_dynamics/solve_time_ms
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import casadi as ca
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


Vec = np.ndarray


@dataclass
class Params:
    m1: float = 1.0
    m2: float = 0.2
    g: float = 9.81
    cable_length: float = 1.0
    dt: float = 0.03
    inner_iters: int = 150
    mu: float = 0.001
    damping1: float = 0.1
    damping2: float = 0.1

    # Numerical settings
    linear_reg: float = 1.0e-10
    residual_tol: float = 1.0e-9


@dataclass
class State:
    x1: Vec
    x2: Vec
    v1: Vec
    v2: Vec


CARRIER_ANCHOR = np.array([0.0, 0.0, 0.0], dtype=float)
CARRIER_DESIRED = np.array([1.5, 0.0, 1.5], dtype=float)


def nominal_control(params: Params) -> Vec:
    return np.array([0.0, 0.0, (params.m1 + params.m2) * params.g], dtype=float)


def geometry_terms(x1: Vec, x2: Vec, v_next: Vec, params: Params) -> tuple[float, Vec, Vec]:
    d = x2 - x1
    dist = np.linalg.norm(d)
    if dist < 1.0e-9:
        dist = 1.0e-9
        n = np.array([0.0, 0.0, -1.0], dtype=float)
    else:
        n = d / dist

    phi = dist - params.cable_length
    rel_vel = v_next[3:] - v_next[:3]
    n_dot = ((np.eye(3) - np.outer(n, n)) / dist) @ rel_vel
    return phi, n, n_dot


def fixed_carrier_controller(state: State, params: Params) -> Vec:
    """Controller that keeps the carrier close to the desired point."""
    kp = np.diag([60.0, 60.0, 100.0])
    kd = np.diag([18.0, 18.0, 20.0])
    gravity_offset = np.array([0.0, 0.0, -params.m2 * params.g], dtype=float)
    return gravity_offset + kp @ (CARRIER_DESIRED - state.x1) - kd @ state.v1


class InnerSoftplusStep:
    """Classic inner solve with implicit complementarity using softplus.

    Unknown:
        z = [v_next(6), w]

    Complementarity variables:
        gamma = b_mu(w)
        s     = b_mu(-w)

    Residual:
        F1 = M (v - v_free) + dt * J * gamma
        F2 = s + phi + dt * J v
    """

    def __init__(self, params: Params):
        self.params = params
        self.M = np.diag([params.m1] * 3 + [params.m2] * 3)
        self.M_inv = np.diag([1.0 / params.m1] * 3 + [1.0 / params.m2] * 3)
        self._build()

    @staticmethod
    def bmu_sym(w, mu):
        return 0.5 * (w + ca.sqrt(w * w + 4.0 * mu))

    def _build(self):
        p = self.params

        z = ca.SX.sym("z", 7)  # [v(6), w]
        v = z[:6]
        w = z[6]

        # Parameters: x1, x2, v1, v2, u
        pvec = ca.SX.sym("pvec", 15)
        x1 = pvec[0:3]
        x2 = pvec[3:6]
        v1 = pvec[6:9]
        v2 = pvec[9:12]
        u = pvec[12:15]

        d = x2 - x1
        dist = ca.sqrt(ca.dot(d, d) + 1.0e-12)
        n = d / dist
        phi = dist - p.cable_length
        J = ca.vertcat(-n, n)

        gvec = ca.DM([0.0, 0.0, -p.g])
        tau = ca.vertcat(
            p.m1 * gvec - p.damping1 * v1 + ca.DM(nominal_control(p)) + u,
            p.m2 * gvec - p.damping2 * v2,
        )

        v_curr = ca.vertcat(v1, v2)
        M = ca.DM(self.M)
        M_inv = ca.DM(self.M_inv)
        v_free = v_curr + p.dt * (M_inv @ tau)

        gamma = self.bmu_sym(w, p.mu)
        s = self.bmu_sym(-w, p.mu)

        # Classic softplus formulation: no stabilization term
        F1 = M @ (v - v_free) + p.dt * J * gamma
        F2 = s + phi + p.dt * ca.dot(J, v)
        F = ca.vertcat(F1, F2)

        JF = ca.jacobian(F, z)
        merit = 0.5 * ca.dot(F, F)

        self.F_fun = ca.Function("F_soft", [z, pvec], [F])
        self.JF_fun = ca.Function("JF_soft", [z, pvec], [JF])
        self.merit_fun = ca.Function("merit_soft", [z, pvec], [merit])
        self.s_fun = ca.Function("s_soft", [z], [s])
        self.gamma_fun = ca.Function("g_soft", [z], [gamma])

    def pack(self, state: State, u: Vec):
        return np.concatenate((state.x1, state.x2, state.v1, state.v2, u)).astype(float)

    def v_free(self, state: State, u: Vec) -> Vec:
        gvec = np.array([0.0, 0.0, -self.params.g], dtype=float)
        tau = np.hstack((
            self.params.m1 * gvec - self.params.damping1 * state.v1 + nominal_control(self.params) + u,
            self.params.m2 * gvec - self.params.damping2 * state.v2,
        ))
        v = np.hstack((state.v1, state.v2))
        return v + self.params.dt * (self.M_inv @ tau)

    def residual(self, z: Vec, state: State, u: Vec) -> Vec:
        return np.array(self.F_fun(z, self.pack(state, u))).astype(float).reshape(-1)

    def jacobian_z(self, z: Vec, state: State, u: Vec) -> np.ndarray:
        return np.array(self.JF_fun(z, self.pack(state, u))).astype(float)

    def merit(self, z: Vec, state: State, u: Vec) -> float:
        return float(self.merit_fun(z, self.pack(state, u)))

    def unpack_aux(self, z: Vec) -> tuple[float, float]:
        s = float(np.array(self.s_fun(z)).reshape(()))
        gamma = float(np.array(self.gamma_fun(z)).reshape(()))
        return s, gamma

    def solve(self, state: State, u: Vec, guess: Vec | None = None) -> Vec:
        vf = self.v_free(state, u)

        phi0, n0, _ = geometry_terms(state.x1, state.x2, vf, self.params)
        J0 = np.hstack((-n0, n0))

        # Initial slack guess from classic constraint residual
        s0 = -(phi0 + self.params.dt * float(J0 @ vf))
        s0 = max(1.0e-6, s0 + 1.0e-4)
        gamma0 = self.params.mu / s0
        w0 = gamma0 - s0

        if guess is None:
            z = np.concatenate((vf, np.array([w0], dtype=float)))
        else:
            z = guess.copy()

        for _ in range(self.params.inner_iters):
            F = self.residual(z, state, u)
            if np.linalg.norm(F, ord=np.inf) < self.params.residual_tol:
                break

            JF = self.jacobian_z(z, state, u)
            dz = -np.linalg.solve(JF + self.params.linear_reg * np.eye(JF.shape[0]), F)

            merit0 = 0.5 * np.dot(F, F)
            alpha = 1.0
            accepted = False

            while alpha > 1.0e-6:
                z_try = z + alpha * dz
                F_try = self.residual(z_try, state, u)
                merit_try = 0.5 * np.dot(F_try, F_try)

                if merit_try < merit0:
                    z = z_try
                    accepted = True
                    break

                alpha *= 0.5

            if not accepted:
                break

        return z


class InnerSoftplusPendulumNode(Node):
    def __init__(self) -> None:
        super().__init__("inner_softplus_pendulum_node")

        self.params = Params()
        self.stepper = InnerSoftplusStep(self.params)

        self.state = State(
            x1=CARRIER_ANCHOR.copy(),
            x2=np.array([0.2, 0.0, -0.2], dtype=float),
            v1=np.zeros(3, dtype=float),
            v2=np.zeros(3, dtype=float),
        )

        self.z_guess = None
        self.sim_time = 0.0

        self.carrier_pub = self.create_publisher(Odometry, "/carrier/odom", 10)
        self.payload_pub = self.create_publisher(Odometry, "/payload/odom", 10)
        self.tension_pub = self.create_publisher(Float64, "/cable/tension", 10)
        self.solve_time_pub = self.create_publisher(Float64, "/constrained_dynamics/solve_time_ms", 10)

        self.timer = self.create_timer(self.params.dt, self.step)

        self.get_logger().info("Inner softplus pendulum node started")

    def step(self) -> None:
        u = fixed_carrier_controller(self.state, self.params)

        t0 = time.perf_counter()
        z = self.stepper.solve(self.state, u, self.z_guess)
        solve_ms = 1.0e3 * (time.perf_counter() - t0)
        self.z_guess = z

        v_next = z[:6]
        s, gamma = self.stepper.unpack_aux(z)

        x_next = np.hstack((self.state.x1, self.state.x2)) + self.params.dt * v_next
        self.state = State(x_next[:3], x_next[3:], v_next[:3], v_next[3:])
        self.sim_time += self.params.dt

        now = self.get_clock().now().to_msg()
        self.carrier_pub.publish(self._make_odom(now, "world", "carrier", self.state.x1, self.state.v1))
        self.payload_pub.publish(self._make_odom(now, "world", "payload", self.state.x2, self.state.v2))
        self.tension_pub.publish(Float64(data=gamma))
        self.solve_time_pub.publish(Float64(data=solve_ms))

        phi = np.linalg.norm(self.state.x2 - self.state.x1) - self.params.cable_length

        self.get_logger().info(
            f"t={self.sim_time:.2f} solve_ms={solve_ms:.3f} "
            f"phi={phi:.3e} gamma={gamma:.3f} s={s:.3e} "
            f"carrier={np.array2string(self.state.x1, precision=3, suppress_small=True)} "
            f"payload={np.array2string(self.state.x2, precision=3, suppress_small=True)}"
        )

    def _make_odom(self, stamp, frame_id: str, child_frame_id: str, pos: Vec, vel: Vec) -> Odometry:
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.child_frame_id = child_frame_id

        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.w = 1.0

        msg.twist.twist.linear.x = float(vel[0])
        msg.twist.twist.linear.y = float(vel[1])
        msg.twist.twist.linear.z = float(vel[2])
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InnerSoftplusPendulumNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
