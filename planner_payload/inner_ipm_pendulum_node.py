#!/usr/bin/env python3
"""ROS 2 node that publishes the inner-IPM slack-to-taut pendulum simulation.

This node reuses the same dynamics structure as
`toy_inner_ipm_slack_to_pendulum.py`:
- carrier approximately fixed at the initial position
- payload starts inside the cable length
- gravity drives a slack-to-taut transition
- the taut phase behaves like a damped pendulum

Published topics:
- `/carrier/odom`  : nav_msgs/msg/Odometry
- `/payload/odom`  : nav_msgs/msg/Odometry
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

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


class InnerIpmStep:
    def __init__(self, params: Params):
        self.params = params
        self.M = np.diag([params.m1] * 3 + [params.m2] * 3)
        self.M_inv = np.diag([1.0 / params.m1] * 3 + [1.0 / params.m2] * 3)

    def v_free(self, state: State, u: Vec) -> Vec:
        gvec = np.array([0.0, 0.0, -self.params.g], dtype=float)
        tau = np.hstack((
            self.params.m1 * gvec - self.params.damping1 * state.v1 + nominal_control(self.params) + u,
            self.params.m2 * gvec - self.params.damping2 * state.v2,
        ))
        v = np.hstack((state.v1, state.v2))
        return v + self.params.dt * (self.M_inv @ tau)

    def residual(self, z: Vec, state: State, u: Vec) -> Vec:
        v = z[:6]
        s = z[6]
        gamma = z[7]
        vf = self.v_free(state, u)
        phi, n, _ = geometry_terms(state.x1, state.x2, v, self.params)
        J = np.hstack((-n, n))
        F1 = self.M @ (v - vf) + self.params.dt * J * gamma
        F2 = np.array([s + phi + self.params.dt * float(J @ v)], dtype=float)
        F3 = np.array([s * gamma - self.params.mu], dtype=float)
        return np.concatenate((F1, F2, F3))

    def jacobian_z(self, z: Vec, state: State, u: Vec) -> np.ndarray:
        v = z[:6]
        s = z[6]
        gamma = z[7]
        _, n, _ = geometry_terms(state.x1, state.x2, v, self.params)
        J = np.hstack((-n, n))
        Fz = np.zeros((8, 8), dtype=float)
        Fz[:6, :6] = self.M
        Fz[:6, 7] = self.params.dt * J
        Fz[6, :6] = self.params.dt * J
        Fz[6, 6] = 1.0
        Fz[7, 6] = gamma
        Fz[7, 7] = s
        return Fz

    def solve(self, state: State, u: Vec, guess: Vec | None = None) -> Vec:
        vf = self.v_free(state, u)
        phi0, _, _ = geometry_terms(state.x1, state.x2, vf, self.params)
        if guess is None:
            s0 = max(1.0e-5, -phi0 + 1.0e-4)
            z = np.concatenate((vf, np.array([s0, self.params.mu / s0], dtype=float)))
        else:
            z = guess.copy()

        for _ in range(self.params.inner_iters):
            F = self.residual(z, state, u)
            if np.linalg.norm(F, ord=np.inf) < 1.0e-9:
                break
            Fz = self.jacobian_z(z, state, u)
            dz = -np.linalg.solve(Fz, F)

            alpha = 1.0
            while alpha > 1.0e-5:
                z_try = z + alpha * dz
                if z_try[6] <= 0.0 or z_try[7] <= 0.0:
                    alpha *= 0.5
                    continue
                if np.linalg.norm(self.residual(z_try, state, u)) < np.linalg.norm(F):
                    z = z_try
                    break
                alpha *= 0.5
        return z


def fixed_carrier_controller(state: State, params: Params) -> Vec:
    """Controller that keeps the carrier close to the anchor."""

    kp = np.diag([260.0, 260.0, 400.0])
    kd = np.diag([18.0, 18.0, 20.0])
    gravity_offset = np.array([0.0, 0.0, -params.m2 * params.g], dtype=float)
    return gravity_offset + kp @ (CARRIER_DESIRED - state.x1) - kd @ state.v1


class InnerIpmPendulumNode(Node):
    def __init__(self) -> None:
        super().__init__("inner_ipm_pendulum_node")

        self.params = Params()
        self.stepper = InnerIpmStep(self.params)
        self.state = State(
            x1=CARRIER_ANCHOR.copy(),
            x2=np.array([0.2, 0.00, -0.2], dtype=float),
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

        self.get_logger().info("Inner IPM pendulum node started")

    def step(self) -> None:
        u = fixed_carrier_controller(self.state, self.params)
        t0 = time.perf_counter()
        z = self.stepper.solve(self.state, u, self.z_guess)
        solve_ms = 1.0e3 * (time.perf_counter() - t0)
        self.z_guess = z

        v_next = z[:6]
        gamma = float(z[7])
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
                f"t={self.sim_time:.2f} solve_ms={solve_ms:.3f} phi={phi:.3e} gamma={gamma:.3f} "
                f"carrier={np.array2string(self.state.x1, precision=3, suppress_small=True)} "
                f"payload={np.array2string(self.state.x2, precision=3, suppress_small=True)}")

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
    node = InnerIpmPendulumNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
