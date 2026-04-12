#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import casadi as ca
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


Vec = np.ndarray


# ============================================================
# Parameters and state
# ============================================================

@dataclass
class Params:
    m1: float = 1.0
    m2: float = 0.2
    g: float = 9.81
    cable_length: float = 1.0
    dt: float = 0.03
    mu: float = 1.0e-3
    damping1: float = 0.1
    damping2: float = 0.1

    inner_iters: int = 80
    residual_tol: float = 1.0e-9
    linear_reg: float = 1.0e-10

    # soft collision-avoidance threshold used in stage cost
    min_distance: float = 0.1


@dataclass
class ILQRConfig:
    N: int = 30
    max_iters: int = 20
    reg_min: float = 1.0e-6
    reg_max: float = 1.0e6
    reg_init: float = 1.0
    alpha_list: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)
    u_min: tuple[float, float, float] = (-40.0, -40.0, -40.0)
    u_max: tuple[float, float, float] = (40.0, 40.0, 40.0)


@dataclass
class Weights:
    Q_payload: np.ndarray
    Q_carrier_vel: np.ndarray
    Q_payload_vel: np.ndarray
    R_u: np.ndarray
    Qf_payload: np.ndarray
    w_collision: float = 100.0
    collision_eps: float = 1.0e-3


@dataclass
class State:
    x1: Vec
    x2: Vec
    v1: Vec
    v2: Vec

    def as_vector(self) -> np.ndarray:
        return np.concatenate((self.x1, self.x2, self.v1, self.v2)).astype(float)

    @staticmethod
    def from_vector(x: np.ndarray) -> "State":
        x = np.asarray(x).reshape(-1)
        return State(
            x1=x[0:3].copy(),
            x2=x[3:6].copy(),
            v1=x[6:9].copy(),
            v2=x[9:12].copy(),
        )


PAYLOAD_TARGET = np.array([1.5, 0.0, 1.5], dtype=float)
INITIAL_CARRIER = np.array([0.0, 0.0, 0.0], dtype=float)
INITIAL_PAYLOAD = np.array([0.0, 0.0, -1.0], dtype=float)


# ============================================================
# Utilities
# ============================================================

def nominal_control(params: Params) -> np.ndarray:
    return np.array([0.0, 0.0, (params.m1 + params.m2) * params.g], dtype=float)


def cable_gap(x1: np.ndarray, x2: np.ndarray, cable_length: float) -> float:
    return float(np.linalg.norm(x2 - x1) - cable_length)


def carrier_payload_distance(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(12)
    return float(np.linalg.norm(x[3:6] - x[0:3]))


def clamp_control(u: np.ndarray, u_min: np.ndarray, u_max: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(u, u_min), u_max)


# ============================================================
# Softplus constrained dynamics with explicit IFT derivatives
# ============================================================

class SoftplusDynamicsIFT:
    """
    Implicit constrained dynamics:
        F(z, x, u) = 0
    with
        z = [v_next(6), w]

    Then:
        x_{k+1} = f(x_k, u_k)
    is built from z^*.

    Explicit backward pass:
        dz/dx = -Fz^{-1} Fx
        dz/du = -Fz^{-1} Fu

    and
        A = df/dx
        B = df/du
    """

    def __init__(self, params: Params):
        self.p = params
        self.M = np.diag([params.m1] * 3 + [params.m2] * 3)
        self.M_inv = np.diag([1.0 / params.m1] * 3 + [1.0 / params.m2] * 3)
        self._build_symbolics()

    @staticmethod
    def bmu_sym(w, mu):
        return 0.5 * (w + ca.sqrt(w * w + 4.0 * mu))

    def _build_symbolics(self) -> None:
        p = self.p

        # state x = [x1,x2,v1,v2]
        x = ca.MX.sym("x", 12)
        x1 = x[0:3]
        x2 = x[3:6]
        v1 = x[6:9]
        v2 = x[9:12]

        # control
        u = ca.MX.sym("u", 3)

        # implicit variable z = [vnext(6), w]
        z = ca.MX.sym("z", 7)
        vnext = z[0:6]
        w = z[6]

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

        M = ca.DM(self.M)
        M_inv = ca.DM(self.M_inv)

        vcurr = ca.vertcat(v1, v2)
        vfree = vcurr + p.dt * (M_inv @ tau)

        gamma = self.bmu_sym(w, p.mu)
        s = self.bmu_sym(-w, p.mu)

        # implicit equations
        F1 = M @ (vnext - vfree) + p.dt * J * gamma
        F2 = s + phi + p.dt * ca.dot(J, vnext)
        F = ca.vertcat(F1, F2)

        Fz = ca.jacobian(F, z)
        Fx = ca.jacobian(F, x)
        Fu = ca.jacobian(F, u)

        # explicit next state map using z*
        x1n = x1 + p.dt * vnext[0:3]
        x2n = x2 + p.dt * vnext[3:6]
        xnext = ca.vertcat(x1n, x2n, vnext)

        self.F_fun = ca.Function("F_fun", [z, x, u], [F])
        self.Fz_fun = ca.Function("Fz_fun", [z, x, u], [Fz])
        self.Fx_fun = ca.Function("Fx_fun", [z, x, u], [Fx])
        self.Fu_fun = ca.Function("Fu_fun", [z, x, u], [Fu])
        self.xnext_fun = ca.Function("xnext_fun", [z, x], [xnext])

        # helper functions need fresh symbolic inputs
        w_aux = ca.MX.sym("w_aux", 1)
        gamma_aux = self.bmu_sym(w_aux[0], p.mu)
        s_aux = self.bmu_sym(-w_aux[0], p.mu)
        self.gamma_fun = ca.Function("gamma_fun", [w_aux], [gamma_aux])
        self.s_fun = ca.Function("s_fun", [w_aux], [s_aux])

    def residual(self, z: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array(self.F_fun(z, x, u)).astype(float).reshape(-1)

    def jacobian_z(self, z: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array(self.Fz_fun(z, x, u)).astype(float)

    def jacobian_x(self, z: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array(self.Fx_fun(z, x, u)).astype(float)

    def jacobian_u(self, z: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array(self.Fu_fun(z, x, u)).astype(float)

    def gamma_from_w(self, w: float) -> float:
        return float(np.array(self.gamma_fun(np.array([w], dtype=float))).reshape(()))

    def s_from_w(self, w: float) -> float:
        return float(np.array(self.s_fun(np.array([w], dtype=float))).reshape(()))

    def solve_z(self, x: np.ndarray, u: np.ndarray, z_guess: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.p
        x = np.asarray(x).reshape(12)
        u = np.asarray(u).reshape(3)

        if z_guess is None:
            z = np.zeros(7, dtype=float)
            z[:6] = x[6:12]
            z[6] = 0.0
        else:
            z = np.asarray(z_guess).reshape(7).copy()

        for _ in range(p.inner_iters):
            F = self.residual(z, x, u)
            if np.linalg.norm(F, ord=np.inf) < p.residual_tol:
                break

            J = self.jacobian_z(z, x, u)
            dz = -np.linalg.solve(J + p.linear_reg * np.eye(7), F)

            alpha = 1.0
            merit0 = 0.5 * np.dot(F, F)
            accepted = False

            while alpha > 1.0e-6:
                z_try = z + alpha * dz
                F_try = self.residual(z_try, x, u)
                merit_try = 0.5 * np.dot(F_try, F_try)
                if merit_try < merit0:
                    z = z_try
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                break

        return z

    def step(self, x: np.ndarray, u: np.ndarray, z_guess: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        z = self.solve_z(x, u, z_guess=z_guess)
        xnext = np.array(self.xnext_fun(z, x)).astype(float).reshape(12)
        return xnext, z

    def linearize(self, x: np.ndarray, u: np.ndarray, z: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            A = dx_next/dx  in R^{12x12}
            B = dx_next/du  in R^{12x3}
            z = solved implicit variable
        """
        p = self.p
        x = np.asarray(x).reshape(12)
        u = np.asarray(u).reshape(3)

        if z is None:
            z = self.solve_z(x, u)

        Fz = self.jacobian_z(z, x, u)
        Fx = self.jacobian_x(z, x, u)
        Fu = self.jacobian_u(z, x, u)

        # IFT
        regI = p.linear_reg * np.eye(7)
        dzdx = -np.linalg.solve(Fz + regI, Fx)   # 7x12
        dzdu = -np.linalg.solve(Fz + regI, Fu)   # 7x3

        dvdx = dzdx[:6, :]
        dvdu = dzdu[:6, :]

        A = np.zeros((12, 12), dtype=float)
        B = np.zeros((12, 3), dtype=float)

        # x1_{k+1} = x1 + dt * v1_{k+1}
        A[0:3, 0:3] = np.eye(3)
        A[0:3, :] += p.dt * dvdx[0:3, :]

        # x2_{k+1} = x2 + dt * v2_{k+1}
        A[3:6, 3:6] = np.eye(3)
        A[3:6, :] += p.dt * dvdx[3:6, :]

        # v_{k+1}
        A[6:12, :] = dvdx

        # control Jacobian
        B[0:3, :] = p.dt * dvdu[0:3, :]
        B[3:6, :] = p.dt * dvdu[3:6, :]
        B[6:12, :] = dvdu

        return A, B, z


# ============================================================
# iLQR cost model with collision-avoidance barrier
# ============================================================

class ILQRCost:
    def __init__(self, weights: Weights, payload_target: np.ndarray, min_distance: float):
        self.W = weights
        self.payload_target = np.asarray(payload_target).reshape(3)
        self.min_distance = float(min_distance)
        self._build_symbolics()

    def _build_symbolics(self) -> None:
        x = ca.MX.sym("x", 12)
        u = ca.MX.sym("u", 3)

        x1 = x[0:3]
        x2 = x[3:6]
        v1 = x[6:9]
        v2 = x[9:12]
        ep = x2 - ca.DM(self.payload_target)

        stage = (
            ca.mtimes([ep.T, ca.DM(self.W.Q_payload), ep])
            + ca.mtimes([v1.T, ca.DM(self.W.Q_carrier_vel), v1])
            + ca.mtimes([v2.T, ca.DM(self.W.Q_payload_vel), v2])
            + ca.mtimes([u.T, ca.DM(self.W.R_u), u])
        )

        terminal = ca.mtimes([ep.T, ca.DM(self.W.Qf_payload), ep])

        lx = ca.gradient(stage, x)
        lu = ca.gradient(stage, u)
        lxx = ca.hessian(stage, x)[0]
        luu = ca.hessian(stage, u)[0]
        lux = ca.jacobian(lu, x)

        terminal_lx = ca.gradient(terminal, x)
        terminal_lxx = ca.hessian(terminal, x)[0]

        self.stage_cost_fun = ca.Function("stage_cost_fun", [x, u], [stage])
        self.terminal_cost_fun = ca.Function("terminal_cost_fun", [x], [terminal])
        self.stage_derivatives_fun = ca.Function(
            "stage_derivatives_fun",
            [x, u],
            [lx, lu, lxx, luu, lux],
        )
        self.terminal_derivatives_fun = ca.Function(
            "terminal_derivatives_fun",
            [x],
            [terminal_lx, terminal_lxx],
        )

    def stage_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        x = np.asarray(x).reshape(12)
        u = np.asarray(u).reshape(3)
        return float(np.array(self.stage_cost_fun(x, u)).reshape(()))

    def terminal_cost(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(12)
        return float(np.array(self.terminal_cost_fun(x)).reshape(()))

    def total_cost(self, X: list[np.ndarray], U: list[np.ndarray]) -> float:
        cost = 0.0
        for k in range(len(U)):
            cost += self.stage_cost(X[k], U[k])
        cost += self.terminal_cost(X[-1])
        return cost

    def stage_derivatives_fd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        eps_x: float = 1.0e-5,
        eps_u: float = 1.0e-5,
    ):
        x = np.asarray(x).reshape(12)
        u = np.asarray(u).reshape(3)
        lx, lu, lxx, luu, lux = self.stage_derivatives_fun(x, u)
        return (
            np.array(lx).astype(float).reshape(12),
            np.array(lu).astype(float).reshape(3),
            np.array(lxx).astype(float),
            np.array(luu).astype(float),
            np.array(lux).astype(float),
        )

    def terminal_derivatives_fd(self, x: np.ndarray, eps_x: float = 1.0e-5):
        x = np.asarray(x).reshape(12)
        lx, lxx = self.terminal_derivatives_fun(x)
        return (
            np.array(lx).astype(float).reshape(12),
            np.array(lxx).astype(float),
        )


# ============================================================
# iLQR using explicit IFT backward pass
# ============================================================

class IFTiLQRController:
    def __init__(self, dyn: SoftplusDynamicsIFT, cfg: ILQRConfig, cost: ILQRCost):
        self.dyn = dyn
        self.cfg = cfg
        self.cost = cost

        self.u_min = np.asarray(cfg.u_min, dtype=float)
        self.u_max = np.asarray(cfg.u_max, dtype=float)

        self.u_seq = [np.zeros(3, dtype=float) for _ in range(cfg.N)]
        self.reg = cfg.reg_init

    def shift_warmstart(self):
        self.u_seq = self.u_seq[1:] + [self.u_seq[-1].copy()]

    def rollout(self, x0: np.ndarray, u_seq: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], float]:
        X = [np.asarray(x0).reshape(12).copy()]
        Z = []
        A_list = []
        B_list = []

        z_guess = None
        for k in range(self.cfg.N):
            xk = X[-1]
            uk = clamp_control(u_seq[k], self.u_min, self.u_max)
            xkp1, z = self.dyn.step(xk, uk, z_guess=z_guess)
            A, B, _ = self.dyn.linearize(xk, uk, z=z)

            X.append(xkp1)
            Z.append(z)
            A_list.append(A)
            B_list.append(B)
            z_guess = z

        cost = self.cost.total_cost(X, [clamp_control(u, self.u_min, self.u_max) for u in u_seq])
        return X, Z, A_list, B_list, cost

    def backward_pass(self, X: list[np.ndarray], U: list[np.ndarray], A_list: list[np.ndarray], B_list: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        N = self.cfg.N
        nx = 12
        nu = 3

        K_list = [np.zeros((nu, nx)) for _ in range(N)]
        d_list = [np.zeros(nu) for _ in range(N)]

        Vx, Vxx = self.cost.terminal_derivatives_fd(X[-1])

        regI = self.reg * np.eye(nu)
        success = True

        for k in reversed(range(N)):
            xk = X[k]
            uk = clamp_control(U[k], self.u_min, self.u_max)
            A = A_list[k]
            B = B_list[k]

            lx, lu, lxx, luu, lux = self.cost.stage_derivatives_fd(xk, uk)

            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            Quu_reg = 0.5 * (Quu + Quu.T) + regI

            try:
                L = np.linalg.cholesky(Quu_reg)
                inv_Quu = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(nu)))
            except np.linalg.LinAlgError:
                success = False
                break

            dk = -inv_Quu @ Qu
            Kk = -inv_Quu @ Qux

            d_list[k] = dk
            K_list[k] = Kk

            Vx = Qx + Kk.T @ Quu @ dk + Kk.T @ Qu + Qux.T @ dk
            Vxx = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qux.T @ Kk
            Vxx = 0.5 * (Vxx + Vxx.T)

        return K_list, d_list, success

    def forward_pass(
        self,
        x0: np.ndarray,
        X_nom: list[np.ndarray],
        U_nom: list[np.ndarray],
        K_list: list[np.ndarray],
        d_list: list[np.ndarray],
        alpha: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray], float]:
        X_new = [np.asarray(x0).reshape(12).copy()]
        U_new = []
        z_guess = None

        for k in range(self.cfg.N):
            dx = X_new[-1] - X_nom[k]
            uk = U_nom[k] + alpha * d_list[k] + K_list[k] @ dx
            uk = clamp_control(uk, self.u_min, self.u_max)
            U_new.append(uk)

            xkp1, z = self.dyn.step(X_new[-1], uk, z_guess=z_guess)
            X_new.append(xkp1)
            z_guess = z

        cost = self.cost.total_cost(X_new, U_new)
        return X_new, U_new, cost

    def solve(self, x0: np.ndarray) -> dict:
        x0 = np.asarray(x0).reshape(12)
        U = [u.copy() for u in self.u_seq]

        X, Z, A_list, B_list, J = self.rollout(x0, U)

        for _ in range(self.cfg.max_iters):
            K_list, d_list, success = self.backward_pass(X, U, A_list, B_list)
            if not success:
                self.reg = min(self.reg * 10.0, self.cfg.reg_max)
                continue

            accepted = False
            best_X = X
            best_U = U
            best_J = J

            for alpha in self.cfg.alpha_list:
                X_try, U_try, J_try = self.forward_pass(x0, X, U, K_list, d_list, alpha)
                if J_try < best_J:
                    best_X = X_try
                    best_U = U_try
                    best_J = J_try
                    accepted = True
                    break

            if accepted:
                X = best_X
                U = best_U
                J = best_J
                self.reg = max(self.reg / 5.0, self.cfg.reg_min)

                # refresh linearization around accepted rollout
                _, _, A_list, B_list, _ = self.rollout(x0, U)
            else:
                self.reg = min(self.reg * 10.0, self.cfg.reg_max)

            if self.reg >= self.cfg.reg_max:
                break

        self.u_seq = [u.copy() for u in U]

        return {
            "X": X,
            "U": U,
            "cost": J,
        }


# ============================================================
# ROS 2 node
# ============================================================

class PayloadIFTNMPCNode(Node):
    def __init__(self) -> None:
        super().__init__("payload_ift_nmpc_node")

        self.params = Params()
        self.ilqr_cfg = ILQRConfig(N=30)

        self.weights = Weights(
            Q_payload=np.diag([40.0, 40.0, 80.0]),
            Q_carrier_vel=np.diag([1.0, 1.0, 1.0]),
            Q_payload_vel=np.diag([3.0, 3.0, 3.0]),
            R_u=np.diag([1.0e-3, 1.0e-3, 1.0e-3]),
            Qf_payload=np.diag([120.0, 120.0, 180.0]),
            w_collision=100.0,
            collision_eps=1.0e-3,
        )

        self.dyn = SoftplusDynamicsIFT(self.params)
        self.cost = ILQRCost(self.weights, PAYLOAD_TARGET, self.params.min_distance)
        self.ctrl = IFTiLQRController(self.dyn, self.ilqr_cfg, self.cost)

        self.state = State(
            x1=INITIAL_CARRIER.copy(),
            x2=INITIAL_PAYLOAD.copy(),
            v1=np.zeros(3, dtype=float),
            v2=np.zeros(3, dtype=float),
        )

        self.sim_time = 0.0
        self.z_guess = None

        self.carrier_pub = self.create_publisher(Odometry, "/carrier/odom", 10)
        self.payload_pub = self.create_publisher(Odometry, "/payload/odom", 10)
        self.tension_pub = self.create_publisher(Float64, "/cable/tension", 10)
        self.solve_time_pub = self.create_publisher(Float64, "/ift_nmpc/solve_time_ms", 10)
        self.cost_pub = self.create_publisher(Float64, "/ift_nmpc/cost", 10)
        self.phi_pub = self.create_publisher(Float64, "/ift_nmpc/phi", 10)
        self.distance_pub = self.create_publisher(Float64, "/ift_nmpc/carrier_payload_distance", 10)

        self.timer = self.create_timer(self.params.dt, self.step)

        self.get_logger().info("IFT-based NMPC node started")

    def step(self) -> None:
        x0 = self.state.as_vector()

        t0 = time.perf_counter()
        sol = self.ctrl.solve(x0)
        solve_ms = 1.0e3 * (time.perf_counter() - t0)

        u0 = sol["U"][0]
        self.ctrl.shift_warmstart()

        x_next, self.z_guess = self.dyn.step(x0, u0, z_guess=self.z_guess)
        gamma = self.dyn.gamma_from_w(self.z_guess[6])

        self.state = State.from_vector(x_next)
        self.sim_time += self.params.dt

        phi = cable_gap(self.state.x1, self.state.x2, self.params.cable_length)
        dist_cp = carrier_payload_distance(self.state.as_vector())

        now = self.get_clock().now().to_msg()
        self.carrier_pub.publish(self._make_odom(now, "world", "carrier", self.state.x1, self.state.v1))
        self.payload_pub.publish(self._make_odom(now, "world", "payload", self.state.x2, self.state.v2))
        self.tension_pub.publish(Float64(data=gamma))
        self.solve_time_pub.publish(Float64(data=solve_ms))
        self.cost_pub.publish(Float64(data=float(sol["cost"])))
        self.phi_pub.publish(Float64(data=phi))
        self.distance_pub.publish(Float64(data=dist_cp))

        self.get_logger().info(
            f"t={self.sim_time:.2f} "
            f"solve_ms={solve_ms:.2f} "
            f"cost={sol['cost']:.3f} "
            f"phi={phi:.3e} "
            f"dist={dist_cp:.3f} "
            f"u={np.array2string(u0, precision=3, suppress_small=True)} "
            f"payload={np.array2string(self.state.x2, precision=3, suppress_small=True)} "
            f"gamma={gamma:.3e}"
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
    node = PayloadIFTNMPCNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
