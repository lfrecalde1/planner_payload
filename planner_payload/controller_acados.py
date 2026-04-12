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

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


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
    damping2: float = 0.05

    # numerical parameters for plant solve only
    inner_iters: int = 50
    residual_tol: float = 1.0e-9
    linear_reg: float = 1.0e-10
    disc_newton_iters: int = 8

    # collision-avoidance distance
    min_distance: float = 0.6


@dataclass
class MPCConfig:
    N: int = 40
    u_min: tuple[float, float, float] = (-20.0, -20.0, -20.0)
    u_max: tuple[float, float, float] = (20.0, 20.0, 20.0)
    qp_cond_N: int = 10


@dataclass
class Weights:
    Q_payload: np.ndarray
    Q_carrier_vel: np.ndarray
    Q_payload_vel: np.ndarray
    R_u: np.ndarray
    Qf_payload: np.ndarray


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
INITIAL_PAYLOAD = np.array([0.0, 0.0, -0.99], dtype=float)


# ============================================================
# Utilities
# ============================================================

def nominal_control(params: Params) -> np.ndarray:
    return np.array([0.0, 0.0, (params.m1 + params.m2) * params.g], dtype=float)


def cable_gap(x1: np.ndarray, x2: np.ndarray, cable_length: float) -> float:
    return float(np.linalg.norm(x2 - x1) - cable_length)


def pair_distance_num(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(12)
    return float(np.linalg.norm(x[3:6] - x[0:3]))


# ============================================================
# acados model with IFT Jacobians
# ============================================================

class SoftplusDynamicsAcadosIFT:
    """
    acados DISCRETE model:
        x_{k+1} = f_disc(x_k, u_k)

    Internal stage equations:
        F(z, x, u) = 0,  z = [v_next(6), w]

    Jacobians use the implicit function theorem:
        dz/dx = -Fz^{-1} Fx
        dz/du = -Fz^{-1} Fu

    Then:
        A = df_disc/dx
        B = df_disc/du

    are passed via:
        model.disc_dyn_custom_jac_ux_expr
    """

    def __init__(self, params: Params, weights: Weights):
        self.p = params
        self.W = weights
        self.M = np.diag([params.m1] * 3 + [params.m2] * 3)
        self.M_inv = np.diag([1.0 / params.m1] * 3 + [1.0 / params.m2] * 3)

        self.model = self._build_model()
        self._build_numeric_helpers()

    @staticmethod
    def bmu_sym(w, mu):
        return 0.5 * (w + ca.sqrt(w * w + 4.0 * mu))

    def _stage_symbolics(self):
        p = self.p

        x = ca.MX.sym("x", 12)
        u = ca.MX.sym("u", 3)
        z = ca.MX.sym("z", 7)  # [v_next(6), w]

        x1 = x[0:3]
        x2 = x[3:6]
        v1 = x[6:9]
        v2 = x[9:12]

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

        F1 = M @ (vnext - vfree) + p.dt * J * gamma
        F2 = s + phi + p.dt * ca.dot(J, vnext)
        F = ca.vertcat(F1, F2)

        Fz = ca.jacobian(F, z)
        Fx = ca.jacobian(F, x)
        Fu = ca.jacobian(F, u)

        return x, u, z, F, Fz, Fx, Fu

    def _build_model(self) -> AcadosModel:
        p = self.p
        W = self.W

        x, u, z, F, Fz, Fx, Fu = self._stage_symbolics()

        # Natural initial guess z0 = [current velocity, 0]
        z_iter = ca.vertcat(x[6:12], ca.DM([0.0]))
        regI7 = p.linear_reg * ca.DM.eye(7)

        # Fixed symbolic Newton iterations to approximate the implicit solution z*(x, u).
        # The IFT linearization is evaluated at the final iterate instead of after one step.
        for _ in range(p.disc_newton_iters):
            F_iter = ca.substitute(F, z, z_iter)
            Fz_iter = ca.substitute(Fz, z, z_iter)
            z_iter = z_iter - ca.solve(Fz_iter + regI7, F_iter)

        z_star = z_iter
        vnext = z_star[0:6]

        # Discrete map x_{k+1} = f_disc(x,u)
        x1 = x[0:3]
        x2 = x[3:6]
        xnext = ca.vertcat(
            x1 + p.dt * vnext[0:3],
            x2 + p.dt * vnext[3:6],
            vnext,
        )

        # IFT Jacobians evaluated at the final Newton iterate.
        Fz_star = ca.substitute(Fz, z, z_star)
        Fx_star = ca.substitute(Fx, z, z_star)
        Fu_star = ca.substitute(Fu, z, z_star)

        dzdx = -ca.solve(Fz_star + regI7, Fx_star)  # 7x12
        dzdu = -ca.solve(Fz_star + regI7, Fu_star)  # 7x3

        dvdx = dzdx[0:6, :]
        dvdu = dzdu[0:6, :]

        A = ca.MX.zeros(12, 12)
        B = ca.MX.zeros(12, 3)

        A[0:3, 0:3] = ca.DM.eye(3)
        A[0:3, :] += p.dt * dvdx[0:3, :]

        A[3:6, 3:6] = ca.DM.eye(3)
        A[3:6, :] += p.dt * dvdx[3:6, :]

        A[6:12, :] = dvdx

        B[0:3, :] = p.dt * dvdu[0:3, :]
        B[3:6, :] = p.dt * dvdu[3:6, :]
        B[6:12, :] = dvdu

        # acados expects Jacobian wrt [u, x]
        jac_ux = ca.horzcat(B, A)  # 12 x 15

        # Practical first approximation for discrete dynamics Hessian
        hess_ux = ca.DM.zeros(15, 15)

        # Nonlinear distance constraint
        dcp = x[3:6] - x[0:3]
        dist = ca.sqrt(ca.dot(dcp, dcp) + 1.0e-12)

        # External cost
        ep = x[3:6] - PAYLOAD_TARGET
        v1 = x[6:9]
        v2 = x[9:12]

        Qp = ca.DM(W.Q_payload)
        Qcv = ca.DM(W.Q_carrier_vel)
        Qpv = ca.DM(W.Q_payload_vel)
        Ru = ca.DM(W.R_u)
        Qf = ca.DM(W.Qf_payload)

        model = AcadosModel()
        model.name = "quadrotor_cable_softplus_disc_ift"
        model.x = x
        model.u = u

        model.disc_dyn_expr = xnext
        model.disc_dyn_custom_jac_ux_expr = jac_ux
        model.disc_dyn_custom_hess_ux_expr = hess_ux

        model.con_h_expr = dist

        model.cost_expr_ext_cost = (
            ca.mtimes([ep.T, Qp, ep])
            + ca.mtimes([v1.T, Qcv, v1])
            + ca.mtimes([v2.T, Qpv, v2])
            + ca.mtimes([u.T, Ru, u])
        )
        model.cost_expr_ext_cost_e = ca.mtimes([ep.T, Qf, ep])

        return model

    def _build_numeric_helpers(self):
        p = self.p

        x, u, z, F, Fz, Fx, Fu = self._stage_symbolics()

        self.F_num_fun = ca.Function("F_num_fun", [z, x, u], [F])
        self.J_num_fun = ca.Function("J_num_fun", [z, x, u], [Fz])

        w_aux = ca.MX.sym("w_aux", 1)
        gamma_aux = self.bmu_sym(w_aux[0], p.mu)
        self.gamma_fun = ca.Function("gamma_fun", [w_aux], [gamma_aux])

    def plant_residual(self, z: np.ndarray, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        return np.array(self.F_num_fun(z, xk, uk)).astype(float).reshape(-1)

    def plant_jacobian(self, z: np.ndarray, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        return np.array(self.J_num_fun(z, xk, uk)).astype(float)

    def plant_step(self, xk: np.ndarray, uk: np.ndarray, z_guess: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        xk = np.asarray(xk).reshape(12)
        uk = np.asarray(uk).reshape(3)

        if z_guess is None:
            z = np.zeros(7, dtype=float)
            z[:6] = xk[6:12]
            z[6] = 0.0
        else:
            z = np.asarray(z_guess).reshape(7).copy()

        for _ in range(p.inner_iters):
            F = self.plant_residual(z, xk, uk)
            if np.linalg.norm(F, ord=np.inf) < p.residual_tol:
                break

            J = self.plant_jacobian(z, xk, uk)
            dz = -np.linalg.solve(J + p.linear_reg * np.eye(7), F)

            merit0 = 0.5 * np.dot(F, F)
            alpha = 1.0
            accepted = False
            while alpha > 1.0e-6:
                z_try = z + alpha * dz
                F_try = self.plant_residual(z_try, xk, uk)
                merit_try = 0.5 * np.dot(F_try, F_try)
                if merit_try < merit0:
                    z = z_try
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                break

        vnext = z[:6]
        x1 = xk[0:3]
        x2 = xk[3:6]
        xnext = np.concatenate((
            x1 + p.dt * vnext[0:3],
            x2 + p.dt * vnext[3:6],
            vnext,
        ))
        return xnext, z

    def gamma_from_w(self, w: float) -> float:
        return float(np.array(self.gamma_fun(np.array([w], dtype=float))).reshape(()))


# ============================================================
# acados OCP
# ============================================================

class AcadosPayloadNMPC:
    def __init__(self, params: Params, cfg: MPCConfig, weights: Weights):
        self.params = params
        self.cfg = cfg
        self.weights = weights

        self.model_builder = SoftplusDynamicsAcadosIFT(params, weights)
        self.model = self.model_builder.model

        self.ocp = self._build_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp_payload_ift.json")

        self._initialize_warm_start()

    def _build_ocp(self) -> AcadosOcp:
        p = self.params
        cfg = self.cfg
        model = self.model

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = cfg.N

        ocp.solver_options.tf = cfg.N * p.dt
        ocp.solver_options.integrator_type = "DISCRETE"

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.constraints.x0 = np.zeros(12)

        ocp.constraints.idxbu = np.array([0, 1, 2], dtype=int)
        ocp.constraints.lbu = np.array(cfg.u_min, dtype=float)
        ocp.constraints.ubu = np.array(cfg.u_max, dtype=float)

        # dist >= min_distance
        ocp.constraints.lh = np.array([p.min_distance], dtype=float)
        ocp.constraints.uh = np.array([1.0e9], dtype=float)

        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = cfg.qp_cond_N
        # The discrete dynamics now use a multi-step symbolic Newton solve.
        # A more robust SQP setup helps when the resulting linearization is stiff.
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.regularize_method = "PROJECT"
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.nlp_solver_max_iter = 50

        return ocp

    def _initialize_warm_start(self):
        N = self.cfg.N
        for k in range(N + 1):
            self.solver.set(k, "x", np.zeros(12))
        for k in range(N):
            self.solver.set(k, "u", np.zeros(3))

    def set_x0(self, x0: np.ndarray):
        x0 = np.asarray(x0).reshape(12)
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)

    def solve(self, x0: np.ndarray) -> dict:
        self.set_x0(x0)

        status = self.solver.solve()
        if status != 0:
            raise RuntimeError(f"acados solver returned status {status}")

        X = []
        U = []
        for k in range(self.cfg.N):
            X.append(self.solver.get(k, "x").copy())
            U.append(self.solver.get(k, "u").copy())
        X.append(self.solver.get(self.cfg.N, "x").copy())

        return {
            "status": status,
            "X": np.array(X),
            "U": np.array(U),
        }

    def shift_warm_start(self, sol: dict):
        X = sol["X"]
        U = sol["U"]
        N = self.cfg.N

        for k in range(N - 1):
            self.solver.set(k, "x", X[k + 1])
            self.solver.set(k, "u", U[k + 1])

        self.solver.set(N - 1, "u", U[-1])
        self.solver.set(N, "x", X[-1])


# ============================================================
# ROS 2 node
# ============================================================

class PayloadAcadosNMPCNode(Node):
    def __init__(self) -> None:
        super().__init__("payload_acados_nmpc_node")

        self.params = Params()
        self.mpc_cfg = MPCConfig(N=40)

        self.weights = Weights(
            Q_payload=np.diag([40.0, 40.0, 80.0]),
            Q_carrier_vel=np.diag([2.0, 2.0, 2.0]),
            Q_payload_vel=np.diag([3.0, 3.0, 3.0]),
            R_u=np.diag([5.0e-3, 5.0e-3, 5.0e-3]),
            Qf_payload=np.diag([120.0, 120.0, 180.0]),
        )

        self.controller = AcadosPayloadNMPC(self.params, self.mpc_cfg, self.weights)
        self.plant = self.controller.model_builder

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
        self.solve_time_pub = self.create_publisher(Float64, "/acados_nmpc/solve_time_ms", 10)
        self.phi_pub = self.create_publisher(Float64, "/acados_nmpc/phi", 10)
        self.distance_pub = self.create_publisher(Float64, "/acados_nmpc/carrier_payload_distance", 10)

        self.timer = self.create_timer(self.params.dt, self.step)

        self.get_logger().info("acados NMPC node with IFT Jacobians started")

    def step(self) -> None:
        x0 = self.state.as_vector()

        t0 = time.perf_counter()
        sol = self.controller.solve(x0)
        solve_ms = 1.0e3 * (time.perf_counter() - t0)

        self.controller.shift_warm_start(sol)

        u0 = sol["U"][0]
        x_next, self.z_guess = self.plant.plant_step(x0, u0, z_guess=self.z_guess)
        gamma0 = self.plant.gamma_from_w(self.z_guess[6])

        self.state = State.from_vector(x_next)
        self.sim_time += self.params.dt

        phi = cable_gap(self.state.x1, self.state.x2, self.params.cable_length)
        dist_cp = pair_distance_num(self.state.as_vector())

        now = self.get_clock().now().to_msg()
        self.carrier_pub.publish(self._make_odom(now, "world", "carrier", self.state.x1, self.state.v1))
        self.payload_pub.publish(self._make_odom(now, "world", "payload", self.state.x2, self.state.v2))
        self.tension_pub.publish(Float64(data=gamma0))
        self.solve_time_pub.publish(Float64(data=solve_ms))
        self.phi_pub.publish(Float64(data=phi))
        self.distance_pub.publish(Float64(data=dist_cp))

        self.get_logger().info(
            f"t={self.sim_time:.2f} "
            f"solve_ms={solve_ms:.2f} "
            f"phi={phi:.3e} "
            f"dist={dist_cp:.3f} "
            f"u={np.array2string(u0, precision=3, suppress_small=True)} "
            f"payload={np.array2string(self.state.x2, precision=3, suppress_small=True)} "
            f"gamma={gamma0:.3e}"
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
    node = PayloadAcadosNMPCNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
