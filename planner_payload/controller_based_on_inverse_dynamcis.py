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
    damping2: float = 0.05

    inner_iters: int = 80
    residual_tol: float = 1.0e-9
    linear_reg: float = 1.0e-10

    # Collision-avoidance distance
    min_distance: float = 0.6


@dataclass
class MPCConfig:
    N: int = 40
    u_min: tuple[float, float, float] = (-20.0, -20.0, -20.0)
    u_max: tuple[float, float, float] = (20.0, 20.0, 20.0)
    ipopt_max_iter: int = 120
    ipopt_tol: float = 1.0e-5
    ipopt_acceptable_tol: float = 1.0e-4


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
    x1 = x[0:3]
    x2 = x[3:6]
    return float(np.linalg.norm(x2 - x1))


def pair_distance_sym(x: ca.MX) -> ca.MX:
    x1 = x[0:3]
    x2 = x[3:6]
    d = x2 - x1
    return ca.sqrt(ca.dot(d, d) + 1.0e-12)


# ============================================================
# Softplus constrained dynamics
# ============================================================

class SoftplusDynamics:
    """
    State x = [x1(3), x2(3), v1(3), v2(3)] in R^12
    Control u = carrier force in R^3
    Algebraic variable w in R

    Multiple-shooting defect constraints:
        M (v_{k+1} - v_free(x_k,u_k)) + dt * J(x_k) * gamma(w_k) = 0
        s(w_k) + phi(x_k) + dt * J(x_k)^T v_{k+1} = 0
        x1_{k+1} - (x1_k + dt*v1_{k+1}) = 0
        x2_{k+1} - (x2_k + dt*v2_{k+1}) = 0
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

        xk = ca.MX.sym("xk", 12)
        xkp1 = ca.MX.sym("xkp1", 12)
        uk = ca.MX.sym("uk", 3)
        wk = ca.MX.sym("wk", 1)

        x1 = xk[0:3]
        x2 = xk[3:6]
        v1 = xk[6:9]
        v2 = xk[9:12]

        x1n = xkp1[0:3]
        x2n = xkp1[3:6]
        v1n = xkp1[6:9]
        v2n = xkp1[9:12]
        vnext = ca.vertcat(v1n, v2n)

        d = x2 - x1
        dist = ca.sqrt(ca.dot(d, d) + 1.0e-12)
        n = d / dist
        phi = dist - p.cable_length
        J = ca.vertcat(-n, n)

        gvec = ca.DM([0.0, 0.0, -p.g])
        tau = ca.vertcat(
            p.m1 * gvec - p.damping1 * v1 + ca.DM(nominal_control(p)) + uk,
            p.m2 * gvec - p.damping2 * v2,
        )

        M = ca.DM(self.M)
        M_inv = ca.DM(self.M_inv)

        vcurr = ca.vertcat(v1, v2)
        vfree = vcurr + p.dt * (M_inv @ tau)

        gamma = self.bmu_sym(wk[0], p.mu)
        s = self.bmu_sym(-wk[0], p.mu)

        F1 = M @ (vnext - vfree) + p.dt * J * gamma
        F2 = s + phi + p.dt * ca.dot(J, vnext)
        F3 = x1n - (x1 + p.dt * v1n)
        F4 = x2n - (x2 + p.dt * v2n)

        g_step = ca.vertcat(F1, F2, F3, F4)

        self.g_step_fun = ca.Function("g_step_fun", [xk, xkp1, uk, wk], [g_step])

        # Plant one-step residual in unknown z = [v_next, w]
        z = ca.MX.sym("z", 7)
        v = z[0:6]
        w = z[6]

        gamma_z = self.bmu_sym(w, p.mu)
        s_z = self.bmu_sym(-w, p.mu)

        Fz1 = M @ (v - vfree) + p.dt * J * gamma_z
        Fz2 = s_z + phi + p.dt * ca.dot(J, v)
        Fz = ca.vertcat(Fz1, Fz2)

        self.Fz_fun = ca.Function("Fz_fun", [z, xk, uk], [Fz])
        self.Jz_fun = ca.Function("Jz_fun", [z, xk, uk], [ca.jacobian(Fz, z)])

        self.gamma_fun = ca.Function("gamma_fun", [wk], [gamma])
        self.s_fun = ca.Function("s_fun", [wk], [s])

    def plant_residual(self, z: np.ndarray, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        return np.array(self.Fz_fun(z, xk, uk)).astype(float).reshape(-1)

    def plant_jacobian(self, z: np.ndarray, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        return np.array(self.Jz_fun(z, xk, uk)).astype(float)

    def simulate_one_step(
        self,
        xk: np.ndarray,
        uk: np.ndarray,
        z_guess: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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

        x1n = x1 + p.dt * vnext[0:3]
        x2n = x2 + p.dt * vnext[3:6]

        xkp1 = np.concatenate((x1n, x2n, vnext)).astype(float)
        return xkp1, z

    def gamma_from_w(self, w: float) -> float:
        return float(np.array(self.gamma_fun(np.array([w]))).reshape(()))

    def s_from_w(self, w: float) -> float:
        return float(np.array(self.s_fun(np.array([w]))).reshape(()))


# ============================================================
# Multiple-shooting NMPC with warm start
# ============================================================

class MultipleShootingNMPC:
    def __init__(
        self,
        dyn: SoftplusDynamics,
        cfg: MPCConfig,
        weights: Weights,
        payload_target: np.ndarray,
    ):
        self.dyn = dyn
        self.p = dyn.p
        self.cfg = cfg
        self.W = weights
        self.payload_target = np.asarray(payload_target).reshape(3)

        self.u_min = np.asarray(cfg.u_min, dtype=float)
        self.u_max = np.asarray(cfg.u_max, dtype=float)

        self._build_nlp()

    def _stage_cost(self, xk: ca.MX, uk: ca.MX) -> ca.MX:
        x2 = xk[3:6]
        v1 = xk[6:9]
        v2 = xk[9:12]

        ep = x2 - self.payload_target

        cost = 0
        cost += ca.mtimes([ep.T, ca.DM(self.W.Q_payload), ep])
        cost += ca.mtimes([v1.T, ca.DM(self.W.Q_carrier_vel), v1])
        cost += ca.mtimes([v2.T, ca.DM(self.W.Q_payload_vel), v2])
        cost += ca.mtimes([uk.T, ca.DM(self.W.R_u), uk])
        return cost

    def _terminal_cost(self, xN: ca.MX) -> ca.MX:
        x2 = xN[3:6]
        ep = x2 - self.payload_target
        return ca.mtimes([ep.T, ca.DM(self.W.Qf_payload), ep])

    def _build_nlp(self) -> None:
        N = self.cfg.N

        X = [ca.MX.sym(f"X_{k}", 12) for k in range(N + 1)]
        U = [ca.MX.sym(f"U_{k}", 3) for k in range(N)]
        Walg = [ca.MX.sym(f"W_{k}", 1) for k in range(N)]

        x0_par = ca.MX.sym("x0_par", 12)

        dec = []
        lbw = []
        ubw = []
        w0 = []

        g = []
        lbg = []
        ubg = []

        J = 0

        # X0
        dec += [X[0]]
        lbw += [-ca.inf] * 12
        ubw += [ca.inf] * 12
        w0 += [0.0] * 12

        # Initial state equality
        g += [X[0] - x0_par]
        lbg += [0.0] * 12
        ubg += [0.0] * 12

        # Initial collision-avoidance inequality
        d0 = pair_distance_sym(X[0])
        g += [d0]
        lbg += [self.p.min_distance]
        ubg += [ca.inf]

        for k in range(N):
            # U_k
            dec += [U[k]]
            lbw += self.u_min.tolist()
            ubw += self.u_max.tolist()
            w0 += [0.0, 0.0, 0.0]

            # W_k
            dec += [Walg[k]]
            lbw += [-ca.inf]
            ubw += [ca.inf]
            w0 += [0.0]

            # X_{k+1}
            dec += [X[k + 1]]
            lbw += [-ca.inf] * 12
            ubw += [ca.inf] * 12
            w0 += [0.0] * 12

            # Shooting constraints
            gk = self.dyn.g_step_fun(X[k], X[k + 1], U[k], Walg[k])
            g += [gk]
            lbg += [0.0] * 13
            ubg += [0.0] * 13

            # Collision-avoidance inequality at next node
            dkp1 = pair_distance_sym(X[k + 1])
            g += [dkp1]
            lbg += [self.p.min_distance]
            ubg += [ca.inf]

            # Cost
            J += self._stage_cost(X[k], U[k])

        J += self._terminal_cost(X[N])

        Wvec = ca.vertcat(*dec)
        Gvec = ca.vertcat(*g)

        nlp = {"x": Wvec, "f": J, "g": Gvec, "p": x0_par}

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.cfg.ipopt_max_iter,
            "ipopt.tol": self.cfg.ipopt_tol,
            "ipopt.acceptable_tol": self.cfg.ipopt_acceptable_tol,
            "ipopt.linear_solver": "mumps",
            "calc_lam_x": False,
            "calc_lam_p": False,
        }

        self.solver = ca.nlpsol("nmpc_solver", "ipopt", nlp, opts)

        self.lbw = np.array(lbw, dtype=float)
        self.ubw = np.array(ubw, dtype=float)
        self.lbg = np.array(lbg, dtype=float)
        self.ubg = np.array(ubg, dtype=float)
        self.w0 = np.array(w0, dtype=float)

        self.X = X
        self.U = U
        self.Walg = Walg
        self.Wvec = Wvec

        self._build_indices()

    def _build_indices(self) -> None:
        N = self.cfg.N
        idx = 0

        self.idx_X = [(idx, idx + 12)]
        idx += 12

        self.idx_U = []
        self.idx_W = []

        for _ in range(N):
            self.idx_U.append((idx, idx + 3))
            idx += 3

            self.idx_W.append((idx, idx + 1))
            idx += 1

            self.idx_X.append((idx, idx + 12))
            idx += 12

        self.n_dec = idx

    def solve(self, x0: np.ndarray, warmstart: Optional[np.ndarray] = None) -> dict:
        x0 = np.asarray(x0).reshape(12)
        xinit = self.w0 if warmstart is None else warmstart

        sol = self.solver(
            x0=xinit,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg,
            p=x0,
        )

        w_opt = np.array(sol["x"]).reshape(-1)
        f_opt = float(sol["f"])

        X_sol = np.array([w_opt[a:b] for a, b in self.idx_X])
        U_sol = np.array([w_opt[a:b] for a, b in self.idx_U])
        W_sol = np.array([w_opt[a:b] for a, b in self.idx_W]).reshape(-1)

        return {
            "w_opt": w_opt,
            "f_opt": f_opt,
            "X": X_sol,
            "U": U_sol,
            "W": W_sol,
        }

    def shift_warmstart(self, sol: dict) -> np.ndarray:
        """
        Receding-horizon warm start:
        - shift states one step forward
        - shift controls and algebraic variables
        - repeat last entries at the tail
        """
        X = sol["X"]
        U = sol["U"]
        W = sol["W"]

        N = self.cfg.N

        Xs = np.vstack((X[1:], X[-1]))
        Us = np.vstack((U[1:], U[-1]))
        Ws = np.concatenate((W[1:], W[-1:]))

        warm = []

        # X0
        warm.extend(Xs[0].tolist())

        for k in range(N):
            warm.extend(Us[k].tolist())
            warm.extend([float(Ws[k])])
            warm.extend(Xs[k + 1].tolist())

        return np.array(warm, dtype=float)


# ============================================================
# ROS 2 NMPC node
# ============================================================

class PayloadNMPCNode(Node):
    def __init__(self) -> None:
        super().__init__("payload_nmpc_node")

        self.params = Params()
        self.mpc_cfg = MPCConfig(N=20)

        self.weights = Weights(
            Q_payload=np.diag([40.0, 40.0, 80.0]),
            Q_carrier_vel=np.diag([2.0, 2.0, 2.0]),
            Q_payload_vel=np.diag([3.0, 3.0, 3.0]),
            R_u=np.diag([5.0e-3, 5.0e-3, 5.0e-3]),
            Qf_payload=np.diag([120.0, 120.0, 180.0]),
        )

        self.dyn = SoftplusDynamics(self.params)
        self.nmpc = MultipleShootingNMPC(
            dyn=self.dyn,
            cfg=self.mpc_cfg,
            weights=self.weights,
            payload_target=PAYLOAD_TARGET,
        )

        self.state = State(
            x1=INITIAL_CARRIER.copy(),
            x2=INITIAL_PAYLOAD.copy(),
            v1=np.zeros(3, dtype=float),
            v2=np.zeros(3, dtype=float),
        )

        self.sim_time = 0.0
        self.warmstart = None
        self.z_guess = None

        self.carrier_pub = self.create_publisher(Odometry, "/carrier/odom", 10)
        self.payload_pub = self.create_publisher(Odometry, "/payload/odom", 10)
        self.tension_pub = self.create_publisher(Float64, "/cable/tension", 10)
        self.solve_time_pub = self.create_publisher(Float64, "/nmpc/solve_time_ms", 10)
        self.cost_pub = self.create_publisher(Float64, "/nmpc/cost", 10)
        self.phi_pub = self.create_publisher(Float64, "/nmpc/phi", 10)
        self.distance_pub = self.create_publisher(Float64, "/nmpc/carrier_payload_distance", 10)

        self.timer = self.create_timer(self.params.dt, self.step)

        self.get_logger().info("Multiple-shooting softplus NMPC node with collision avoidance started")

    def step(self) -> None:
        x0 = self.state.as_vector()

        t0 = time.perf_counter()
        sol = self.nmpc.solve(x0, warmstart=self.warmstart)
        solve_ms = 1.0e3 * (time.perf_counter() - t0)

        self.warmstart = self.nmpc.shift_warmstart(sol)

        u0 = sol["U"][0]
        w0 = sol["W"][0]

        gamma0 = self.dyn.gamma_from_w(w0)

        x_next, self.z_guess = self.dyn.simulate_one_step(x0, u0, z_guess=self.z_guess)
        self.state = State.from_vector(x_next)
        self.sim_time += self.params.dt

        phi = cable_gap(self.state.x1, self.state.x2, self.params.cable_length)
        dist_cp = pair_distance_num(self.state.as_vector())

        now = self.get_clock().now().to_msg()
        self.carrier_pub.publish(self._make_odom(now, "world", "carrier", self.state.x1, self.state.v1))
        self.payload_pub.publish(self._make_odom(now, "world", "payload", self.state.x2, self.state.v2))
        self.tension_pub.publish(Float64(data=gamma0))
        self.solve_time_pub.publish(Float64(data=solve_ms))
        self.cost_pub.publish(Float64(data=float(sol["f_opt"])))
        self.phi_pub.publish(Float64(data=phi))
        self.distance_pub.publish(Float64(data=dist_cp))

        self.get_logger().info(
            f"t={self.sim_time:.2f} "
            f"solve_ms={solve_ms:.2f} "
            f"cost={sol['f_opt']:.3f} "
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
    node = PayloadNMPCNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
