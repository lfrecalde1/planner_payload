#!/usr/bin/env python3
import time

import casadi as ca
import numpy as np
import rclpy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from quadrotor_msgs.msg import PositionCommand
from rclpy.node import Node


class TwoQuadrotorPayloadPlanner(Node):
    def __init__(self):
        super().__init__("TwoQuadrotorPayloadPlanner")

        self.ts = 0.05
        self.t_N = 1.0
        self.N_prediction = int(self.t_N / self.ts)

        self.mass_payload = 0.11
        self.gravity = 9.81
        self.length = 0.83
        self.e3 = np.array([0.0, 0.0, 1.0], dtype=np.double)

        # State: [p(3), v(3), n1(3), n2(3), r1(3), r2(3)]
        self.n_x = 18
        # Input: [t1, t2, r1_dot(3), r2_dot(3)]
        self.n_u = 8

        self.tension_nominal = self.mass_payload * self.gravity / 2.0
        self.tension_min = 0.3 * self.tension_nominal
        self.tension_max = 10.0 * self.tension_nominal
        self.r_dot_limit = 6.0
        self.dot_separation_gain = 0.5
        self.norm_consistency_gain = 10.0
        self.orthogonality_gain = 10.0

        self.u_min = np.hstack(
            (
                np.array([self.tension_min, self.tension_min], dtype=np.double),
                -self.r_dot_limit * np.ones(6, dtype=np.double),
            )
        )
        self.u_max = np.hstack(
            (
                np.array([self.tension_max, self.tension_max], dtype=np.double),
                self.r_dot_limit * np.ones(6, dtype=np.double),
            )
        )

        # Measured states
        self.payload_p = np.array([0.0, 0.0, 0.47], dtype=np.double)
        self.payload_v = np.zeros(3, dtype=np.double)
        self.quad1_p = np.array([0.4, 0.0, 1.3], dtype=np.double)
        self.quad1_v = np.zeros(3, dtype=np.double)
        self.quad2_p = np.array([-0.4, 0.0, 1.3], dtype=np.double)
        self.quad2_v = np.zeros(3, dtype=np.double)

        self.x_0 = np.zeros(self.n_x, dtype=np.double)
        self.xd = np.zeros(self.n_x, dtype=np.double)
        self.ud = np.zeros(self.n_u, dtype=np.double)

        self.payload_odom_received = False
        self.quad1_odom_received = False
        self.quad2_odom_received = False
        self.reference_initialized = False
        self.start_time = None
        self.payload_ref_start = self.payload_p.copy()

        self.flag = 0
        self.acados_ocp_solver = None

        # Subscribers
        self.create_subscription(Odometry, "/quadrotor1/payload/odom", self.cb_payload_odom, 10)
        self.create_subscription(Odometry, "/quadrotor1/odom", self.cb_quad1_odom, 10)
        self.create_subscription(Odometry, "/quadrotor2/odom", self.cb_quad2_odom, 10)

        # Publishers
        self.pub_cmd_q1 = self.create_publisher(PositionCommand, "/quadrotor1/position_cmd", 10)
        self.pub_cmd_q2 = self.create_publisher(PositionCommand, "/quadrotor2/position_cmd", 10)

        self.pub_pred_q1 = self.create_publisher(Path, "/quadrotor1/predicted_path", 10)
        self.pub_pred_q2 = self.create_publisher(Path, "/quadrotor2/predicted_path", 10)
        self.pub_pred_payload = self.create_publisher(Path, "/quadrotor1/payload/predicted_path", 10)

        self.pub_des_q1 = self.create_publisher(Path, "/quadrotor1/desired_path", 10)
        self.pub_des_q2 = self.create_publisher(Path, "/quadrotor2/desired_path", 10)
        self.pub_des_payload = self.create_publisher(Path, "/quadrotor1/payload/desired_path", 10)

        self.timer = self.create_timer(self.ts, self.run)

    def cb_payload_odom(self, msg: Odometry):
        self.payload_p = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.double
        )
        self.payload_v = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=np.double
        )
        self.payload_odom_received = True
        self.update_measured_state()

    def cb_quad1_odom(self, msg: Odometry):
        self.quad1_p = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.double
        )
        self.quad1_v = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=np.double
        )
        self.quad1_odom_received = True
        self.update_measured_state()

    def cb_quad2_odom(self, msg: Odometry):
        self.quad2_p = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.double
        )
        self.quad2_v = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=np.double
        )
        self.quad2_odom_received = True
        self.update_measured_state()

    def _unit_n_n_dot_r_from_measurements(self, p_l, v_l, p_q, v_q):
        a = p_l - p_q
        norm_a = np.linalg.norm(a)
        if norm_a < 1e-6:
            n = np.array([0.0, 0.0, -1.0], dtype=np.double)
            n_dot = np.zeros(3, dtype=np.double)
            r = np.zeros(3, dtype=np.double)
            return n, n_dot, r

        n = a / norm_a
        a_dot = v_l - v_q
        I = np.eye(3)
        n_dot = ((I - np.outer(n, n)) @ a_dot) / norm_a
        r = np.cross(n, n_dot)
        return n, n_dot, r

    def update_measured_state(self):
        if not (self.payload_odom_received and self.quad1_odom_received and self.quad2_odom_received):
            return

        n1, n1_dot, r1 = self._unit_n_n_dot_r_from_measurements(
            self.payload_p, self.payload_v, self.quad1_p, self.quad1_v
        )
        n2, n2_dot, r2 = self._unit_n_n_dot_r_from_measurements(
            self.payload_p, self.payload_v, self.quad2_p, self.quad2_v
        )

        self.x_0 = np.hstack((self.payload_p, self.payload_v, n1, n2, r1, r2)).astype(np.double)

        if not self.reference_initialized:
            self.payload_ref_start = self.payload_p.copy()
            self.start_time = time.time()
            self.reference_initialized = True
            self.get_logger().info(
                f"Initialized two-quad planner at payload position {np.round(self.payload_ref_start, 3)}"
            )
        # Keep cable references at measured values
        self.xd[6:9] = np.array([0, 0, -1])
        self.xd[9:12] = np.array([0, 0, -1])
        self.xd[12:15] = np.array([0, 0, 0])
        self.xd[15:18] = np.array([0, 0, 0])

        # Equilibrium/nominal controls: tensions + zero cable angular acceleration
        self.ud = np.hstack(
            (
                np.array([self.tension_nominal, self.tension_nominal], dtype=np.double),
                np.zeros(6, dtype=np.double),
            )
        )

    def _base_lissajous(self, t):
        xc, yc, zc = self.payload_ref_start
        Ax, Ay, Az = 1.2, 0.8, 0.0
        wx, wy, wz = 0.8, 1.4, 0.6
        phix, phiy, phiz = 0.0, np.pi / 3.0, np.pi / 6.0

        pd = np.array(
            [
                xc + Ax * (np.sin(wx * t + phix) - np.sin(phix)),
                yc + Ay * (np.sin(wy * t + phiy) - np.sin(phiy)),
                zc + Az * (np.sin(wz * t + phiz) - np.sin(phiz)),
            ],
            dtype=np.double,
        )
        vd = np.array(
            [
                Ax * wx * np.cos(wx * t + phix),
                Ay * wy * np.cos(wy * t + phiy),
                Az * wz * np.cos(wz * t + phiz),
            ],
            dtype=np.double,
        )
        ad = np.array(
            [
                -Ax * wx * wx * np.sin(wx * t + phix),
                -Ay * wy * wy * np.sin(wy * t + phiy),
                -Az * wz * wz * np.sin(wz * t + phiz),
            ],
            dtype=np.double,
        )
        return pd, vd, ad

    def payload_model(self):
        model_name = "two_quad_payload_r_dynamics"

        p = ca.MX.sym("p", 3)
        v = ca.MX.sym("v", 3)
        n1 = ca.MX.sym("n1", 3)
        n2 = ca.MX.sym("n2", 3)
        r1 = ca.MX.sym("r1", 3)
        r2 = ca.MX.sym("r2", 3)
        x = ca.vertcat(p, v, n1, n2, r1, r2)

        t1 = ca.MX.sym("t1")
        t2 = ca.MX.sym("t2")
        r1_dot_cmd = ca.MX.sym("r1_dot_cmd", 3)
        r2_dot_cmd = ca.MX.sym("r2_dot_cmd", 3)
        u = ca.vertcat(t1, t2, r1_dot_cmd, r2_dot_cmd)

        p_dot = v
        v_dot = -(1.0 / self.mass_payload) * (t1 * n1 + t2 * n2) - self.gravity * ca.DM(self.e3)
        n1_dot = ca.cross(r1, n1)
        n2_dot = ca.cross(r2, n2)
        r1_dot = r1_dot_cmd
        r2_dot = r2_dot_cmd

        f_expl = ca.vertcat(p_dot, v_dot, n1_dot, n2_dot, r1_dot, r2_dot)

        p_ref = ca.MX.sym("p_ref", self.n_x + self.n_u)

        model = AcadosModel()
        model.name = model_name
        model.x = x
        model.u = u
        model.p = p_ref
        model.f_expl_expr = f_expl
        return model

    def solver(self, x0):
        model = self.payload_model()

        ocp = AcadosOcp()
        ocp.model = model
        ocp.p = model.p
        ocp.dims.N = self.N_prediction

        x = ocp.model.x
        u = ocp.model.u
        p = ocp.model.p

        p_err = x[0:3] - p[0:3]
        v_err = x[3:6] - p[3:6]

        n1_err = ca.cross(x[6:9], p[6:9])
        n2_err = ca.cross(x[9:12], p[9:12])

        r1_err = x[12:15] - p[12:15]
        r2_err = x[15:18] - p[15:18]

        t_err = u[0:2] - p[18:20]
        rdot_err = u[2:8] - p[20:26]

        n1 = x[6:9]
        n2 = x[9:12]
        r1 = x[12:15]
        r2 = x[15:18]

        # Physical consistency soft constraints
        norm_pen = (ca.dot(n1, n1) - 1.0) ** 2 + (ca.dot(n2, n2) - 1.0) ** 2
        ortho_pen = (ca.dot(n1, r1)) ** 2 + (ca.dot(n2, r2)) ** 2
        # Separation objective:
        # for fixed cable lengths, ||q1-q2||^2 = L^2 * (2 - 2*(n1·n2)),
        # so minimizing (n1·n2) increases quadrotor separation.
        dot_n1_n2 = ca.dot(n1, n2)

        running_cost = (
            100.0 * ca.dot(p_err, p_err)
            + 50.0 * ca.dot(v_err, v_err)
            + 0.1 * ca.dot(n1_err, n1_err)
            + 0.1 * ca.dot(n2_err, n2_err)
            + 0.1 * ca.dot(r1_err, r1_err)
            + 0.1 * ca.dot(r2_err, r2_err)
            + 5.0 * ca.dot(t_err, t_err)
            + 10.0 * ca.dot(rdot_err, rdot_err)
            + self.norm_consistency_gain * norm_pen
            + self.orthogonality_gain * ortho_pen
            - self.dot_separation_gain * dot_n1_n2
        )

        terminal_cost = (
            100.0 * ca.dot(p_err, p_err)
            + 50.0 * ca.dot(v_err, v_err)
            + 0.1 * ca.dot(n1_err, n1_err)
            + 0.1 * ca.dot(n2_err, n2_err)
            + 0.1 * ca.dot(r1_err, r1_err)
            + 0.1 * ca.dot(r2_err, r2_err)
            + self.norm_consistency_gain * norm_pen
            + self.orthogonality_gain * ortho_pen
            - self.dot_separation_gain * dot_n1_n2
        )

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = running_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

        ocp.parameter_values = np.hstack((self.x_0, self.ud)).astype(np.double)

        ocp.constraints.lbu = self.u_min
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.arange(self.n_u)
        ocp.constraints.x0 = x0

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.Tsim = self.ts
        ocp.solver_options.tf = self.t_N

        return ocp

    def _quad_kinematics_from_state_input(self, xk, uk):
        p = xk[0:3]
        v = xk[3:6]
        n1 = xk[6:9]
        n2 = xk[9:12]
        r1 = xk[12:15]
        r2 = xk[15:18]

        t1, t2 = uk[0], uk[1]
        r1_dot = uk[2:5]
        r2_dot = uk[5:8]

        q1_p = p - self.length * n1
        q2_p = p - self.length * n2

        q1_v = v - self.length * np.cross(r1, n1)
        q2_v = v - self.length * np.cross(r2, n2)

        payload_acc = -(1.0 / self.mass_payload) * (t1 * n1 + t2 * n2) - self.gravity * self.e3

        q1_a = payload_acc - self.length * (
            np.cross(r1_dot, n1) + np.cross(r1, np.cross(r1, n1))
        )
        q2_a = payload_acc - self.length * (
            np.cross(r2_dot, n2) + np.cross(r2, np.cross(r2, n2))
        )

        return q1_p, q1_v, q1_a, q2_p, q2_v, q2_a

    def _send_position_cmd(self, pub, p, v, a):
        msg = PositionCommand()
        msg.position.x = float(p[0])
        msg.position.y = float(p[1])
        msg.position.z = float(p[2])
        msg.velocity.x = float(v[0])
        msg.velocity.y = float(v[1])
        msg.velocity.z = float(v[2])
        msg.acceleration.x = float(a[0])
        msg.acceleration.y = float(a[1])
        msg.acceleration.z = float(a[2])
        pub.publish(msg)

    def _publish_prediction_paths(self):
        hdr = self.get_clock().now().to_msg()

        path_q1 = Path()
        path_q1.header.stamp = hdr
        path_q1.header.frame_id = "world"

        path_q2 = Path()
        path_q2.header.stamp = hdr
        path_q2.header.frame_id = "world"

        path_payload = Path()
        path_payload.header.stamp = hdr
        path_payload.header.frame_id = "world"

        for k in range(self.N_prediction):
            xk = self.acados_ocp_solver.get(k, "x")

            pose_q1 = PoseStamped()
            pose_q1.header = path_q1.header
            pose_q1.pose.position.x = float(xk[0] - self.length * xk[6])
            pose_q1.pose.position.y = float(xk[1] - self.length * xk[7])
            pose_q1.pose.position.z = float(xk[2] - self.length * xk[8])
            path_q1.poses.append(pose_q1)

            pose_q2 = PoseStamped()
            pose_q2.header = path_q2.header
            pose_q2.pose.position.x = float(xk[0] - self.length * xk[9])
            pose_q2.pose.position.y = float(xk[1] - self.length * xk[10])
            pose_q2.pose.position.z = float(xk[2] - self.length * xk[11])
            path_q2.poses.append(pose_q2)

            pose_p = PoseStamped()
            pose_p.header = path_payload.header
            pose_p.pose.position.x = float(xk[0])
            pose_p.pose.position.y = float(xk[1])
            pose_p.pose.position.z = float(xk[2])
            path_payload.poses.append(pose_p)

        self.pub_pred_q1.publish(path_q1)
        self.pub_pred_q2.publish(path_q2)
        self.pub_pred_payload.publish(path_payload)

    def _publish_desired_paths(self, t_now):
        hdr = self.get_clock().now().to_msg()

        path_q1 = Path()
        path_q1.header.stamp = hdr
        path_q1.header.frame_id = "world"

        path_q2 = Path()
        path_q2.header.stamp = hdr
        path_q2.header.frame_id = "world"

        path_payload = Path()
        path_payload.header.stamp = hdr
        path_payload.header.frame_id = "world"

        n1_d = self.xd[6:9]
        n2_d = self.xd[9:12]

        for k in range(self.N_prediction + 1):
            pd, _, _ = self._base_lissajous(t_now + k * self.ts)

            pose_p = PoseStamped()
            pose_p.header = path_payload.header
            pose_p.pose.position.x = float(pd[0])
            pose_p.pose.position.y = float(pd[1])
            pose_p.pose.position.z = float(pd[2])
            path_payload.poses.append(pose_p)

            q1_d = pd - self.length * n1_d
            q2_d = pd - self.length * n2_d

            pose_q1 = PoseStamped()
            pose_q1.header = path_q1.header
            pose_q1.pose.position.x = float(q1_d[0])
            pose_q1.pose.position.y = float(q1_d[1])
            pose_q1.pose.position.z = float(q1_d[2])
            path_q1.poses.append(pose_q1)

            pose_q2 = PoseStamped()
            pose_q2.header = path_q2.header
            pose_q2.pose.position.x = float(q2_d[0])
            pose_q2.pose.position.y = float(q2_d[1])
            pose_q2.pose.position.z = float(q2_d[2])
            path_q2.poses.append(pose_q2)

        self.pub_des_q1.publish(path_q1)
        self.pub_des_q2.publish(path_q2)
        self.pub_des_payload.publish(path_payload)

    def prepare(self):
        if self.flag != 0 or not self.reference_initialized:
            return

        self.flag = 1
        ocp = self.solver(self.x_0)
        self.acados_ocp_solver = AcadosOcpSolver(
            ocp,
            json_file="acados_ocp_" + ocp.model.name + ".json",
            build=True,
            generate=True,
        )
        self.acados_ocp_solver.reset()

        for k in range(self.N_prediction + 1):
            self.acados_ocp_solver.set(k, "x", self.x_0)
        for k in range(self.N_prediction):
            self.acados_ocp_solver.set(k, "u", self.ud)

    def run(self):
        if not self.reference_initialized:
            return

        self.prepare()
        if self.flag == 0:
            return

        self.acados_ocp_solver.set(0, "lbx", self.x_0)
        self.acados_ocp_solver.set(0, "ubx", self.x_0)

        t_now = time.time() - self.start_time

        for k in range(self.N_prediction):
            pd, vd, _ = self._base_lissajous(t_now + k * self.ts)

            xref = np.zeros(self.n_x, dtype=np.double)
            xref[0:3] = pd
            xref[3:6] = vd
            xref[6:9] = self.xd[6:9]
            xref[9:12] = self.xd[9:12]
            xref[12:15] = self.xd[12:15]
            xref[15:18] = self.xd[15:18]

            pref = np.hstack((xref, self.ud))
            self.acados_ocp_solver.set(k, "p", pref)

        pdN, vdN, _ = self._base_lissajous(t_now + self.N_prediction * self.ts)
        xref_N = np.zeros(self.n_x, dtype=np.double)
        xref_N[0:3] = pdN
        xref_N[3:6] = vdN
        xref_N[6:9] = self.xd[6:9]
        xref_N[9:12] = self.xd[9:12]
        xref_N[12:15] = self.xd[12:15]
        xref_N[15:18] = self.xd[15:18]

        pref_N = np.hstack((xref_N, self.ud))
        self.acados_ocp_solver.set(self.N_prediction, "p", pref_N)

        status = self.acados_ocp_solver.solve()
        if status != 0:
            self.get_logger().error(f"acados returned status {status}")
            try:
                stats = self.acados_ocp_solver.get_stats("statistics")
                self.get_logger().error(f"acados statistics:\n{stats}")
            except Exception:
                pass
            return

        u0 = self.acados_ocp_solver.get(0, "u")
        x1 = self.acados_ocp_solver.get(1, "x")

        q1_p, q1_v, q1_a, q2_p, q2_v, q2_a = self._quad_kinematics_from_state_input(x1, u0)

        self._send_position_cmd(self.pub_cmd_q1, q1_p, q1_v, q1_a)
        self._send_position_cmd(self.pub_cmd_q2, q2_p, q2_v, q2_a)

        self._publish_prediction_paths()
        self._publish_desired_paths(t_now)


def main(args=None):
    rclpy.init(args=args)
    node = TwoQuadrotorPayloadPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Simulation stopped manually.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
