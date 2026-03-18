#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
from casadi import Function
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import PositionCommand
from scipy.spatial.transform import Rotation as R
import time
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


class PayloadControlMujocoNode(Node):
    def __init__(self):
        super().__init__('SinglePayloadPlanner')

        # Runtime parameters (mirrors dq_nmpc style parameterization).
        self.declare_parameter('planner.ts', 0.05)
        self.declare_parameter('planner.horizon_time', 1.0)
        self.declare_parameter('planner.trajectory_speed_scale', 1.0)
        self.declare_parameter('planner.transition_hold_time', 2.0)
        self.declare_parameter('planner.transition_blend_time', 3.5)
        self.declare_parameter('planner.acceleration_phase_time', 3.5)
        self.declare_parameter('planner.cruise_speed_factor', 1.5)
        self.declare_parameter('planner.height_offset', 0.8)
        self.declare_parameter('planner.reference_mode', 'trajectory')
        self.declare_parameter('planner.regulation_offset', [1.0, 1.0, 0.5])
        self.declare_parameter('nmpc.weight_cable_direction', 0.1)
        self.declare_parameter('nmpc.weight_tension', 5.0)
        self.declare_parameter('nmpc.weight_rdot', 5.0)
        self.declare_parameter('nmpc.weight_orthogonality', 10.0)
        self.declare_parameter('nmpc.norm_constraint_slack_weight', 100.0)
        self.declare_parameter('nmpc.unit_vector_norm_tol', 1e-3)
        self.declare_parameter('nmpc.regularize_method', 'CONVEXIFY')

        # Time Definition
        self.ts = float(self.get_parameter('planner.ts').value)
        self.final = 15

        # Prediction Node of the NMPC formulation
        self.t_N = float(self.get_parameter('planner.horizon_time').value)
        self.N = np.arange(0, self.t_N + self.ts, self.ts)
        self.N_prediction = self.N.shape[0]

        # Internal parameters defintion
        self.robot_num = 1
        self.mass = 0.11
        self.gravity = 9.81


        # Quadrotor paramaters
        self.mass_quad = 1.05
        self.inertia_quad = np.array([[0.00345398, 0.0, 0.0], [0.0, 0.00179687, 0.0], [0.0, 0.0, 0.00179676]], dtype=np.double)

        # Control gains
        c1 = 1
        kv_min = c1 + 1/4 + 0.1
        kp_min = (c1*(kv_min*kv_min) + 2*kv_min*c1 - c1*c1)/((self.mass)*(4*(kv_min - c1)-1))
        kp_min = 100
        self.kp_min = kp_min
        self.kv_min = 50
        self.c1 = c1
        
        # Cable length
        self.length = 0.88
        self.e3 = ca.DM([0, 0, 1])

        # Position of the system payload
        pos_0 = np.array([0.0, 0.0, 0.47], dtype=np.double)
        # Linear velocity of the payload
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        
        # Initial Wrench
        # This is just the mass of the payload an gravity
        Wrench0 = np.array([0, 0, (self.mass)*self.gravity])

        # Init Tension of the cables so we can get initial cable direction
        # Init state payload
        self.init = np.hstack((pos_0, vel_0))

        # Compute the initial tension based on the the Wrench
        self.tensions_init = np.linalg.norm(Wrench0, axis=0)

        # Compute the cable direction initial condition
        self.n_init = -Wrench0/self.tensions_init

        # Compute the cable initial angular velocity
        self.r_init = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)

        # Init states for the optimizer
        self.x_0 = np.hstack((pos_0, vel_0, self.n_init, self.r_init))

        # Init Control Actions or equilibirum
        self.r_dot_init = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)
        self.u_equilibrium = np.hstack((self.tensions_init, self.r_dot_init))

        # Maximum and minimun control actions
        self.tension_min = 0.8*self.tensions_init
        self.tension_max = 8.0*self.tensions_init

        self.r_dot_max = np.array([6.0, 6.0, 6.0]*self.robot_num, dtype=np.double)
        self.r_dot_min = -self.r_dot_max

        self.u_min =  np.hstack((self.tension_min, self.r_dot_min))
        self.u_max =  np.hstack((self.tension_max, self.r_dot_max))

        # Define state dimension and control action
        self.n_x = self.x_0.shape[0]
        self.n_u = self.u_equilibrium.shape[0]

        # Define odometry subscriber
        self.subscriber_payload_ = self.create_subscription(Odometry, "/quadrotor/payload/odom", self.callback_get_odometry_payload, 10)
        self.publisher_desired_payload = self.create_publisher(Path, "/quadrotor/payload/desired_path", 10)


        # Define PositionCmd publisher for each droe
        self.publisher_ref_drone_0 = self.create_publisher(PositionCommand, "/quadrotor/position_cmd", 10)
        self.publisher_prediction_drone_0 = self.create_publisher(Path, "/quadrotor/predicted_path", 10)
        self.publisher_prediction_payload = self.create_publisher(Path, "/quadrotor/payload/predicted_path", 10)
        self.publisher_desired_quadrotor = self.create_publisher(Path, "/quadrotor/desired_path", 10)
        
        # Subcriber of each drone
        self.subscriber_drone_0_ = self.create_subscription(Odometry, "/quadrotor/odom", self.callback_get_odometry_drone_0, 10)

        # TF
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create the initial states for quadrotor
        pos_quad_0 = np.array([0.0, 0.0, 1.3], dtype=np.double)
        # Linear velocity of the sytem respect to the inertial frame
        vel_quad_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Angular velocity respect to the Body frame
        omega_quad_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Initial Orientation expressed as quaternionn
        quat_quad_0 = np.array([1.0, 0.0, 0.0, 0.0])

        self.xq0_0 = np.hstack((pos_quad_0, vel_quad_0, quat_quad_0, omega_quad_0))

        self.unit_vector_from_measurements = self.quadrotor_payload_unit_vector_c()
        self.cable_angular_velocity_from_measurements = self.cable_angular_velocity_c()

        self.quadrotor_position = self.quadrotor_position_c()
        self.quadrotor_velocity = self.quadrotor_velocity_c()
        self.quadrotor_acceleration = self.quadrotor_acceleration_c()

        # Desired States
        self.xd = np.zeros((self.n_x, ), dtype=np.double)
        self.ud = np.zeros((self.n_u, ), dtype=np.double)

        self.xd[0] = 0.8
        self.xd[1] = 1.07
        self.xd[2] = 1.0

        self.xd[3] = 0.0
        self.xd[4] = 0.0
        self.xd[5] = 0.0

        self.xd[6] = 0.0
        self.xd[7] = 0.0
        self.xd[8] = -1.0

        self.xd[9] = 0.0
        self.xd[10] = 0.0
        self.xd[11] = 0.0

        # Set Desired Control Actions
        self.ud[0] = self.tensions_init
        self.ud[1] = 0.0
        self.ud[2] = 0.0
        self.ud[3] = 0.0

        # Flag
        self.flag = 0
        self.payload_odom_received = False
        self.quad_odom_received = False
        self.reference_initialized = False
        self.payload_ref_start = self.x_0[0:3].copy()

        self.timer = self.create_timer(self.ts, self.run)
        self.start_time = None
        self.trajectory_start_time = None
        self.trajectory_speed_scale = float(self.get_parameter('planner.trajectory_speed_scale').value)
        self.transition_hold_time = float(self.get_parameter('planner.transition_hold_time').value)
        self.transition_blend_time = float(self.get_parameter('planner.transition_blend_time').value)
        self.acceleration_phase_time = float(self.get_parameter('planner.acceleration_phase_time').value)
        self.cruise_speed_factor = float(self.get_parameter('planner.cruise_speed_factor').value)
        self.height_offset = float(self.get_parameter('planner.height_offset').value)
        self.reference_mode = str(self.get_parameter('planner.reference_mode').value).strip().lower()
        self.regulation_offset = np.array(self.get_parameter('planner.regulation_offset').value, dtype=np.double).reshape((3,))
        self.weight_cable_direction = float(self.get_parameter('nmpc.weight_cable_direction').value)
        self.weight_tension = float(self.get_parameter('nmpc.weight_tension').value)
        self.weight_rdot = float(self.get_parameter('nmpc.weight_rdot').value)
        self.weight_orthogonality = float(self.get_parameter('nmpc.weight_orthogonality').value)
        self.norm_constraint_slack_weight = float(self.get_parameter('nmpc.norm_constraint_slack_weight').value)
        self.unit_vector_norm_tol = float(self.get_parameter('nmpc.unit_vector_norm_tol').value)
        self.regularize_method = str(self.get_parameter('nmpc.regularize_method').value)

    def _base_lissajous(self, t: float):
        xc, yc, zc = self.payload_ref_start[0], self.payload_ref_start[1], self.payload_ref_start[2] + self.height_offset

        # Amplitudes
        Ax, Ay, Az = 5.0, 1.0, 0.0

        # Different frequencies -> true Lissajous
        wx, wy, wz = 0.8, 1.4, 0.6

        # Phases
        phix, phiy, phiz = 0.0, np.pi / 3.0, np.pi / 6.0

        xd = np.array([
            xc + Ax * (np.sin(wx * t + phix) - np.sin(phix)),
            yc + Ay * (np.sin(wy * t + phiy) - np.sin(phiy)),
            zc + Az * (np.sin(wz * t + phiz) - np.sin(phiz))
        ], dtype=np.double)

        vd = np.array([
            Ax * wx * np.cos(wx * t + phix),
            Ay * wy * np.cos(wy * t + phiy),
            Az * wz * np.cos(wz * t + phiz)
        ], dtype=np.double)

        ad = np.array([
            -Ax * wx * wx * np.sin(wx * t + phix),
            -Ay * wy * wy * np.sin(wy * t + phiy),
            -Az * wz * wz * np.sin(wz * t + phiz)
        ], dtype=np.double)

        jd = np.array([
            -Ax * wx * wx * wx * np.cos(wx * t + phix),
            -Ay * wy * wy * wy * np.cos(wy * t + phiy),
            -Az * wz * wz * wz * np.cos(wz * t + phiz)
        ], dtype=np.double)

        sd = np.array([
            Ax * wx * wx * wx * wx * np.sin(wx * t + phix),
            Ay * wy * wy * wy * wy * np.sin(wy * t + phiy),
            Az * wz * wz * wz * wz * np.sin(wz * t + phiz)
        ], dtype=np.double)

        return xd, vd, ad, jd, sd

    def _phase_time_scaling(self, t_move: float):
        # Explicit acceleration phase: tau_dot ramps from 0 -> v_cruise with C2 continuity.
        T = max(self.acceleration_phase_time, 1e-3)
        v_cruise = max(self.cruise_speed_factor * self.trajectory_speed_scale, 1e-3)
        if t_move >= T:
            tau = v_cruise * (t_move - 0.5 * T)
            return tau, v_cruise, 0.0, 0.0, 0.0

        s = np.clip(t_move / T, 0.0, 1.0)
        s2 = s * s
        s3 = s2 * s
        s4 = s3 * s
        s5 = s4 * s
        s6 = s5 * s

        # tau_dot(s) = 6 s^5 - 15 s^4 + 10 s^3
        tau_dot = v_cruise * (6.0 * s5 - 15.0 * s4 + 10.0 * s3)
        # Integral of tau_dot over t in [0, t_move]
        tau = T * v_cruise * (s6 - 3.0 * s5 + 2.5 * s4)
        tau_ddot = v_cruise * (30.0 * s4 - 60.0 * s3 + 30.0 * s2) / T
        tau_3dot = v_cruise * (120.0 * s3 - 180.0 * s2 + 60.0 * s) / (T * T)
        tau_4dot = v_cruise * (360.0 * s2 - 360.0 * s + 60.0) / (T * T * T)
        return tau, tau_dot, tau_ddot, tau_3dot, tau_4dot

    def desired_lissajous(self, t: float):
        if t <= self.transition_hold_time:
            z = np.zeros(3, dtype=np.double)
            return self.payload_ref_start.copy(), z, z, z, z

        t_move = t - self.transition_hold_time
        tau, tau_dot, tau_ddot, tau_3dot, tau_4dot = self._phase_time_scaling(t_move)
        pd_nom, vd_nom, ad_nom, jd_nom, sd_nom = self._base_lissajous(tau)

        # Chain rule from nominal time tau to real time t.
        pd = pd_nom
        vd = vd_nom * tau_dot
        ad = ad_nom * (tau_dot * tau_dot) + vd_nom * tau_ddot
        jd = jd_nom * (tau_dot ** 3) + 3.0 * ad_nom * tau_dot * tau_ddot + vd_nom * tau_3dot
        sd = (
            sd_nom * (tau_dot ** 4)
            + 6.0 * jd_nom * (tau_dot ** 2) * tau_ddot
            + 3.0 * ad_nom * (tau_ddot ** 2)
            + 4.0 * ad_nom * tau_dot * tau_3dot
            + vd_nom * tau_4dot
        )
        return pd, vd, ad, jd, sd

    def _smoothstep_c4(self, s: float):
        # 9th-order smoothstep with zero derivatives up to 4th order at both ends.
        a = 126.0 * s**5 - 420.0 * s**6 + 540.0 * s**7 - 315.0 * s**8 + 70.0 * s**9
        a_s = 630.0 * s**4 - 2520.0 * s**5 + 3780.0 * s**6 - 2520.0 * s**7 + 630.0 * s**8
        a_ss = 2520.0 * s**3 - 12600.0 * s**4 + 22680.0 * s**5 - 17640.0 * s**6 + 5040.0 * s**7
        a_sss = 7560.0 * s**2 - 50400.0 * s**3 + 113400.0 * s**4 - 105840.0 * s**5 + 35280.0 * s**6
        a_ssss = 15120.0 * s - 151200.0 * s**2 + 453600.0 * s**3 - 529200.0 * s**4 + 211680.0 * s**5
        return a, a_s, a_ss, a_sss, a_ssss

    def desired_regulation(self, t: float):
        p0 = self.payload_ref_start
        pf = self.payload_ref_start + self.regulation_offset
        delta = pf - p0

        if t <= self.transition_hold_time:
            z = np.zeros(3, dtype=np.double)
            return p0.copy(), z, z, z, z

        T = max(self.transition_blend_time, 1e-3)
        t_move = t - self.transition_hold_time
        if t_move >= T:
            z = np.zeros(3, dtype=np.double)
            return pf.copy(), z, z, z, z

        s = np.clip(t_move / T, 0.0, 1.0)
        a, a_s, a_ss, a_sss, a_ssss = self._smoothstep_c4(s)
        a_dot = a_s / T
        a_ddot = a_ss / (T * T)
        a_3dot = a_sss / (T ** 3)
        a_4dot = a_ssss / (T ** 4)

        pd = p0 + a * delta
        vd = a_dot * delta
        ad = a_ddot * delta
        jd = a_3dot * delta
        sd = a_4dot * delta
        return pd, vd, ad, jd, sd

    def desired_reference(self, t: float):
        if self.reference_mode == 'regulation':
            return self.desired_regulation(t)
        return self.desired_lissajous(t)
    def publish_desired_path(self):
        now = self.get_clock().now().to_msg()

        payload_path = Path()
        payload_path.header.stamp = now
        payload_path.header.frame_id = "world"

        quad_path = Path()
        quad_path.header.stamp = now
        quad_path.header.frame_id = "world"

        # Use the same reference cable direction you use in MPC
        n_ref = np.array([0.0, 0.0, -1.0], dtype=np.double)
        t0 = self.trajectory_start_time if self.trajectory_start_time is not None else time.time()

        for k in range(self.N_prediction + 1):
            tk = (time.time() - t0) + k * self.ts
            pd, vd, ad, _, _ = self.desired_reference(tk)

            # Desired payload pose
            pose_payload = PoseStamped()
            pose_payload.header = payload_path.header
            pose_payload.pose.position.x = float(pd[0])
            pose_payload.pose.position.y = float(pd[1])
            pose_payload.pose.position.z = float(pd[2])
            pose_payload.pose.orientation.w = 1.0
            payload_path.poses.append(pose_payload)

            # Desired quadrotor pose reconstructed from payload + cable
            pq_d = pd - self.length * n_ref

            pose_quad = PoseStamped()
            pose_quad.header = quad_path.header
            pose_quad.pose.position.x = float(pq_d[0])
            pose_quad.pose.position.y = float(pq_d[1])
            pose_quad.pose.position.z = float(pq_d[2])
            pose_quad.pose.orientation.w = 1.0
            quad_path.poses.append(pose_quad)

        self.publisher_desired_payload.publish(payload_path)
        self.publisher_desired_quadrotor.publish(quad_path)

    def callback_get_odometry_payload(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((6, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        x[3] = msg.twist.twist.linear.x
        x[4] = msg.twist.twist.linear.y
        x[5] = msg.twist.twist.linear.z

        # Get quadrotor  positions
        xquadrotors = self.xq0_0[0:3]
        # Get quadrotor  velocities
        vquadrotors = self.xq0_0[3:6]
        # Get Full vector quadrotor
        x_quadrotor = np.hstack((xquadrotors, vquadrotors))

        # Compute unit Vector
        unit = np.array(self.unit_vector_from_measurements(x, xquadrotors)).reshape((self.robot_num*3, ))

        # Extended Vector of the Payload
        payload_states = np.hstack((x, unit))

        # Compute cable angular velocity
        r = np.array(self.cable_angular_velocity_from_measurements(payload_states, x_quadrotor)).reshape((self.robot_num*3, ))

        payload_states_full = np.hstack((x, unit, r))
        self.x_0 = payload_states_full
        self.payload_odom_received = True
        self.try_initialize_reference()

        #arr_str = np.array2string(self.x_0, precision=3, separator=', ', suppress_small=True)
        #self.get_logger().info(f"x_0 = {arr_str}")
        return None

    def callback_get_odometry_drone_0(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        x[3] = msg.twist.twist.linear.x
        x[4] = msg.twist.twist.linear.y
        x[5] = msg.twist.twist.linear.z
        
        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w

        self.xq0_0 = x
        self.quad_odom_received = True
        self.try_initialize_reference()
        return None

    def try_initialize_reference(self):
        if self.reference_initialized:
            return
        if not (self.payload_odom_received and self.quad_odom_received):
            return
        self.payload_ref_start = self.x_0[0:3].copy()
        self.reference_initialized = True
        arr_str = np.array2string(self.payload_ref_start, precision=3, separator=", ", suppress_small=True)
        self.get_logger().info(f"Initialized desired path at measured payload position {arr_str}")
        if self.reference_mode == 'regulation':
            target = self.payload_ref_start + self.regulation_offset
            target_str = np.array2string(target, precision=3, separator=", ", suppress_small=True)
            self.get_logger().info(f"Regulation target (initial + offset): {target_str}")

    def payloadModel(self)->AcadosModel:
        # Model Name
        model_name = "simple_payload"

        #position 
        p_x = ca.MX.sym('p_x')
        p_y = ca.MX.sym('p_y')
        p_z = ca.MX.sym('p_z')
        x_p = ca.vertcat(p_x, p_y, p_z)
    
        #linear vel
        vx_p = ca.MX.sym("vx_p")
        vy_p = ca.MX.sym("vy_p")
        vz_p = ca.MX.sym("vz_p")   
        v_p = ca.vertcat(vx_p, vy_p, vz_p)

        # Cable kinematics
        nx_1 = ca.MX.sym('nx_1')
        ny_1 = ca.MX.sym('ny_1')
        nz_1 = ca.MX.sym('nz_1')
        n1 = ca.vertcat(nx_1, ny_1, nz_1)

        # Cable kinematics
        rx_1 = ca.MX.sym('rx_1')
        ry_1 = ca.MX.sym('ry_1')
        rz_1 = ca.MX.sym('rz_1')
        r1 = ca.vertcat(rx_1, ry_1, rz_1)
        
        # Full states of the system
        x = ca.vertcat(x_p, v_p, n1, r1)
        
        # Control actions of the system
        t_1_cmd = ca.MX.sym("t_1_cmd")
        rx_1_cmd = ca.MX.sym("rx_1_cmd")
        ry_1_cmd = ca.MX.sym("ry_1_cmd")
        rz_1_cmd = ca.MX.sym("rz_1_cmd")

        r1_cmd = ca.vertcat(rx_1_cmd, ry_1_cmd, rz_1_cmd) 

        # Vector of control actions
        u = ca.vertcat(t_1_cmd, rx_1_cmd, ry_1_cmd, rz_1_cmd)

        # Linear Dynamics
        linear_velocity = v_p
        cross_angular_payload = ca.cross(r1, n1)
        linear_acceleration = -(1/(self.mass))*t_1_cmd*n1 - self.gravity*self.e3

        # Angular dynamics
        # Cable Kinematics
        n1_dot = ca.cross(r1, n1)

        r1_dot = (r1_cmd)

        # Explicit Dynamics
        f_expl = ca.vertcat(linear_velocity, linear_acceleration, n1_dot, r1_dot)
        p = ca.MX.sym('p', x.shape[0] + u.shape[0], 1)

        # Dynamics
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.p = p
        model.name = model_name
        return model

    def solver(self, x0):
        # get dynamical model
        model = self.payloadModel()
        
        # Optimal control problem
        ocp = AcadosOcp()
        ocp.model = model

        # Get size of the system
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu

        # Set Dimension of the problem
        ocp.p = model.p
        ocp.dims.N = self.N_prediction

        # Definition of the cost functions (EXTERNAL)
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL" 

        # some variables
        x = ocp.model.x
        u = ocp.model.u
        p = ocp.model.p

        # Split states of the system
        x_p = x[0:3]
        v_p = x[3:6]
        n1 = x[6:9]
        r1 = x[9:12]

        # Split control actions
        t_cmd = u[0]
        r_dot_cmd = u[1:4]

        # Get desired states of the system
        x_p_d = p[0:3]
        v_p_d = p[3:6]
        n1_d = p[6:9]
        r1_d = p[9:12]
        
        # Desired Control Actions 
        t_d = p[12]
        r_dot_d = p[13:16]
        
        # Error of linear dynamics
        error_position_quad = x_p - x_p_d
        error_velocity_quad = v_p - v_p_d

        # Cost functions
        lyapunov_position = 100*(1/2)*self.kp_min*error_position_quad.T@error_position_quad + self.kv_min*(1/2)*(self.mass)*error_velocity_quad.T@error_velocity_quad

        # Error cable direction
        error_n1 = ca.cross(n1_d, n1)
        r_error = r1_d - r1

        # Cost Function control actions
        tension_error = t_d - t_cmd
        r_dot_error = r_dot_d - r_dot_cmd 
        
        # Enforce the velocity is orthogonal
        orthogonality_error = ca.dot(n1, r1)

        ocp.model.cost_expr_ext_cost = (
            lyapunov_position
            + self.weight_cable_direction * (error_n1.T @ error_n1)
            + self.weight_tension * (tension_error * tension_error)
            + self.weight_rdot * (r_dot_error.T @ r_dot_error)
            + self.weight_orthogonality * (orthogonality_error**2)
        )
        ocp.model.cost_expr_ext_cost_e = (
            lyapunov_position
            + self.weight_cable_direction * (error_n1.T @ error_n1)
            + self.weight_orthogonality * (orthogonality_error**2)
        )

        ref_params = np.hstack((self.x_0, self.u_equilibrium))

        ocp.parameter_values = ref_params

        ocp.constraints.constr_type = 'BGH'

        # Set constraints
        ocp.constraints.lbu = self.u_min
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.x0 = x0

        # Softly enforce ||n1|| ~= 1 to improve robustness against numerical drift.
        ocp.model.con_h_expr = ca.vertcat(ca.dot(n1, n1))
        nh = 1
        nsbx = 0
        nsh = nh
        ns = nsh + nsbx
        ocp.cost.zl = self.norm_constraint_slack_weight * np.ones((ns, ))
        ocp.cost.Zl = self.norm_constraint_slack_weight * np.ones((ns, ))
        ocp.cost.zu = self.norm_constraint_slack_weight * np.ones((ns, ))
        ocp.cost.Zu = self.norm_constraint_slack_weight * np.ones((ns, ))
        ocp.constraints.lh = np.array([1.0 - self.unit_vector_norm_tol])
        ocp.constraints.uh = np.array([1.0 + self.unit_vector_norm_tol])
        ocp.constraints.lsh = np.zeros((nsh, ))
        ocp.constraints.ush = np.zeros((nsh, ))
        ocp.constraints.idxsh = np.array(range(nsh), dtype=np.int32)

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" 
        ocp.solver_options.qp_solver_cond_N = self.N_prediction // 4
        ocp.solver_options.hessian_approx = "EXACT"  
        ocp.solver_options.regularize_method = self.regularize_method
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.Tsim = self.ts
        ocp.solver_options.tf = self.t_N

        return ocp
    
    def quadrotor_payload_unit_vector_c(self):
        x = ca.MX.sym('x', 6, 1)

        x_p = x[0:3]

        L = self.length

        xq = ca.MX.sym('xq', 3*self.robot_num, 1)

        # Vectorized expression:
        term = x_p - xq
        n_k      = term / ca.fmax(ca.norm_2(term), 1e-6)
        quadrotor_payload_vector_f = ca.Function('quadrotor_payload_vector_f', [x, xq], [n_k])
        return quadrotor_payload_vector_f

    def quadrotor_position_c(self):
        x = ca.MX.sym('x', self.n_x, 1)

        x_p   = x[0:3]          # 3x1
        n = x[6:9]        

        # Vectorized expression:
        quadrotor = x_p - (self.length * n)  # 3 x m
        quad_position_f = ca.Function('quad_position_f', [x], [quadrotor])
        return quad_position_f

    def quadrotor_velocity_c(self):
        L = self.length

        # state & input
        x = ca.MX.sym('x', self.n_x, 1)

        # unpack state
        x_p = x[0:3]
        v_p = x[3:6]
        n_p = x[6:9]
        r_p = x[9:12]

        term_n   = L * ca.cross(r_p, n_p)
        v_k      = v_p - term_n     
        quad_velocity_f = ca.Function('quad_velocity_f', [x], [v_k])
        return quad_velocity_f

    def quadrotor_acceleration_c(self):
        # --- symbols ---
        L = self.length

        # state & input
        x = ca.MX.sym('x', self.n_x, 1)
        u = ca.MX.sym('u', self.n_u, 1) 

        # unpack state
        x_p = x[0:3]
        v_p = x[3:6]
        n_p = x[6:9]
        r_p = x[9:12]

        # unpack Control Action
        t_1_cmd = u[0]          
        r_dot = u[1:]

        # Linear Acceleration Payload
        linear_acceleration = -(1/(self.mass))*t_1_cmd*n_p - self.gravity*self.e3

        term_acceleration = linear_acceleration
        input_angular_acc_cable = - L*(ca.cross(r_dot, n_p))
        angular_velocity_cable_aux = ca.cross(r_p, n_p)
        angular_velocity_cable = -L*ca.cross(r_p, angular_velocity_cable_aux)
        a_k      = term_acceleration + input_angular_acc_cable + angular_velocity_cable
        quad_acc_f = ca.Function('quad_acc_f', [x, u], [a_k])
        return quad_acc_f

    def cable_angular_velocity_c(self):
        L = self.length

        # state & input
        x = ca.MX.sym('x', 9, 1)
        xQ = ca.MX.sym('xQ', 6, 1)  # general: 3 thrust comps + 3m 'r' comps

        # unpack state
        x_p   = x[0:3]      # 3x1
        v_p   = x[3:6]      # 3x1
        n_p   = x[6:9]

        # unpack Quadrotor velocity
        x_Q = xQ[0:3]
        v_Q = xQ[3:6]

        a = x_p - x_Q
        norm_a = ca.fmax(ca.norm_2(a), 1e-6)
        dot_a = ca.fmax(a.T@a, 1e-12)
        I = ca.MX.eye(3)
        a_dot = v_p - v_Q

        n_dot_k = (1/norm_a)*(I - (a@a.T)/dot_a)@a_dot
        r_k = ca.cross(n_p, n_dot_k)
        r_velocity_f = ca.Function('r_velocity_f', [x, xQ], [r_k])
        return r_velocity_f

    def send_position_cmd(self, publisher, x, v, a, tension, direction):
        position_cmd_msg = PositionCommand()
        position_cmd_msg.position.x = x[0]
        position_cmd_msg.position.y = x[1]
        position_cmd_msg.position.z = x[2]

        position_cmd_msg.velocity.x = v[0]
        position_cmd_msg.velocity.y = v[1]
        position_cmd_msg.velocity.z = v[2]
        
        position_cmd_msg.acceleration.x = a[0]
        position_cmd_msg.acceleration.y = a[1]
        position_cmd_msg.acceleration.z = a[2]

        cable_force = tension*direction

        position_cmd_msg.cable_force.x = cable_force[0]
        position_cmd_msg.cable_force.y = cable_force[1]
        position_cmd_msg.cable_force.z = cable_force[2]


        publisher.publish(position_cmd_msg)
        return None 

    def publish_transforms(self):
        # Payload
        tf_world_load = TransformStamped()
        tf_world_load.header.stamp = self.get_clock().now().to_msg()
        tf_world_load.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load.child_frame_id = 'payload'          # <-- imu_link is rotated

        tf_world_load.transform.translation.x = self.x_0[0]
        tf_world_load.transform.translation.y = self.x_0[1]
        tf_world_load.transform.translation.z = self.x_0[2]

        tf_world_load.transform.rotation.w = 1.0
        tf_world_load.transform.rotation.x = 0.0
        tf_world_load.transform.rotation.y = 0.0
        tf_world_load.transform.rotation.z = 0.0

        # Payload Verification with unit vector
        tf_world_load_verification = TransformStamped()
        tf_world_load_verification.header.stamp = self.get_clock().now().to_msg()
        tf_world_load_verification.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load_verification.child_frame_id = 'payload_verification'          # <-- imu_link is rotated

        tf_world_load_verification.transform.translation.x = self.xq0_0[0] + self.x_0[6]*self.length
        tf_world_load_verification.transform.translation.y = self.xq0_0[1] + self.x_0[7]*self.length
        tf_world_load_verification.transform.translation.z = self.xq0_0[2] + self.x_0[8]*self.length

        tf_world_load_verification.transform.rotation.w = 1.0
        tf_world_load_verification.transform.rotation.x = 0.0
        tf_world_load_verification.transform.rotation.y = 0.0
        tf_world_load_verification.transform.rotation.z = 0.0

        # Quadrotor
        tf_world_quad1 = TransformStamped()
        tf_world_quad1.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad1.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad1.child_frame_id = 'quadrotor'          # <-- imu_link is rotated

        tf_world_quad1.transform.translation.x = self.xq0_0[0]
        tf_world_quad1.transform.translation.y = self.xq0_0[1]
        tf_world_quad1.transform.translation.z = self.xq0_0[2]

        tf_world_quad1.transform.rotation.x = self.xq0_0[7]
        tf_world_quad1.transform.rotation.y = self.xq0_0[8]
        tf_world_quad1.transform.rotation.z = self.xq0_0[9]
        tf_world_quad1.transform.rotation.w = self.xq0_0[6]
        self.tf_broadcaster.sendTransform([tf_world_load, tf_world_quad1, tf_world_load_verification])
        return None

    def publish_prediction(self):
        # Create one Path message per drone
        path_msgs = []
        payload_msgs = []

        # Quadrotors
        for i in range(self.robot_num):
            msg = Path()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "world"
            path_msgs.append(msg)
        
        # Payload
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        payload_msgs.append(msg)
        
        # Fill poses for each drone
        for k in range(self.N_prediction):
            x_k = self.acados_ocp_solver.get(k, "x")
            xq = np.array(self.quadrotor_position(x_k)).reshape((self.robot_num * 3,))

            # Quadrotor positions
            for i in range(self.robot_num):
                pose = PoseStamped()
                pose.header = path_msgs[i].header
                pose.pose.position.x = xq[3*i + 0]
                pose.pose.position.y = xq[3*i + 1]
                pose.pose.position.z = xq[3*i + 2]
                path_msgs[i].poses.append(pose)

            # Payload positions
            pose = PoseStamped()
            pose.header = payload_msgs[0].header
            pose.pose.position.x = x_k[0]
            pose.pose.position.y = x_k[1]
            pose.pose.position.z = x_k[2]
            payload_msgs[0].poses.append(pose)

        # Publish drone and payload desired path
        self.publisher_prediction_drone_0.publish(path_msgs[0])
        self.publisher_prediction_payload.publish(payload_msgs[0])
    
    def prepare(self):
        if self.flag == 0:
            if not self.reference_initialized:
                return None
            self.flag = 1
            self.ocp = self.solver(self.x_0)
            self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp_" + self.ocp.model.name + ".json", build= True, generate= True)

            ### Reset Solver
            self.acados_ocp_solver.reset()

            ### Initial Conditions optimization problem
            for stage in range(self.N_prediction + 1):
                self.acados_ocp_solver.set(stage, "x", self.x_0)
            for stage in range(self.N_prediction):
                self.acados_ocp_solver.set(stage, "u", self.ud)
            # Start trajectory clock only when the solver is ready.
            self.trajectory_start_time = time.time()
            self.start_time = self.trajectory_start_time
        return None

    def run(self):
        if not self.reference_initialized:
            return
        self.prepare()
        if self.flag == 0:
            return

        self.acados_ocp_solver.set(0, "lbx", self.x_0)
        self.acados_ocp_solver.set(0, "ubx", self.x_0)

        if self.trajectory_start_time is None:
            return
        t_now = time.time() - self.trajectory_start_time
        for j in range(self.N_prediction):
            tj = t_now + j * self.ts
            pd, vd, ad, _, _ = self.desired_reference(tj)

            yref = np.zeros((self.n_x,), dtype=np.double)
            yref[0:3] = pd
            yref[3:6] = vd

            # keep cable reference simple for now
            yref[6:9] = np.array([0.0, 0.0, -1.0], dtype=np.double)
            yref[9:12] = np.zeros(3, dtype=np.double)

            uref = np.zeros((self.n_u,), dtype=np.double)
            tension_ff = -self.mass * np.dot(ad + np.array([0.0, 0.0, self.gravity]), np.array([0.0, 0.0, -1.0]))
            uref[0] = float(np.clip(tension_ff, self.tension_min, self.tension_max))
            uref[1:4] = 0.0

            aux_ref = np.hstack((yref, uref))
            self.acados_ocp_solver.set(j, "p", aux_ref)

        pdN, vdN, adN, _, _ = self.desired_reference(t_now + self.N_prediction * self.ts)
        yref_N = np.zeros((self.n_x,), dtype=np.double)
        yref_N[0:3] = pdN
        yref_N[3:6] = vdN
        yref_N[6:9] = np.array([0.0, 0.0, -1.0], dtype=np.double)
        yref_N[9:12] = np.zeros(3, dtype=np.double)

        uref_N = np.zeros((self.n_u,), dtype=np.double)
        tension_ff_N = -self.mass * np.dot(adN + np.array([0.0, 0.0, self.gravity]), np.array([0.0, 0.0, -1.0]))
        uref_N[0] = float(np.clip(tension_ff_N, self.tension_min, self.tension_max))
        aux_ref_N = np.hstack((yref_N, uref_N))
        self.acados_ocp_solver.set(self.N_prediction, "p", aux_ref_N)

        status = self.acados_ocp_solver.solve()
        if status != 0:
            self.get_logger().error(f"acados returned status {status}")
            self.publish_transforms()
            return

        self.publish_desired_path()
        u = self.acados_ocp_solver.get(0, "u")
        x_k = self.acados_ocp_solver.get(1, "x")

        self.publish_prediction()

        xQ = np.array(self.quadrotor_position(x_k)).reshape((3,))
        xQ_dot = np.array(self.quadrotor_velocity(x_k)).reshape((3,))
        xQ_dot_dot = np.array(self.quadrotor_acceleration(x_k, u)).reshape((3,))

        self.send_position_cmd(self.publisher_ref_drone_0, xQ[0:3], xQ_dot[0:3], xQ_dot_dot[0:3], u[0], x_k[6:9])
        self.get_logger().info("Solving the MPC problem")
        # Build Optimization Problem just once
        self.publish_transforms()

    def validation(self):
        # Build Optimization Problem just once
        self.prepare()

        self.acados_ocp_solver.set(0, "lbx", self.x_0)
        self.acados_ocp_solver.set(0, "ubx", self.x_0)

        for j in range(self.N_prediction):
            yref = self.xd
            uref = self.ud
            aux_ref = np.hstack((yref, uref))
            self.acados_ocp_solver.set(j, "p", aux_ref)

        # Desired Trayectory at the last Horizon
        yref_N = self.xd
        uref_N = self.ud
        aux_ref_N = np.hstack((yref_N, uref_N))
        self.acados_ocp_solver.set(self.N_prediction, "p", aux_ref_N)
        # Check Solution since there can be possible errors 
        status = self.acados_ocp_solver.solve()
        if status != 0:
            self.get_logger().error(f"acados returned status {status}")
            return
        
        self.publish_desired_path()
        # Get Control Actions and predictions
        u = self.acados_ocp_solver.get(0, "u")
        x_k = self.acados_ocp_solver.get(1, "x")

        self.publish_prediction()

        xQ = np.array(self.quadrotor_position(x_k)).reshape((self.robot_num*3, ))
        xQ_dot = np.array(self.quadrotor_velocity(x_k)).reshape((self.robot_num*3, ))
        xQ_dot_dot = np.array(self.quadrotor_acceleration(x_k, u)).reshape((self.robot_num*3, ))

        self.send_position_cmd(self.publisher_ref_drone_0, xQ[0:3], xQ_dot[0:3], xQ_dot_dot[0:3], u[0], x_k[6:9])
        self.get_logger().info("Solving the MPC problem")

        # Build Optimization Problem just once
        self.publish_transforms()



def main(arg = None):
    rclpy.init(args=arg)
    payload_node = PayloadControlMujocoNode()
    try:
        rclpy.spin(payload_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        payload_node.get_logger().info('Simulation stopped manually.')
    finally:
        payload_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()
