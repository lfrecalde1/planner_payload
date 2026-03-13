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

        # Time Definition
        self.ts = 0.03
        self.final = 15
        self.t =np.arange(0, self.final + self.ts, self.ts, dtype=np.double)

        # Prediction Node of the NMPC formulation
        self.t_N = 1.0
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
        kp_min = 80
        self.kp_min = kp_min
        self.kv_min = 50
        self.c1 = c1
        
        # Cable length
        self.length = 0.83
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

        ### Init states for the optimizer
        self.x_0 = np.hstack((pos_0, vel_0, self.n_init, self.r_init))

        ### Init Control Actions or equilibirum
        self.r_dot_init = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)
        self.u_equilibrium = np.hstack((self.tensions_init, self.r_dot_init))

        #### Maximum and minimun control actions
        self.tension_min = 0.5*self.tensions_init
        self.tension_max = 4.5*self.tensions_init

        self.r_dot_max = np.array([2.0, 2.0, 2.0]*self.robot_num, dtype=np.double)
        self.r_dot_min = -self.r_dot_max

        self.u_min =  np.hstack((self.tension_min, self.r_dot_min))
        self.u_max =  np.hstack((self.tension_max, self.r_dot_max))

        ## Check values
        print(self.u_equilibrium)
        print(self.u_min)
        print(self.u_max)

        #### Define state dimension and control action
        self.n_x = self.x_0.shape[0]
        self.n_u = self.u_equilibrium.shape[0]
        print(self.n_x)
        print(self.n_u)

        # Define odometry subscriber
        self.subscriber_payload_ = self.create_subscription(Odometry, "/quadrotor/payload/odom", self.callback_get_odometry_payload, 10)
        self.subscriber_drone_0_ = self.create_subscription(Odometry, "/quadrotor/odom", self.callback_get_odometry_drone_0, 10)

        # Define PositionCmd publisher for each droe
        self.publisher_ref_drone_0 = self.create_publisher(PositionCommand, "/quadrotor/position_cmd", 10)

        self.publisher_prediction_drone_0 = self.create_publisher(Path, "/quadrotor/predicted_path", 10)

        self.publisher_prediction_payload = self.create_publisher(Path, "/quadrotor/predicted_path", 10)

        # TF
        self.tf_broadcaster = TransformBroadcaster(self)

        ### Create the initial states for quadrotor
        pos_quad_0 = np.array([0.0, 0.0, 1.3], dtype=np.double)
        ### Linear velocity of the sytem respect to the inertial frame
        vel_quad_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Angular velocity respect to the Body frame
        omega_quad_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Initial Orientation expressed as quaternionn
        quat_quad_0 = np.array([1.0, 0.0, 0.0, 0.0])

        self.xq0_0 = np.hstack((pos_quad_0, vel_quad_0, quat_quad_0, omega_quad_0))
        print(self.xq0_0)

        self.unit_vector_from_measurements = self.quadrotor_payload_unit_vector_c()
        self.cable_angular_velocity_from_measurements = self.cable_angular_velocity_c()

        self.quadrotor_position = self.quadrotor_position_c()
        self.quadrotor_velocity = self.quadrotor_velocity_c()
        self.quadrotor_acceleration = self.quadrotor_acceleration_c()

        ## Desired States
        ### Desired states and control actions
        self.xd = np.zeros((self.n_x, ), dtype=np.double)
        self.ud = np.zeros((self.n_u, ), dtype=np.double)

        self.xd[0] = 1.0
        self.xd[1] = 0.0
        self.xd[2] = 2.0

        self.xd[3] = 0.0
        self.xd[4] = 0.0
        self.xd[5] = 0.0

        self.xd[6] = 0.0
        self.xd[7] = 0.0
        self.xd[8] = -1.0

        self.xd[9] = 0.0
        self.xd[10] = 0.0
        self.xd[11] = 0.0

        #### Set Desired Control Actions
        self.ud[0] = self.tensions_init

        self.ud[1] = 0.0
        self.ud[2] = 0.0
        self.ud[3] = 0.0

        ## Flag
        self.flag = 0

        ## Control Loop
        #self.timer = self.create_timer(self.ts, self.run)  # 0.01 seconds = 100 Hz
        #self.start_time = time.time()

        self.timer = self.create_timer(self.ts, self.validation)  # 0.01 seconds = 100 Hz
        self.start_time = time.time()

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
        return None

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
        linear_acceleration = -(1/(self.mass))*t_1_cmd*n1 - self.gravity*self.e3 - ((self.mass_quad*self.length)/(self.mass + self.mass_quad))*(cross_angular_payload.T@cross_angular_payload)*n1 

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

        t_d = p[12]
        r_dot_d = p[13:16]

        error_position_quad = x_p - x_p_d
        error_velocity_quad = v_p - v_p_d

        # Cost functions
        lyapunov_position = (1/2)*self.kp_min*error_position_quad.T@error_position_quad + self.kv_min*(1/2)*(self.mass)*error_velocity_quad.T@error_velocity_quad + self.c1*error_position_quad.T@error_velocity_quad

        # Error cable direction
        error_n1 = ca.cross(n1_d, n1)
        # Cost Function control actions
        tension_error = t_d - t_cmd
        r_error = r1_d - r1

        ocp.model.cost_expr_ext_cost = lyapunov_position   + error_n1.T@error_n1 + 1*(r_error.T@r_error) + 1*(tension_error*tension_error) + 10*(r_dot_cmd.T@r_dot_cmd)
        ocp.model.cost_expr_ext_cost_e = lyapunov_position + error_n1.T@error_n1 + 1*(r_error.T@r_error) 

        ref_params = np.hstack((self.x_0, self.u_equilibrium))

        ocp.parameter_values = ref_params

        ocp.constraints.constr_type = 'BGH'

        # Set constraints
        ocp.constraints.lbu = self.u_min
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.x0 = x0

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" 
        ocp.solver_options.qp_solver_cond_N = self.N_prediction // 4
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  
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
        n_k      = (term/ca.norm_2(term))
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
        cross_angular_payload = ca.cross(r_p, n_p)
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
        norm_a = ca.norm_2(a)
        dot_a = a.T@a
        I = ca.MX.eye(3)
        a_dot = v_p - v_Q

        n_dot_k = (1/norm_a)*(I - (a@a.T)/dot_a)@a_dot
        r_k = ca.cross(n_p, n_dot_k)
        r_velocity_f = ca.Function('r_velocity_f', [x, xQ], [r_k])
        return r_velocity_f

    def send_position_cmd(self, publisher, x, v, a):
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
        publisher.publish(position_cmd_msg)
        return None 

    def publish_transforms(self):
        # Payload
## -------------------------------------------------------------------------------------------------------------------
        tf_world_load = TransformStamped()
        tf_world_load.header.stamp = self.get_clock().now().to_msg()
        tf_world_load.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load.child_frame_id = 'payload'          # <-- imu_link is rotated
        tf_world_load.transform.translation.x = self.x_0[0]
        tf_world_load.transform.translation.y = self.x_0[1]
        tf_world_load.transform.translation.z = self.x_0[2]
        
        # Payload Verification with unit vector
        tf_world_load_verification = TransformStamped()
        tf_world_load_verification.header.stamp = self.get_clock().now().to_msg()
        tf_world_load_verification.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load_verification.child_frame_id = 'payload_verification'          # <-- imu_link is rotated
        tf_world_load_verification.transform.translation.x = self.xq0_0[0] + self.x_0[6]*self.length
        tf_world_load_verification.transform.translation.y = self.xq0_0[1] + self.x_0[7]*self.length
        tf_world_load_verification.transform.translation.z = self.xq0_0[2] + self.x_0[8]*self.length

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

        # Publish each drone’s path
        self.publisher_prediction_drone_0.publish(path_msgs[0])
        self.publisher_prediction_payload.publish(payload_msgs[0])

    def geometric_control(self, xd, vd, ad, t, n):
        # Control Error
        aux_variable = ad 
        ad = (aux_variable - t*n)
        return xd, vd, ad
    
    def prepare(self):
        if self.flag == 0:
            self.flag = 1
            # Init Optimization Problem
            for k in range(5000):
                arr_str = np.array2string(self.x_0, precision=3, separator=", ", suppress_small=True)
                self.get_logger().info(f"state[] = {arr_str}")

            print(self.x_0.shape)
            self.ocp = self.solver(self.x_0)
            self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp_" + self.ocp.model.name + ".json", build= True, generate= True)

            ### Reset Solver
            self.acados_ocp_solver.reset()

            ### Initial Conditions optimization problem
            for stage in range(self.N_prediction + 1):
                self.acados_ocp_solver.set(stage, "x", self.x_0)
            for stage in range(self.N_prediction):
                self.acados_ocp_solver.set(stage, "u", self.ud)
        return None
        
    def run(self):
        # Build Optimization Problem just once
        self.prepare()

        self.acados_ocp_solver.set(0, "lbx", self.x_0)
        self.acados_ocp_solver.set(0, "ubx", self.x_0)

        # Desired Trajectory of the system
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
        self.acados_ocp_solver.solve()

        # get Control Actions and predictions
        u = self.acados_ocp_solver.get(0, "u")
        x_k = self.acados_ocp_solver.get(1, "x")

        ## Send Desired States Quadrotor
        self.publish_prediction()

        xQ = np.array(self.quadrotor_position(x_k)).reshape((self.robot_num*3, ))
        xQ_dot = np.array(self.quadrotor_velocity(x_k)).reshape((self.robot_num*3, ))
        xQ_dot_dot = np.array(self.quadrotor_acceleration(x_k, u)).reshape((self.robot_num*3, ))

        xd_q0, vd_q0, ad_q0 = self.geometric_control(xQ[0:3], xQ_dot[0:3], xQ_dot_dot[0:3], u[0], x_k[6:9])

        self.send_position_cmd(self.publisher_ref_drone_0, xd_q0, vd_q0, ad_q0)
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
        self.acados_ocp_solver.solve()

        # get Control Actions and predictions
        u = self.acados_ocp_solver.get(0, "u")
        x_k = self.acados_ocp_solver.get(1, "x")
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
        payload_node.destroy_node()
        rclpy.shutdown()
    finally:
        payload_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()
