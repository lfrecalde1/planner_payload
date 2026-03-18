#!/usr/bin/env python3
"""ROS2 quadrotor policy node for a trained MjLab checkpoint.

The loaded policy is the actor trained for the ``Mjlab-Quadrotor-Hover`` task.
This node performs pose regulation only and reproduces the training-time
observation and action conversion:

  obs = [dq pose error, body linear velocity, body angular velocity,
         projected gravity, previous raw action]

  thrust_cmd = clip(m g + 8 a0, 0, 60)
  omega_d    = clip(6 [a1, a2, a3], -6, 6)

The ACP TRPY simulator then applies the inner body-rate controller through the
``kom`` gains in ``quadrotor_msgs/TRPYCommand``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
import torch
import torch.nn as nn
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import TRPYCommand
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster


class QuadrotorActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(19, 128),
      nn.ELU(),
      nn.Linear(128, 128),
      nn.ELU(),
      nn.Linear(128, 4),
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(obs)


class QuadrotorPolicyNode(Node):
  def __init__(self):
    super().__init__("quadrotor_policy_node")

    # Timing.
    self.ts = 0.05
    self.start_time: float | None = None

    # Policy/task parameters. These must match the trained task.
    self.checkpoint_file = Path(
      "/home/fer/mjlab/logs/rsl_rl/quadrotor_hover/2026-03-16_17-29-35/model_499.pt"
    )
    self.mass = 0.94
    self.gravity = 9.81
    self.rate_scale = np.array([6.0, 6.0, 6.0], dtype=np.float32)
    self.rate_gains = np.array([20.0, 35.0, 45.0], dtype=np.float32)
    self.dq_eps = 1e-8
    self.declare_parameter("target_x", float("nan"))
    self.declare_parameter("target_y", float("nan"))
    self.declare_parameter("target_z", float("nan"))

    # State.
    self.prev_action = np.zeros(4, dtype=np.float32)
    self.odom_received = False
    self.target_initialized = False
    self.target_position_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    self.base_pos_w = np.zeros(3, dtype=np.float64)
    self.base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    self.base_lin_vel_b = np.zeros(3, dtype=np.float64)
    self.base_ang_vel_b = np.zeros(3, dtype=np.float64)

    # ROS interfaces.
    self.subscriber_quad_odom = self.create_subscription(
      Odometry, "/quadrotor/odom", self.callback_get_odometry_drone_0, 10
    )
    self.publisher_policy_cmd = self.create_publisher(
      TRPYCommand, "/quadrotor/trpy_cmd", 10
    )
    self.tf_broadcaster = TransformBroadcaster(self)

    # Load the trained actor once.
    self.device = "cpu"
    self.policy = self._load_policy()

    self.timer = self.create_timer(self.ts, self.run)

  # Policy loading.

  def _load_policy(self):
    checkpoint = torch.load(str(self.checkpoint_file), map_location=self.device)
    actor_state = checkpoint["actor_state_dict"]
    mlp_state = {
      key: value for key, value in actor_state.items() if key.startswith("mlp.")
    }
    policy = QuadrotorActor().to(self.device)
    policy.load_state_dict(mlp_state, strict=True)
    policy.eval()
    self.get_logger().info(f"Loaded checkpoint: {self.checkpoint_file}")
    return policy

  # Quaternion / dual-quaternion math matching the training task.

  def _h_plus(self, q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
      [
        [w, -x, -y, -z],
        [x, w, -z, y],
        [y, z, w, -x],
        [z, -y, x, w],
      ],
      dtype=np.float64,
    )

  def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

  def _quat_left_product(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return self._h_plus(q) @ p

  def _pose_to_dual_quat(self, pos_w: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    trans_quat = np.array([0.0, pos_w[0], pos_w[1], pos_w[2]], dtype=np.float64)
    dual = 0.5 * self._quat_left_product(trans_quat, quat_wxyz)
    return np.hstack((quat_wxyz, dual))

  def _dual_quat_conjugate(self, qd: np.ndarray) -> np.ndarray:
    return np.hstack(
      (self._quat_conjugate(qd[0:4]), self._quat_conjugate(qd[4:8]))
    )

  def _dual_h_plus(self, qd: np.ndarray) -> np.ndarray:
    h_real = self._h_plus(qd[0:4])
    h_dual = self._h_plus(qd[4:8])
    zeros = np.zeros((4, 4), dtype=np.float64)
    return np.block([[h_real, zeros], [h_dual, h_real]])

  def _dual_quat_pose_error_log(
    self, current_dq: np.ndarray, desired_dq: np.ndarray
  ) -> np.ndarray:
    desired_conj = self._dual_quat_conjugate(desired_dq)
    q_error = self._dual_h_plus(desired_conj) @ current_dq

    q_error_real = q_error[0:4]
    q_error_dual = q_error[4:8]
    q_error_real_conj = self._quat_conjugate(q_error_real)

    imag = q_error_real[1:4]
    imag_norm = np.linalg.norm(imag)
    safe_imag_norm = max(imag_norm, self.dq_eps)
    angle = np.arctan2(imag_norm, q_error_real[0])
    trans_error = 2.0 * self._quat_left_product(q_error_dual, q_error_real_conj)

    log_quat = 0.5 * angle * imag / safe_imag_norm
    log_trans = 0.5 * trans_error[1:4]
    return np.hstack((log_quat, log_trans)).astype(np.float32)

  # Observation helpers.

  def quat_wxyz_to_rotation(self, q_wxyz: np.ndarray) -> R:
    return R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

  def world_to_body(self, q_wxyz: np.ndarray, v_w: np.ndarray) -> np.ndarray:
    return self.quat_wxyz_to_rotation(q_wxyz).inv().apply(v_w)

  def projected_gravity_body(self, q_wxyz: np.ndarray) -> np.ndarray:
    return self.world_to_body(q_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float64))

  def build_observation(self) -> np.ndarray:
    # Pose is expressed in the world frame. The desired attitude for this
    # regulation task is the identity orientation in the world frame.
    current_dq = self._pose_to_dual_quat(self.base_pos_w, self.base_quat_wxyz)
    desired_dq = self._pose_to_dual_quat(
      self.target_position_w, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    dq_pose_error = self._dual_quat_pose_error_log(current_dq, desired_dq)
    g_b = self.projected_gravity_body(self.base_quat_wxyz).astype(np.float32)

    obs = np.concatenate(
      [
        dq_pose_error,
        self.base_lin_vel_b.astype(np.float32),
        self.base_ang_vel_b.astype(np.float32),
        g_b,
        self.prev_action,
      ],
      axis=0,
    )
    return obs.astype(np.float32)

  # Policy / command conversion.

  def policy_step(self, obs: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
      obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
      raw_action = self.policy(obs_t)[0].detach().cpu().numpy().astype(np.float32)
    self.prev_action = raw_action.copy()
    return raw_action

  def action_to_command(self, raw_action: np.ndarray) -> tuple[float, np.ndarray]:
    force_cmd = float(np.clip(self.mass * self.gravity + 8.0 * raw_action[0], 0.0, 60.0))
    desired_rates = np.clip(self.rate_scale * raw_action[1:4], -6.0, 6.0)
    return force_cmd, desired_rates

  # ROS callbacks / publishing.

  def callback_get_odometry_drone_0(self, msg: Odometry):
    self.base_pos_w[0] = msg.pose.pose.position.x
    self.base_pos_w[1] = msg.pose.pose.position.y
    self.base_pos_w[2] = msg.pose.pose.position.z
    self.base_quat_wxyz[0] = msg.pose.pose.orientation.w
    self.base_quat_wxyz[1] = msg.pose.pose.orientation.x
    self.base_quat_wxyz[2] = msg.pose.pose.orientation.y
    self.base_quat_wxyz[3] = msg.pose.pose.orientation.z
    base_lin_vel_w = np.array(
      [
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.linear.z,
      ],
      dtype=np.float64,
    )
    self.base_lin_vel_b = self.world_to_body(self.base_quat_wxyz, base_lin_vel_w)
    self.base_ang_vel_b[0] = msg.twist.twist.angular.x
    self.base_ang_vel_b[1] = msg.twist.twist.angular.y
    self.base_ang_vel_b[2] = msg.twist.twist.angular.z
    self.odom_received = True
    self.try_initialize_target()

  def try_initialize_target(self):
    if self.target_initialized or not self.odom_received:
      return
    configured_target = np.array(
      [
        self.get_parameter("target_x").value,
        self.get_parameter("target_y").value,
        self.get_parameter("target_z").value,
      ],
      dtype=np.float64,
    )
    if np.all(np.isfinite(configured_target)):
      self.target_position_w = configured_target
      target_source = "configured target"
    else:
      self.target_position_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
      target_source = "training default"
    self.start_time = time.time()
    self.target_initialized = True
    arr_str = np.array2string(
      self.target_position_w, precision=3, separator=", ", suppress_small=True
    )
    self.get_logger().info(f"Initialized regulation target from {target_source}: {arr_str}")

  def publish_transforms(self):
    tf_world_quad = TransformStamped()
    tf_world_quad.header.stamp = self.get_clock().now().to_msg()
    tf_world_quad.header.frame_id = "world"
    tf_world_quad.child_frame_id = "quadrotor"
    tf_world_quad.transform.translation.x = float(self.base_pos_w[0])
    tf_world_quad.transform.translation.y = float(self.base_pos_w[1])
    tf_world_quad.transform.translation.z = float(self.base_pos_w[2])
    tf_world_quad.transform.rotation.w = float(self.base_quat_wxyz[0])
    tf_world_quad.transform.rotation.x = float(self.base_quat_wxyz[1])
    tf_world_quad.transform.rotation.y = float(self.base_quat_wxyz[2])
    tf_world_quad.transform.rotation.z = float(self.base_quat_wxyz[3])
    self.tf_broadcaster.sendTransform(tf_world_quad)

  def publish_learned_command(self, force_cmd: float, desired_rates: np.ndarray):
    msg = TRPYCommand()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "world"
    msg.pred_input[0] = float(force_cmd)
    msg.pred_input[1] = float(desired_rates[0])
    msg.pred_input[2] = float(desired_rates[1])
    msg.pred_input[3] = float(desired_rates[2])
    msg.thrust = float(force_cmd)
    msg.roll = 0.0
    msg.pitch = 0.0
    msg.yaw = 0.0
    msg.angular_velocity.x = float(desired_rates[0])
    msg.angular_velocity.y = float(desired_rates[1])
    msg.angular_velocity.z = float(desired_rates[2])
    msg.torque.x = 0.0
    msg.torque.y = 0.0
    msg.torque.z = 0.0
    msg.quaternion.w = float(self.base_quat_wxyz[0])
    msg.quaternion.x = float(self.base_quat_wxyz[1])
    msg.quaternion.y = float(self.base_quat_wxyz[2])
    msg.quaternion.z = float(self.base_quat_wxyz[3])
    msg.kr[0] = 0.0
    msg.kr[1] = 0.0
    msg.kr[2] = 0.0
    msg.kom[0] = float(self.rate_gains[0])
    msg.kom[1] = float(self.rate_gains[1])
    msg.kom[2] = float(self.rate_gains[2])
    msg.aux.current_yaw = 0.0
    msg.aux.kf_correction = 0.0
    msg.aux.angle_corrections[0] = 0.0
    msg.aux.angle_corrections[1] = 0.0
    msg.aux.enable_motors = True
    msg.aux.use_external_yaw = False
    self.publisher_policy_cmd.publish(msg)

  # Main loop.

  def run(self):
    if not self.target_initialized or self.start_time is None:
      return

    obs = self.build_observation()
    raw_action = self.policy_step(obs)
    force_cmd, desired_rates = self.action_to_command(raw_action)

    self.publish_learned_command(force_cmd, desired_rates)
    self.publish_transforms()

  def destroy_node(self):
    super().destroy_node()


def main(args=None):
  rclpy.init(args=args)
  node = QuadrotorPolicyNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    node.get_logger().info("Policy node stopped manually.")
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
