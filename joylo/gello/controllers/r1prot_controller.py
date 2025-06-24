# import logging
# import threading
# import torch
# from gello.controllers.controller_base import BaseController
# from omnigibson.robots import R1
# from time import time

# logger = logging.getLogger("R1ProT Controller")


# try:
#     import rclpy
# except ImportError:
#     # use system libstdc++ instead of the one in the conda env
#     raise ImportError(
#         """
#         rclpy cannot be imported. Make sure you have sourced the ros2 setup script and set the LD_PRELOAD path:
#         source /opt/ros/humble/setup.bash
#         export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
#         """
#     )
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import TwistStamped


# class R1ProTController(BaseController):
#     """
#     A class to control the R1ProT robot using ROS2.
#     """

#     def __init__(self, robot, *args, **kwargs):
#         """
#         Initialize the R1ProT controller.
#         """
#         super().__init__(robot)

#         self._stopped = False  # whether the controller is stopped

#         # initialize locks
#         self.left_arm_lock = threading.Lock()
#         self.right_arm_lock = threading.Lock()
#         self.left_gripper_lock = threading.Lock()
#         self.right_gripper_lock = threading.Lock()
#         self.base_lock = threading.Lock()
#         self.torso_lock = threading.Lock()

#         # initialize command variables
#         self.left_arm_cmd = None
#         self.right_arm_cmd = None
#         self.left_gripper_cmd = None
#         self.right_gripper_cmd = None
#         self.base_cmd = None
#         self.torso_cmd = None

#         rclpy.init()
#         self.node = rclpy.create_node('r1prot_controller')
#         # leader arm jont pos
#         self.node.create_subscription(JointState, "/motion_target/target_joint_state_arm_left", self.cb_left_arm_control_command, 10)
#         self.node.create_subscription(JointState, "/motion_target/target_joint_state_arm_right", self.cb_right_arm_control_command, 10)
#         # leader arm gripper pos
#         self.node.create_subscription(JointState, "/motion_target/target_position_gripper_left", self.cb_left_gripper_control_command, 10)
#         self.node.create_subscription(JointState, "/motion_target/target_position_gripper_right", self.cb_right_gripper_control_command, 10)
#         # leader arm base and torso speed
#         self.node.create_subscription(TwistStamped, "/motion_target/target_speed_chassis", self.cb_base_control_command, 10)
#         self.node.create_subscription(TwistStamped, "/motion_target/target_speed_torso", self.cb_torso_control_command, 10)

#         # follower arm jont pos
#         self.left_arm_status_publisher_ = self.node.create_publisher(JointState, '/hdas/feedback_arm_left', 10)
#         self.right_arm_status_publisher_ = self.node.create_publisher(JointState, '/hdas/feedback_arm_right', 10)
#         self.torso_status_publisher_ = self.node.create_publisher(JointState, '/hdas/feedback_torso', 10)

#     def start(self):
#         """
#         Start the controller.
#         """
#         if self._stopped:
#             raise RuntimeError("Controller is already stopped.")
#         self.ros_spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,))
#         self.ros_spin_thread.start()
#         logger.info("[START] R1ProT controller started.")

#     def update_observations(self, joint_pos: torch.Tensor):
#         self.current_left_arm_pos = joint_pos[self.robot.arm_control_idx["left"]]
#         self.current_right_arm_pos = joint_pos[self.robot.arm_control_idx["right"]]
#         trunk_joint_pos = joint_pos[self.robot.trunk_control_idx]
#         self._pub_robot_joint_states(trunk_joint_pos, self.current_left_arm_pos, self.current_right_arm_pos)

#     def is_running(self) -> bool:
#         """
#         Check if the controller is running.

#         Returns:
#             bool: True if the controller is running, False otherwise.
#         """
#         return not self._stopped and rclpy.ok()

#     def get_action(self):
#         # Start an empty action
#         action = torch.zeros(self.robot.action_dim)

#         # Apply arm action + extra dimension from base.
#         if isinstance(self.robot, R1):
#             # Apply arm action
#             if(self.current_left_arm_pos is None or self.current_left_arm_pos.dim()!= 1 or self.current_left_arm_pos.numel() != 7
#                 or self.current_right_arm_pos is None or self.current_right_arm_pos.dim()!= 1 or self.current_right_arm_pos.numel() != 7):
#                 if self.current_left_arm_pos is None:
#                     print("❌ current_left_arm_pos is None")
#                 else:
#                     print(f"ℹ️ current_left_arm_pos: dim={self.current_left_arm_pos.dim()}, shape={tuple(self.current_left_arm_pos.shape)}, numel={self.current_left_arm_pos.numel()} (expected dim=1 and numel=7)")

#                 if self.current_right_arm_pos is None:
#                     print("❌ current_right_arm_pos is None")
#                 else:
#                     print(f"ℹ️ current_right_arm_pos: dim={self.current_right_arm_pos.dim()}, shape={tuple(self.current_right_arm_pos.shape)}, numel={self.current_right_arm_pos.numel()} (expected dim=1 and numel=7)")
#                 return action

#             with self.left_arm_lock:
#                 left_arm_control_cmd = self.current_left_arm_pos if self.left_arm_cmd is None else self.left_arm_cmd
#             with self.right_arm_lock:
#                 right_arm_control_cmd = self.current_right_arm_pos if self.right_arm_cmd is None else self.right_arm_cmd
#             action[self.robot.arm_action_idx["left"]] = left_arm_control_cmd
#             action[self.robot.arm_action_idx["right"]] = right_arm_control_cmd

#             with self.base_lock:
#                 ros_time = self.node.get_clock().now()
#                 current_time = ros_time.nanoseconds / 1e9
#                 if self.base_cmd is None:
#                     base_control_cmd = [0.0, 0.0, 0.0]
#                 else:
#                     vx, vy, wz, cmd_time = self.base_cmd
#                     print(f"cmd_time: {cmd_time:.9f} 秒")
#                     print(f" ✅ base cmd deltaT: {(current_time - cmd_time):.9f}")
#                     if current_time - cmd_time > 0.2:
#                         base_control_cmd = [0.0, 0.0, 0.0]
#                     else:
#                         base_control_cmd = [vx, vy, wz]
#             action[self.robot.base_action_idx] = torch.tensor(base_control_cmd, dtype=torch.float32)

#             with self.right_gripper_lock:    
#                 right_control_cmd = [0] if self.right_gripper_cmd is None else [self.right_gripper_cmd[0]]
#             with self.left_gripper_lock:    
#                 left_gripper_cmd = [0] if self.left_gripper_cmd is None else [self.left_gripper_cmd[0]]
#             # Apply gripper action
#             action[self.robot.gripper_action_idx["left"]] = torch.tensor([left_gripper_cmd[0]], dtype=torch.float32)
#             action[self.robot.gripper_action_idx["right"]] = torch.tensor([right_control_cmd[0]], dtype=torch.float32)

#             # Apply trunk action
#             if SIMPLIFIED_TRUNK_CONTROL:
#                 with self.torso_lock:    
#                     current_time = time.time()  # 当前时间（单位：秒）
                    
#                     if self.torso_cmd is None:
#                         torso_control_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#                     else:
#                         vx, vy, vz, wx, wy, wz, cmd_time = self.torso_cmd
#                         # print(f"torso cmd deltaT: {(current_time - cmd_time):.9f}")
#                         if current_time - cmd_time > 0.2:
#                             torso_control_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#                         else:
#                             torso_control_cmd = [vx, vy, vz, wx, wy, wz]

#                 # self._current_trunk_translate = np.clip(self._current_trunk_translate + self._joint_cmd["trunk"][0].item() * og.sim.get_sim_step_dt(), 0.25, 0.65)
#                 # self._current_trunk_tilt = np.clip(self._current_trunk_tilt + self._joint_cmd["trunk"][1].item() * og.sim.get_sim_step_dt(), -np.pi / 2, np.pi / 2)
#                 self._current_trunk_translate = np.clip(self._current_trunk_translate + torso_control_cmd[2] * og.sim.get_sim_step_dt(), 0.25, 0.65)
#                 self._current_trunk_tilt = np.clip(self._current_trunk_tilt + torso_control_cmd[4] * og.sim.get_sim_step_dt(), -np.pi / 2, np.pi / 2)
#                 # print(f" _current_trunk_translate { self._current_trunk_translate}")
#                 # print(f" _current_trunk_tilt { self._current_trunk_tilt}")
#                 # print(f" control trunk { torso_control_cmd}")
#                 # Convert desired values into corresponding trunk joint positions
#                 # Trunk link 1 is 0.4m, link 2 is 0.3m
#                 # See https://www.mathworks.com/help/symbolic/derive-and-apply-inverse-kinematics-to-robot-arm.html
#                 xe, ye = self._current_trunk_translate, 0.1
#                 sol_sign = 1.0      # or -1.0
#                 # xe, ye = 0.5, 0.1
#                 l1, l2 = 0.4, 0.3
#                 xe2 = xe**2
#                 ye2 = ye**2
#                 xeye = xe2 + ye2
#                 l12 = l1**2
#                 l22 = l2**2
#                 l1l2 = l12 + l22
#                 # test = -(l12**2) + 2*l12*l22 + 2*l12*(xe2+ye2) - l22**2 + 2*l22*(xe2+ye2) - xe2**2 - 2*xe2*ye2 - ye2**2
#                 sigma1 = np.sqrt(-(l12**2) + 2*l12*l22 + 2*l12*xe2 + 2*l12*ye2 - l22**2 + 2*l22*xe2 + 2*l22*ye2 - xe2**2 - 2*xe2*ye2 - ye2**2)
#                 theta1 = 2 * np.arctan2(2*l1*ye + sol_sign * sigma1, l1**2 + 2*l1*l2 - l2**2 + xeye)
#                 theta2 = -sol_sign * 2 * np.arctan2(np.sqrt(l1l2 - xeye + 2*l1*l2), np.sqrt(-l1l2 + xeye + 2*l1*l2))
#                 theta3 = (theta1 + theta2 - self._current_trunk_tilt)
#                 theta4 = 0.0

#                 action[self.robot.trunk_action_idx] = th.tensor([theta1, theta2, theta3, theta4], dtype=th.float)
#             else:
#                 self._current_trunk_translate = float(th.clamp(
#                     th.tensor(self._current_trunk_translate, dtype=th.float) - th.tensor(self._joint_cmd["trunk"][0].item() * og.sim.get_sim_step_dt(), dtype=th.float),
#                     0.0,
#                     2.0
#                 ))

#                 # Interpolate between the three pre-determined joint positions
#                 if self._current_trunk_translate <= 1.0:
#                     # Interpolate between upright and down positions
#                     interpolation_factor = self._current_trunk_translate
#                     interpolated_trunk_pos = (1 - interpolation_factor) * R1_UPRIGHT_TORSO_JOINT_POS + \
#                                             interpolation_factor * R1_DOWNWARD_TORSO_JOINT_POS
#                 else:
#                     # Interpolate between down and ground positions
#                     interpolation_factor = self._current_trunk_translate - 1.0
#                     interpolated_trunk_pos = (1 - interpolation_factor) * R1_DOWNWARD_TORSO_JOINT_POS + \
#                                             interpolation_factor * R1_GROUND_TORSO_JOINT_POS

#                 action[self.robot.trunk_action_idx] = interpolated_trunk_pos

#     def get_base_cmd(self):
#         # TODO
#         pass

#     def get_button_input_cmd(self):
#         # TODO
#         pass

#     def _pub_robot_joint_states(self, trunk_joint_pos, left_arm_pos, right_arm_pos):
#         now = self.node.get_clock().now().to_msg()
#         # === Trunk ===
#         trunk_state = JointState()
#         trunk_state.header.stamp = now
#         trunk_state.name = ["torso_joint_1", "torso_joint_2", "torso_joint_3", "torso_joint_4"]
#         trunk_state.position = trunk_joint_pos.tolist() if hasattr(trunk_joint_pos, 'tolist') else list(trunk_joint_pos)
#         trunk_state.velocity = [0.0, 0.0, 0.0, 0.0]
#         trunk_state.effort = [0.0, 0.0, 0.0, 0.0]
#         self.torso_status_publisher_.publish(trunk_state)
#         # === Left Arm ===
#         left_state = JointState()
#         left_state.header.stamp = now
#         left_state.name = [f"left_arm_joint_{i+1}" for i in range(len(left_arm_pos))]
#         left_state.position = left_arm_pos.tolist() if hasattr(left_arm_pos, 'tolist') else list(left_arm_pos)
#         left_state.velocity = [0.0] * len(left_state.name)
#         left_state.effort = [0.0] * len(left_state.name)
#         self.left_arm_status_publisher_.publish(left_state)
#         # === Right Arm ===
#         right_state = JointState()
#         right_state.header.stamp = now
#         right_state.name = [f"right_arm_joint_{i+1}" for i in range(len(right_arm_pos))]
#         right_state.position = right_arm_pos.tolist() if hasattr(right_arm_pos, 'tolist') else list(right_arm_pos)
#         right_state.velocity = [0.0] * len(right_state.name)
#         right_state.effort = [0.0] * len(right_state.name)
#         self.right_arm_status_publisher_.publish(right_state)

#     # === callback functions ===
#     def cb_left_arm_control_command(self,  msg: JointState):
#         joint_pos = msg.position
#         assert len(joint_pos) == 7, "Left arm joint position should be of length 7"
#         with self.left_arm_lock:
#             self.left_arm_cmd = torch.tensor(joint_pos, dtype=torch.float32)

#     def cb_right_arm_control_command(self, msg: JointState):
#         joint_pos = msg.position
#         assert len(joint_pos) == 7, "Right arm joint position should be of length 7"
#         with self.right_arm_lock:
#             self.right_arm_cmd = torch.tensor(joint_pos, dtype=torch.float32)

#     def cb_left_gripper_control_command(self, msg):
#         joint_pos = msg.position
#         with self.left_gripper_lock:    
#             raw = max(0.0, min(100.0, joint_pos[0]))
#             self.left_gripper_cmd = [2.0 * (raw / 100.0) - 1.0]

#     def cb_right_gripper_control_command(self, msg):
#         joint_pos = msg.position
#         with self.right_gripper_lock:    
#             raw = max(0.0, min(100.0, joint_pos[0]))
#             self.right_gripper_cmd = [2.0 * (raw / 100.0) - 1.0]

#     def cb_base_control_command(self, msg):
#         with self.base_lock:    
#             stamp = time()
#             self.base_cmd = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z, stamp]

#     def cb_torso_control_command(self, msg):
#         with self.torso_lock:    
#             stamp = time()
#             self.torso_cmd = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z, msg.twist.angular.x , msg.twist.angular.y, msg.twist.angular.z, stamp]

#     def stop(self):
#         if self._stopped:
#             return
#         self._stopped = True
#         try:
#             if rclpy.ok():
#                 rclpy.shutdown()
#         except Exception as e:
#             logger.error(f"[STOP] rclpy.shutdown() exception: {e}")

#         try:
#             if self.node:
#                 self.node.destroy_node()
#         except Exception as e:
#             logger.error(f"[STOP] destroy_node exception: {e}")

#         if self.ros_spin_thread.is_alive():
#             self.ros_spin_thread.join(timeout=5)
#             logger.info("[STOP] ROS spin exited")