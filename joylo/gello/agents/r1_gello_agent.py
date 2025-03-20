import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
from enum import Enum

import torch as th
import numpy as np
import time

from gello.agents.joycon_agent import JoyconAgent
from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent
from gello.robots.dynamixel import DynamixelRobot
from gello.dynamixel.driver import OperatingMode, GainType


class MotorFeedbackConfig(Enum):
    """
    Enum specifying different types of force feedback, which are used in
    compute_feedback_currents
    """
    NONE = -1
    JOINT_SPACE = 0
    OPERATIONAL_SPACE = 1


class R1GelloAgent(GelloAgent):
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
        default_joints: Optional[np.ndarray] = None,
        damping_motor_kp: Optional[float] = 0.0,
        motor_feedback_type: MotorFeedbackConfig = MotorFeedbackConfig.OPERATIONAL_SPACE,
        enable_locking_joints: bool = True,
        joycon_agent: Optional[JoyconAgent] = None,
    ):
        # Create local variables
        self.arm_info = {
            "left": {
                "locked": {
                    "upper": False,
                    "lower": False,
                },
                "gello_ids": np.arange(8),
                "locked_wrist_angle": None,
                "colliding": False
            },
            "right": {
                "locked": {
                    "upper": False,
                    "lower": False,
                },
                "gello_ids": np.arange(8) + 8,
                "locked_wrist_angle": None,
                "colliding": False
            },
        }
        self.joycon_agent = joycon_agent

        self._motor_feedback_type = motor_feedback_type

        # Can only enable locking joints if we have a valid joycon agent
        self.enable_locking_joints = enable_locking_joints and self.joycon_agent is not None

        # Stores joint offsets to apply dynamically (note: including motor redundancies!)
        self.joint_offsets = np.zeros(16)

        # No feedback by default
        self.default_operation_modes = np.array([OperatingMode.NONE for _ in range(16)])

        # Run super
        super().__init__(
            port=port,
            dynamixel_config=dynamixel_config,
            start_joints=start_joints,
            default_joints=default_joints,
            damping_motor_kp=damping_motor_kp,
        )

    def _gello_joints_to_obs_joints(self, joints: np.ndarray) -> np.ndarray:
        # Filter out redundant motors
        return (joints - self.joint_offsets)[[0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15]]

    def _obs_joints_to_gello_joints(self, obs: Dict) -> np.ndarray:
        """
        Maps observed joints from the environment into equivalent GELLO joint configuration

        Args:
            obs (dict): Keyword-mapped dictionary of relevant joints obtained from the environment

        Returns:
            np.ndarray: Equivalent joint configuration in GELLO-compatible array form
        """
        # Convert [left, right] arm qpos into single array
        obs_jnts = np.concatenate([obs[f"arm_{arm}_joint_positions"].detach().cpu().numpy() for arm in ["left", "right"]])
        # Duplicate values for redundant motors
        return obs_jnts[[0, 0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 11]] + self.joint_offsets
    

    def compute_feedback_currents(self, joint_error, joint_vel, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute the feedback currents (i.e. motor torques) given the current discrepancy between
        the sim joints and gello joints

        Parameters:
            joint_error:    array of joint errors (in degrees), in GELLO format
            joint_vel:      array of joint velocities, in GELLO format
            obs:            dict of observations from the sim
        Returns:
            Array of feedback currents, one for each GELLO joint

        Overrides method defined in subclass
        """
        if self._motor_feedback_type == MotorFeedbackConfig.NONE:
            # No feedback
            current = np.zeros(joint_error.shape)

        elif self._motor_feedback_type == MotorFeedbackConfig.JOINT_SPACE:
            # Joint space feecback
            current = self._damping_motor_kp * 0.2 * (joint_error ** 2) * np.sign(joint_error) - self._damping_motor_kv * joint_vel * 0.1
        
        elif self._motor_feedback_type == MotorFeedbackConfig.OPERATIONAL_SPACE:
            # Operational space
            
            # Duplicate the columns in the jacobian to handle duplicate joints
            # Only use velocyt part as we are intereted in position offset
            J_left = obs["arm_left_jacobian"][:3, [0, 0, 1, 1, 2, 3, 4, 5]]
            J_right = obs["arm_right_jacobian"][:3, [0, 0, 1, 1, 2, 3, 4, 5]]

            # Stack jacobians into block diagonal matrix so the whole computation
            # can be performat at once
            J_shape = J_left.shape
            assert J_left.shape == J_right.shape

            J = np.block([
                [J_left, np.zeros(J_shape)],
                [np.zeros(J_shape), J_right]
            ])

            eef_error = J @ np.deg2rad(joint_error)
            current = self._damping_motor_kp * J.T @ eef_error - self._damping_motor_kv * joint_vel * 0.01
        
        else:
            raise ValueError(f"Unexpected joint feedback type: {self._motor_feedback_type}!")
        
        return current
    

    def start(self):
        super().start()

        # Set all joints to default operating modes
        self._robot.set_operating_mode(self.default_operation_modes)


    def act(self, obs: Dict) -> th.Tensor:
        # Run super first
        jnts = super().act(obs=obs)

        # Convert back to gello form
        gello_jnts = jnts.numpy()[[0, 0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 11]]

        active_operating_mode_idxs = np.array([], dtype=int)
        operating_modes = np.zeros(len(gello_jnts), dtype=int)
        active_commanded_jnt_idxs = np.array([], dtype=int)
        commanded_jnts = gello_jnts + self.joint_offsets                # We include offsets because these are the raw commands to send to GELLO

        # If we have a joycon agent, we possibly provide additional constraints to GELLO
        if self.enable_locking_joints:
            # Feedback case 1: - / + is pressed -- this will disable final motor current while locking
            # all other joints' positions to allow for positional offsetting in the final joint
            # Feedback case 2: L / R is pressed -- this will lock the final two joints of the given arm
            # while freeing all the other joints to allow for easy elbow maneuvering
            for arm, (lock_upper, lock_lower) in zip(
                    ("left", "right"),
                    (
                        (self.joycon_agent.jc_left.get_button_minus(), self.joycon_agent.jc_left.get_button_l()),
                        (self.joycon_agent.jc_right.get_button_plus(), self.joycon_agent.jc_right.get_button_r()),
                    ),
            ):
                arm_info = self.arm_info[arm]
                upper_currently_locked = arm_info["locked"]["upper"]
                wrist_id = arm_info["gello_ids"][-1]

                if self._motor_feedback_type != MotorFeedbackConfig.NONE:
                    # If using feedback, enable current control for the arms which are colliding
                    if obs[f"arm_{arm}_contact"] and not arm_info["colliding"]:
                            arm_info["colliding"] = True

                            operating_modes[arm_info["gello_ids"]] = OperatingMode.CURRENT
                            active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                    elif not obs[f"arm_{arm}_contact"] and arm_info["colliding"]:
                            arm_info["colliding"] = False

                            operating_modes[arm_info["gello_ids"]] = self.default_operation_modes[arm_info["gello_ids"]]
                            active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                if lock_upper:
                    if upper_currently_locked:
                        # Already locked, do nothing
                        pass
                    else:
                        # Just became locked, update this arm's operating mode (all joints for the arm except final
                        # two should be using POSITION mode)
                        operating_modes[arm_info["gello_ids"]] = [OperatingMode.POSITION] * 7 + [OperatingMode.NONE]
                        active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                        # In addition, the final wrist joint should NOT change in the environment -- so keep track of
                        # the current angle so we can apply an offset as long as the upper arm is locked
                        # NOTE: This value will ALREADY have any pre-existing offset applied, which is expected
                        # because the "freezing upper" effect is assumed to be cumulative
                        arm_info["locked_wrist_angle"] = gello_jnts[arm_info["gello_ids"][-1]]

                        # Add upper joint to commanded set of joint idxs
                        active_commanded_jnt_idxs = np.concatenate([active_commanded_jnt_idxs, arm_info["gello_ids"][:-1]])

                        # Finally, update our lock state
                        arm_info["locked"]["upper"] = True

                    # If we're locked, force the returned wrist value to be the locked value
                    gello_jnts[wrist_id] = arm_info["locked_wrist_angle"]

                else:
                    if not upper_currently_locked:
                        # Already not locked, do nothing
                        pass
                    else:
                        # Just became not locked, so update this arm's operating mode (all joints should be using
                        # the default mode)
                        operating_modes[arm_info["gello_ids"]] = self.default_operation_modes[arm_info["gello_ids"]]
                        active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                        # Update the joint offset to include the difference between the locked wrist and the current
                        # wrist qpos
                        additional_wrist_offset = gello_jnts[wrist_id] - arm_info["locked_wrist_angle"]
                        self.joint_offsets[wrist_id] += additional_wrist_offset

                        # This offset hasn't been applied yet to the current joints, so apply it now (negative because
                        # we're going from GELLO -> Env, whereas the offset captures the env -> GELLO delta)
                        gello_jnts[wrist_id] -= additional_wrist_offset

                        # Finally, update our lock state
                        arm_info["locked"]["upper"] = False

                lower_currently_locked = arm_info["locked"]["lower"]
                if lock_lower:
                    if lower_currently_locked:
                        # Already locked, do nothing
                        pass
                    else:
                        # Just became locked, update this arm's operating mode (final two joints should be using
                        # POSITION mode)
                        operating_modes[arm_info["gello_ids"]] = [OperatingMode.CURRENT] * 6 + [OperatingMode.POSITION] * 2
                        active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                        # Add lower joints to commanded set of joint idxs
                        active_commanded_jnt_idxs = np.concatenate([active_commanded_jnt_idxs, arm_info["gello_ids"][-2:]])

                        # Finally, update our lock state
                        arm_info["locked"]["lower"] = True

                else:
                    if not lower_currently_locked:
                        # Already not locked, do nothing
                        pass
                    else:
                        # Just became not locked, so update this arm's operating mode (all joints should be using
                        # the default mode)
                        operating_modes[arm_info["gello_ids"]] = self.default_operation_modes[arm_info["gello_ids"]]
                        active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                        # Finally, update our lock state
                        arm_info["locked"]["lower"] = False

                
                    


            # Update our operating mode if requested
            if len(active_operating_mode_idxs) > 0:
                self._robot.set_operating_mode(operating_modes[active_operating_mode_idxs], idxs=active_operating_mode_idxs)

            # Command joints if requested
            if len(active_commanded_jnt_idxs) > 0:
                self._robot.command_joint_state(commanded_jnts[active_commanded_jnt_idxs], idxs=active_commanded_jnt_idxs)

        return th.from_numpy(gello_jnts[[0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15]])
