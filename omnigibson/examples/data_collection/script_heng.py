import argparse
import math
import time

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects import PrimitiveObject
from omnigibson.robots import Fetch
from omnigibson.scenes import Scene
from omnigibson.utils.control_utils import IKSolver


def euler_to_quaternion(roll, pitch, yaw):
    """ """
    roll, pitch, yaw = map(math.radians, [roll, pitch, yaw])

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return th.tensor([w, x, y, z], dtype=th.float32)


def quat_mul(q1, q2):
    """ """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return th.tensor([w, x, y, z])


def generate_action(robot, arm_targets):
    """
    Generate a no-op action that will keep the robot still but aim to move the arms to the saved pose targets, if possible

    Returns:
        th.tensor or None: Action array for one step for the robot to do nothing
    """
    action = th.zeros(robot.action_dim)
    for name, controller in robot._controllers.items():
        # if desired arm targets are available, generate an action that moves the arms to the saved pose targets
        if name in arm_targets:
            arm = name.replace("arm_", "")
            target_pos, target_orn_axisangle = arm_targets[name]
            current_pos = robot.get_eef_position(arm)
            delta_pos = target_pos - current_pos
            partial_action = th.cat((delta_pos, target_orn_axisangle))
        else:
            partial_action = controller.compute_no_op_action(robot.get_control_dict())
        action_idx = robot.controller_action_idx[name]
        action[action_idx] = partial_action
    return action


def main():

    scene_cfg = {"type": "Scene"}
    left_robot_cfg = {
        "type": "FrankaPanda",
        "fixed_base": True,
        # "end_effector": "inspire",
        "self_collision": False,
        "action_normalize": True,
        "action_type": "continuous",
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "mode": "pose_absolute_ori",
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "mode": "independent",
            },
        },
    }
    right_robot_cfg = {
        "type": "FrankaPanda",
        "fixed_base": True,
        "self_collision": False,
        "action_normalize": True,
        "action_type": "continuous",
        # "end_effector": "inspire",
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "mode": "pose_absolute_ori",
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "mode": "independent",
            },
        },
    }

    cfg = dict(scene=scene_cfg, robots=[left_robot_cfg, right_robot_cfg])
    # cfg = dict(scene=scene_cfg, robots=[left_robot_cfg])
    env = og.Environment(configs=cfg)

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.7038, -0.8032, 1.2708]),
        orientation=th.tensor([0.4872, 0.2458, 0.3775, 0.7482]),
    )

    robot = env.robots[0]
    right_robot = env.robots[1]

    # Set robot base at the origin
    robot.set_position_orientation(position=th.tensor([0.0, 0, 0]), orientation=th.tensor([0, 0, 0, 1]))
    right_robot.set_position_orientation(position=th.tensor([0.0, 1.0, 0]), orientation=th.tensor([0, 0, 0, 1]))

    # At least one simulation step while the simulator is playing must occur for the robot (or in general, any object)
    # to be fully initialized after it is imported into the simulator
    # og.sim.play()
    og.sim.step()
    # Make sure none of the joints are moving
    robot.keep_still()
    right_robot.keep_still()

    l_robot_eef_pos = robot.get_eef_position()
    l_robot_eef_ori = T.quat2axisangle(robot.get_eef_orientation())
    r_robot_eef_pos = right_robot.get_eef_position()
    r_robot_eef_ori = T.quat2axisangle(right_robot.get_eef_orientation())

    l_robot_target_pos = l_robot_eef_pos + th.tensor([0.1, 0.0, 0.0])
    r_robot_target_pos = r_robot_eef_pos + th.tensor([0.1, 0.0, 0.0])

    reached_target = False

    left_arm_targets = {"arm_0": (l_robot_target_pos, l_robot_eef_ori)}
    right_arm_targets = {"arm_0": (r_robot_target_pos, r_robot_eef_ori)}

    while not reached_target:
        left_action = generate_action(robot, left_arm_targets)
        right_action = generate_action(right_robot, right_arm_targets)

        env.step(th.cat([left_action, right_action]))

        # Tweak this tolerance
        reached_target = (
            th.isclose(l_robot_eef_pos, l_robot_target_pos, atol=0.05).all()
            and th.isclose(r_robot_eef_pos, r_robot_target_pos, atol=0.05).all()
        )

    # Always shut the simulation down cleanly at the end
    og.clear()


if __name__ == "__main__":
    main()
