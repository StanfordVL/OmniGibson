import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuroboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm, macros
from omnigibson.object_states import Touching
from omnigibson.objects import PrimitiveObject
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.robots.tiago import Tiago


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


cfg = {
    "env": {
        "action_frequency": 30,
        "physics_frequency": 300,
    },
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "load_object_categories": ["floors", "breakfast_table", "bottom_cabinet"],
    },
    "objects": [
        {
            "type": "DatasetObject",
            "name": "cologne",
            "category": "bottle_of_cologne",
            "model": "lyipur",
            "position": [1.1629, 0.0040, 0.82],
            "orientation": [0, 0, 0, 1],
        },
        # {
        #     "type": "DatasetObject",
        #     "name": "apple",
        #     "category": "apple",
        #     "model": "agveuv",
        #     "position": [1.2, 0.0040, 0.8],
        #     "orientation": [0, 0, 0, 1],
        # },
    ],
    "robots": [
        {
            "type": "Tiago",
            "obs_modalities": "rgb",
            "position": [0, 0, 0],
            "orientation": [0, 0, 0, 1],
            "self_collisions": True,
            "action_normalize": False,
            "rigid_trunk": False,
            "grasping_mode": "sticky",
            "reset_joint_pos": [
                0.0000,
                0.0000,
                -0.0000,
                -0.0000,
                -0.0000,
                -0.0000,
                0.3500,
                0.9052,
                0.9052,
                0.0000,
                -0.4281,
                -0.4281,
                -0.4500,
                2.2350,
                2.2350,
                1.6463,
                1.6463,
                0.7687,
                0.7687,
                -0.7946,
                -0.7946,
                -1.0891,
                -1.0891,
                0.0450,
                0.0450,
                0.0450,
                0.0450,
            ],
            "controller_config": {
                "base": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 200.0,
                    "pos_ki": 0.5,
                },
                "trunk": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 500.0,
                },
                "arm_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 500.0,
                    "pos_ki": 0.5,
                    "max_integral_error": 10.0,
                },
                "arm_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 500.0,
                    "pos_ki": 0.5,
                    "max_integral_error": 10.0,
                },
                "gripper_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 200.0,
                },
                "gripper_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 200.0,
                },
            },
        },
        # {
        #     "type": "R1",
        #     "obs_modalities": "rgb",
        #     "position": [0, 0, 0],
        #     "orientation": [0, 0, 0, 1],
        #     "self_collisions": True,
        #     "action_normalize": False,
        #     "rigid_trunk": False,
        #     "grasping_mode": "sticky",
        #     "controller_config": {
        #         "base": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": True,
        #             "pos_kp": 50.0,
        #         },
        #         "trunk": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": True,
        #         },
        #         "arm_left": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": True,
        #         },
        #         "arm_right": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": True,
        #         },
        #         "gripper_left": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": True,
        #         },
        #         "gripper_right": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": True,
        #         },
        #     },
        # }
    ],
}

env = og.Environment(configs=cfg)
robot = env.robots[0]

marker = PrimitiveObject(
    relative_prim_path=f"/marker",
    name="marker",
    primitive_type="Cone",
    scale=0.05,
    # radius=0.03,
    visual_only=True,
    rgba=[1.0, 0, 1.0, 0.5],
    position=[100.0, 100.0, 0.0],
    orientation=[0, 0, 0, 1],
)
env.scene.add_object(marker)

# og.sim.viewer_camera.set_position_orientation([-1.6840, 1.2508, 1.5873], [-0.2597, 0.5574, 0.7147, -0.3332])

# Open the gripper(s) to match cuRobo's default state
for arm_name in robot.gripper_control_idx.keys():
    grpiper_control_idx = robot.gripper_control_idx[arm_name]
    robot.set_joint_positions(th.ones_like(grpiper_control_idx), indices=grpiper_control_idx, normalized=True)
robot.keep_still()

for _ in range(5):
    og.sim.step()

env.scene.update_initial_state()
env.scene.reset()

# Let the object settle
for _ in range(30):
    og.sim.step()

# from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
# env = DataCollectionWrapper(env=env, output_path="/home/yhang/OmniGibson/R1_demo_grasp1.hdf5", only_successes=False)

action_primitives = StarterSemanticActionPrimitives(
    robot, enable_head_tracking=isinstance(robot, Tiago), planning_batch_size=1, debug_visual_marker=marker
)

# FULL PRIMITIVE TEST
grasp_obj = env.scene.object_registry("name", "cologne")
# Grasp apple
print("Executing controller")

# TODO: here is a trick here, the apply_ref is veryhard to debug, can directly use the following code to debug
execute_controller(action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj), env)

# for action in action_primitives._move_hand(grasp_pose):
#     env.step(action)
print("Finished executing grasp")

# env.save_data()

# cabinet = env.scene.object_registry("name", "bottom_cabinet_jhymlr_0")
# execute_controller(action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, cabinet), env)


print("done")

og.shutdown()
