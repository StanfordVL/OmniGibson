import os
import time
import math
import pickle
import imageio
import yaml
import omnigibson as og
import torch as th
th.set_printoptions(precision=3, sci_mode=False)
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
import omnigibson.lazy as lazy
from omnigibson import object_states


seed = 5
np.random.seed(seed)
th.manual_seed(seed)

RESOLUTION = [1080, 1080]  # [H, W]
# RESOLUTION = [196, 320]  # [H, W]

env_cfg = {
    "env": {
        "action_frequency": 30,
        "physics_frequency": 120,
        "external_sensors": [
            {
                "sensor_type": "VisionSensor",
                "name": "external_sensor0",
                "relative_prim_path": "/controllable__r1__robot0/base_link/external_sensor0",
                "modalities": [],
                "sensor_kwargs": {
                    "viewport_name": "Viewport",
                    "image_height": RESOLUTION[0],
                    "image_width": RESOLUTION[1],
                },
                "position": [-0.4, 0, 2.0],  # [-0.74, 0, 2.0],
                "orientation": [0.369, -0.369, -0.603, 0.603],
                "pose_frame": "parent",
                "include_in_obs": False,
            },
        ],
    },
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "house_single_floor",
        # "load_room_instances": ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"],
        "load_room_instances": ["kitchen_0", "dining_room_0"],
        "not_load_object_categories": ["taboret"],
    },
    "objects": [
        {
            "type": "DatasetObject",
            "name": "plate_1",
            "category": "plate",
            "model": "luhkiz",
            "position": [6.4349, -1.9089,  0.9457],
        },
        {
            "type": "DatasetObject",
            "name": "plate_2",
            "category": "plate",
            "model": "pkkgzc",
            "position": [ 6.6578, -1.7081,  0.9472],
        },
        {
            "type": "DatasetObject",
            "name": "bowl_1",
            "category": "bowl",
            "model": "mspdar",
            "position": [6.9947, -1.7043,  0.9626],
        },
        {
            "type": "DatasetObject",
            "name": "sponge",
            "category": "sponge",
            "model": "aewrov",
            "position": [6.067, -2.18, 1.16],
        },
        {
            "type": "DatasetObject",
            "name": "scrub_brush",
            "category": "scrub_brush",
            "model": "hsejyi",
            "position": [5.96, -2.12, 1.16],
            "scale":[2.5, 1., 1.0],
            "orientation": T.euler2quat(th.tensor([0, -math.pi * 0.5, 0])),
        },
        {
            "type": "DatasetObject",
            "name": "saucepot",
            "category": "saucepot",
            "model": "wfryvm", # "uvzmss", "chjetk"
            "position": [4.2538, -1.3665,  1.0286],
            "orientation": T.euler2quat(th.tensor([0, 0, math.pi * 0.5])),
        },
        {
            "type": "DatasetObject",
            "name": "chestnut_1",
            "category": "chestnut",
            "model": "gjbnba",
            "position": [4.2538, -1.37,  1.0286],
        },
        {
            "type": "DatasetObject",
            "name": "chestnut_2",
            "category": "chestnut",
            "model": "gjbnba",
            "position": [4.2538, -1.4,  1.0286],
        },
        {
            "type": "DatasetObject",
            "name": "chestnut_3",
            "category": "chestnut",
            "model": "gjbnba",
            "position": [4.2538, -1.43,  1.0286],
        },
        {
            "type": "DatasetObject",
            "name": "spatula",
            "category": "spatula",
            "model": "crkmux",
            "position": [4.2299, 0.27, 1.1422],
            "orientation": T.euler2quat(th.tensor([math.pi * 0.5,0, 0])),
        },
        {
            "type": "DatasetObject",
            "name": "desk_organizer",
            "category": "desk_organizer",
            "model": "yidrel",
            "scale": [1.5, 1.5, 1.5],
            "position": [4.2299, 0.3772, 1.0422],
        },
        {
            "type": "DatasetObject", 
            "name": "teacup", 
            "category": "teacup", 
            "model": "cpozxi", 
            "scale": [1.0, 1.0, 1.0], 
            "position": [1.5, 0.15, 0.8], 
            "orientation": [0.0, 0.0, -0.7071067690849304, 0.7071067690849304]
         }, 
         {
            "type": "DatasetObject", 
            "name": "coffee_cup", 
            "category": "coffee_cup", 
            "model": "dkxddg", 
            "scale": [2.0, 2.0, 1.5], 
            "position": [1.55, -0.2, 0.8], 
            "orientation": [-0.0, 0.0, 0.7071067690849304, -0.7071067690849304]
        },
    ],
    "robots": [
        {
            "type": "R1",
            # "position": [9.0, 1.5,  1.0286],   # [5.2, -.8,  1.0286]
            # "orientation": [    -0.0000,      0.0000,      0.8734,     -0.4870],
            "name": "robot0",
            "action_normalize": False,
            "self_collisions": False,
            "obs_modalities": [],
            "sensor_config": {
                "VisionSensor": {
                    "sensor_kwargs": {
                        "image_height": RESOLUTION[0],
                        "image_width": RESOLUTION[1],
                    },
                },
            },
            "reset_joint_pos": [
                0.0000,
                0.0000,
                0.000,
                0.000,
                0.000,
                -0.0000, # 6 virtual base joint 
                0.5,
                -1.0,
                -0.8,
                -0.0000, # 4 torso joints
                -0.000,
                0.000,
                1.8944,
                1.8945,
                -0.9848,
                -0.9849,
                1.5612,
                1.5621,
                0.9097,
                0.9096,
                -1.5544,
                -1.5545,
                0.0500,
                0.0500,
                0.0500,
                0.0500,
            ],
        }
    ]
}

env = og.Environment(configs=env_cfg)

controller_config = {
    "base": {"name": "HolonomicBaseJointController", "motor_type": "position", "command_input_limits": None, "use_impedances": False},
    "trunk": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "arm_left": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "arm_right": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "gripper_left": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
    "gripper_right": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
    "camera": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
}

env.robots[0].reload_controllers(controller_config=controller_config)
# env.robots[0]._grasping_mode = "sticky"
robot = env.robots[0]
primitive = StarterSemanticActionPrimitives(env, env.robots[0], enable_head_tracking=False, curobo_batch_size=10)

# og.sim.viewer_camera.set_position_orientation(th.tensor([16.066,  3.745,  2.556]), th.tensor([0.256, 0.515, 0.732, 0.364]))

bar = env.scene.object_registry("name", "bar_udatjt_0")
teacup = env.scene.object_registry("name", "teacup")
coffee_cup = env.scene.object_registry("name", "coffee_cup")
teacup.states[object_states.OnTop].set_value(other=bar, new_value=True)
coffee_cup.states[object_states.OnTop].set_value(other=bar, new_value=True)
primitive._tracking_object = teacup

# bar_current_scale = bar.scale
# z_scale = 0.75
# # z_scale = np.random.uniform(0.8, 1.2)
# temp_state = og.sim.dump_state(serialized=False)
# og.sim.stop()
# bar.scale = th.tensor([bar_current_scale[0], bar_current_scale[1], 1.0 * z_scale])
# og.sim.play()
# og.sim.load_state(temp_state)
# bar.keep_still()
# for _ in range(10): og.sim.step()

robot.set_position_orientation(position=th.tensor([9.0, 1.5,  1.0286]), orientation=th.tensor([-0.0000, 0.0000, 0.8734, -0.4870]))
# bar.set_position_orientation(position=th.tensor([7.287, 0.189, 0.40]))
# robot.set_joint_positions(th.tensor([1.6499,  3.0020, -2.5360,  1.5709,  0.9149, -1.5714]), indices=robot.arm_control_idx["left"])
# robot.set_joint_positions(th.tensor([-1.6499,  3.0020, -2.5360,  1.5709,  0.9149, -1.5714]), indices=robot.arm_control_idx["right"])

# # hands facing up
# robot.set_joint_positions(th.tensor([-0.7664,  1.9036, -0.9915,  1.5709,  0.9161, -1.5703]), indices=robot.arm_control_idx["left"])
# robot.set_joint_positions(th.tensor([0.7664,  1.9036, -0.9915,  1.5709,  0.9161, -1.5703]), indices=robot.arm_control_idx["right"])

# top-down hands
# robot.set_joint_positions(th.tensor([-0.141, 2.248, -0.983,  0.227, 1.460, -1.230]), indices=robot.arm_control_idx["left"])
# robot.set_joint_positions( th.tensor([0.027, 2.550, -1.416,  -0.072, -1.417, 1.214]), indices=robot.arm_control_idx["right"])

# trial
robot.set_joint_positions(th.tensor([-0.3681,  1.2081, -0.2686,  1.5397,  0.9159, -1.5726]), indices=robot.arm_control_idx["left"])
robot.set_joint_positions( th.tensor([0.3681,  1.2081, -0.2686,  1.5397,  0.9159, -1.5726]), indices=robot.arm_control_idx["right"])
robot.reset_joint_pos = th.tensor([
                0.0000,
                0.0000,
                0.000,
                0.000,
                0.000,
                -0.0000, # 6 virtual base joint 
                0.5,
                -1.0,
                -0.8,
                -0.0000, # 4 torso joints
                -0.3681,
                0.3681,
                1.2081,
                1.2081,
                -0.2686,
                -0.2686,
                1.5397,
                1.5397,
                0.9159,
                0.9159,
                -1.5726,
                -1.5726,
                0.0500,
                0.0500,
                0.0500,
                0.0500,
            ],)

for _ in range(100): og.sim.step()
# breakpoint()

base_sampling_failures, base_mp_ik_failures, base_mp_trajopt_failures, succ = 0, 0, 0, 0
init_state = og.sim.dump_state()
for i in range(20):
    print(f"===================== {i} =====================")
    teacup.states[object_states.OnTop].set_value(other=bar, new_value=True)
    coffee_cup.states[object_states.OnTop].set_value(other=bar, new_value=True)

    for _ in range(50): og.sim.step()

    action_generator = primitive._navigate_to_obj(obj=teacup, visibility_constraint=True)
    
    retval = next(iter(action_generator))

    # for mp_action in action_generator:
    #     if mp_action is None:
    #         break
        
    #     mp_action = mp_action.cpu().numpy()
    #     obs, _, _, _, info = env.step(mp_action)
    
    print("primitive.mp_err: ", primitive.mp_err)
    
    if primitive.mp_err == "BaseSamplingFailed":
        base_sampling_failures += 1
    elif primitive.mp_err == "BaseMPIKFailed":
        base_mp_ik_failures += 1
    elif primitive.mp_err == "BaseMPFailed":
        base_mp_trajopt_failures += 1
    elif primitive.mp_err == "None":
        succ += 1

    # for _ in range(50): og.sim.step()
    # breakpoint()
    # og.sim.load_state(init_state)
    # for _ in range(50): og.sim.step()

print("base_sampling_failures, base_mp_ik_failures, base_mp_trajopt_failures, succ: ", base_sampling_failures, base_mp_ik_failures, base_mp_trajopt_failures, succ)

breakpoint()



# with visibility constraint: 4,5,0
# without visibility constraint: 0,0,1
# visibility constraint used now
# with regular hands: 3 16 0 1
# with hands on side: 6 4 1 9
# with regular hands + default mp: 3 7 2
# regular hands + lower table: 1 2 0 17
# new hands + modified eyes pos sampling range: 0 4 4 12
# new hands + modified eyes pos sampling range + collision check after sampling: 2 0 0 18

# Changes
# 1. Start pose of the arm
# 2. Change the eyes position sampling range (include positions close to the canonical pos)

# Try:
# Good home hand pose (side or something else)
# changing to side hands, leads to more base sampling failures (mainly failing at IK in "arm" mode I think) 
# So, try original hand pose + default mode for base mp 
# lower table height

# remove collision checking for ik when doing base mp -> This helped, explaining why IK was failing during MP
# warmup