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


seed = 0
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
        "load_room_instances": ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"],
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


# og.clear()
# og.sim.restore(scene_files=["/home/arpit/test_projects/OmniGibson/custom_scenes/house_single_floor_task_datagen_tidy_table_0_0_template.json"])
# breakpoint()

# bar = env.scene.object_registry("name", "bar_udatjt_0")
# teacup = env.scene.object_registry("name", "teacup")
# coffee_cup = env.scene.object_registry("name", "coffee_cup")
# teacup.states[object_states.OnTop].set_value(other=bar, new_value=True)
# coffee_cup.states[object_states.OnTop].set_value(other=bar, new_value=True)
# sink = env.scene.object_registry("name", "drop_in_sink_awvzkn_0")

breakpoint()

for _ in range(100): og.sim.step()

# env.robots[0].set_position_orientation(position=th.tensor([9.0, 1.5,  1.0286]), orientation=th.tensor([-0.0000, 0.0000, 0.8734, -0.4870]))

# for _ in range(100): og.sim.step()

# curr_pose = robot.get_position_orientation()
action_generator = primitive._navigate_to_obj(obj=sink)

for mp_action in action_generator:
    if mp_action is None:
        break
    
    mp_action = mp_action.cpu().numpy()
    obs, _, _, _, info = env.step(mp_action)

breakpoint()

# taboret_1 = env.scene.object_registry("name", "taboret_ivtoxu_1")
# taboret_2 = env.scene.object_registry("name", "taboret_ivtoxu_2")
# taboret_3 = env.scene.object_registry("name", "taboret_ivtoxu_3")
# taboret_4 = env.scene.object_registry("name", "taboret_ivtoxu_4")
# taboret_5 = env.scene.object_registry("name", "taboret_ivtoxu_5")

