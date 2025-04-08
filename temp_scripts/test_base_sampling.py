import os
import yaml
import pickle
import imageio
import omnigibson as og
import torch as th
th.set_printoptions(precision=3, sci_mode=False)
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.objects.primitive_object import PrimitiveObject
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from omnigibson.utils.misc_utils import hori_concatenate_image, combine_videos

import scipy.spatial.transform as tf

seed = 0
np.random.seed(seed)
th.manual_seed(seed)

def reset():
    # obs, info = env.reset()
    orn = R.from_euler("xyz", [0, 0, 0], degrees=True).as_quat()
    robot.set_position_orientation(position=th.tensor([-1.0, 0.0, 0.0]), orientation=orn)

    coffee_cup.states[object_states.OnTop].set_value(other=env.scene.object_registry("name", "breakfast_table"), new_value=True)

    for _ in range(20): og.sim.step()


robot = "R1"

config_filename = os.path.join(og.example_config_path, "r1_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"]["type"] = "Scene"

with open("/home/arpit/test_projects/mimicgen/kwargs.pickle", "rb") as f:
    kwargs = pickle.load(f)
    # breakpoint()
    # kwargs["scene"] = {"type": "Scene"}
    if robot == "R1":
        kwargs["robots"][0]["type"] = "R1"
        del kwargs["robots"][0]["reset_joint_pos"]
env = og.Environment(configs=kwargs)

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
env.robots[0]._grasping_mode = "sticky"
robot = env.robots[0]
primitive = StarterSemanticActionPrimitives(env, env.robots[0], enable_head_tracking=False, curobo_batch_size=1)

og.sim.viewer_camera.set_position_orientation(th.tensor([0.0077, 0.1327, 4.9984]), th.tensor([ 0.0108, -0.0106, -0.6998,  0.7142]))

for _ in range(10): og.sim.step()
coffee_cup = env.scene.object_registry("name", "coffee_cup")


for i in range(10):
        
    reset()

    action_generator = primitive._navigate_to_obj(obj=coffee_cup, visibility_constraint=False)
    next(iter(action_generator))
    breakpoint()

    # for mp_action in action_generator:
    #     if mp_action is None:
    #         break
        
    #     mp_action = mp_action.cpu().numpy()
    #     obs, _, _, _, info = env.step(mp_action)
        
    #     if write_video:
    #         robot_name = env.robots[0].name
    #         ego_img = obs[f"{robot_name}::{robot_name}:eyes:Camera:0::rgb"].numpy()[:, :, :3]
    #         viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'].numpy()[:, :, :3]
    #         concatenated_img = hori_concatenate_image([ego_img, viewer_img])
    #         video_writer.append_data(concatenated_img)

    # if write_video:
    #     video_writer.close()
    # # breakpoint()



# # Folder containing MP4 files
# folder_path = "/home/arpit/test_projects/OmniGibson/debug_videos/soft_visibility_constraint"
# output_path = "/home/arpit/test_projects/OmniGibson/debug_videos"
# combine_videos(folder_path, output_path)


