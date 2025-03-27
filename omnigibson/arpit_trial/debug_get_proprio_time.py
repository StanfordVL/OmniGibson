import pickle
import time
import omnigibson as og
import torch as th
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.lazy as lazy

with open("/home/arpit/test_projects/mimicgen/kwargs.pickle", "rb") as f:
    kwargs = pickle.load(f)
    # kwargs["scene"] = {"type": "Scene"}
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

# with open("/home/arpit/test_projects/mimicgen/scene_0.pickle", "rb") as f:
with open("/home/arpit/test_projects/mimicgen/debug_no_valid_pose.pickle", "rb") as f:
    scene_0 = pickle.load(f)
og.sim.load_state(scene_0, serialized=False)

for _ in range(20): og.sim.step()

while True:
    # start_time = time.time()
    prop_state = env.robots[0]._get_proprioception_dict()
    # print("Time taken: {:.2f} seconds".format(time.time() - start_time))
