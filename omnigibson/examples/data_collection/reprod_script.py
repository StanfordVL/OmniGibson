import math
import tempfile
import torch as th
th.set_printoptions(precision=3, sci_mode=False)
import omnigibson as og

import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

def execute_controller(ctrl_gen, env, robot, grasp_action=1.0):
    obs, info = env.get_obs()
    for action in ctrl_gen:
        action[robot.gripper_action_idx["right"]] = grasp_action
        obs, reward, terminated, truncated, info = env.step(action)
    return obs, info


cfg = {
    "env": {
        "action_frequency": 30,
        "rendering_frequency": 60 if gm.ENABLE_HQ_RENDERING else 30,
        "physics_frequency": 120,
        "external_sensors": [
            {
                "sensor_type": "VisionSensor",
                "name": "external_sensor0",
                "relative_prim_path": "/controllable__r1__robot0/base_link/external_sensor0",
                "modalities": [],
                "sensor_kwargs": {
                    "viewport_name": "Viewport",
                    "image_height": 720,
                    "image_width": 1280,
                },
                "position": [-0.4, 0, 2.0],  # [-0.74, 0, 2.0],
                "orientation": [0.369, -0.369, -0.603, 0.603],
                "pose_frame": "parent",
                "include_in_obs": False,
            },
        ],
    },
    "robots": [
        {
            "type": "Tiago",
            "obs_modalities": ["rgb"],
            "action_normalize": False,
            "grasping_mode": "physical",
            "controller_config": {
                "arm_left": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                    "mode": "pose_absolute_ori"
                },
                "gripper_left": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
                "arm_right": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                    "mode": "pose_absolute_ori"
                },
                "gripper_right": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    # "isaac_kp": 1e11,
                    "mode": "binary",
                    # "name": "JointController",
                    # "motor_type": "position",
                    # "command_input_limits": None,
                    # "use_delta_commands": False,
                    # "use_impedances": True,
                    # "pos_kp": 1000,
                },
            },
            "reset_joint_pos": [
                0.0000, 
                0.0000,
                -0.0000,
                -0.0000,
                -0.0000,
                -0.0000,
                0.3500,
                0.775,
                0.775,
                0.0000,
                -1.083,
                -1.083,
                -0.4500,
                1.598,
                1.598,
                1.781,
                1.781,
                0.681,
                0.681,
                -1.290,
                -1.290,
                -0.718,
                -0.718,
                0.0450,
                0.0450,
                0.0450,
                0.0450,
            ]
        }
    ],
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "load_object_categories": ["floors"],
    },
    "objects": [
        {
            "type": "DatasetObject",
            "name": "breakfast_table",
            "category": "breakfast_table",
            "model": "bhszwe",
            "scale": [0.8, 0.8, 1.0],
            "position": [1.6, 0.0, 0.7],
            "orientation": T.euler2quat(th.tensor([0, 0, 0])),
        },
        {
            "type": "DatasetObject",
            "name": "teacup",
            "category": "teacup",
            "model": "cpozxi",
            "scale": [1.0, 1.0, 1.0],
            "position": [1.5, 0.15, 0.8],
            "orientation": T.euler2quat(th.tensor([0, 0, -math.pi / 2.0])),
        },
        {
            "type": "DatasetObject",
            "name": "coffee_cup",
            "category": "coffee_cup",
            "model": "dkxddg",
            "scale": [2.0, 2.0, 1.5],
            "position": [1.55, -0.2, 0.8],
            "orientation": T.euler2quat(th.tensor([0, 0, math.pi * 1.5])),
        },
    ],
}

if og.sim is None:
    # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth)
    gm.ENABLE_OBJECT_STATES = True
    # gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_TRANSITION_RULES = False
else:
    # Make sure sim is stopped
    og.sim.stop()

# Create the environment (wrapped as a DataCollection env)
env = og.Environment(configs=cfg)

for _ in range(20): og.sim.step()

# update viewer camera pose
og.sim.viewer_camera.set_position_orientation(position=[-0.22, 0.99, 1.09], orientation=[-0.14, 0.47, 0.84, -0.23])
# og.sim.viewer_camera.set_position_orientation(position=th.tensor([ 3.22, -0.026,  2.476]),orientation=th.tensor([0.323, 0.329, 0.634, 0.621]),)
# Start teleoperation system
robot = env.robots[0]
# robot.set_joint_positions(th.tensor([0.35]), indices=robot.trunk_control_idx)
robot.set_joint_positions(th.tensor([-0.0, -0.5]), indices=robot.camera_control_idx)
robot.set_position_orientation(position=th.tensor([0.9, 0.0, 0.0]))

state = og.sim.dump_state()
og.sim.stop()
# Set friction
from omni.isaac.core.materials import PhysicsMaterial
gripper_mat = PhysicsMaterial(
    prim_path=f"{robot.prim_path}/gripper_mat",
    name="gripper_material",
    static_friction=10.0,
    dynamic_friction=10.0,
    restitution=None,
)
for arm, links in robot.finger_links.items():
    for link in links:
        for msh in link.collision_meshes.values():
            msh.apply_physics_material(gripper_mat)

og.sim.play()
og.sim.load_state(state)

for _ in range(20): og.sim.step()

# obj = env.scene.object_registry("name", "teacup")
# obj.root_link.mass = 0.05
# obj = env.scene.object_registry("name", "coffee_cup")
# obj.root_link.mass = 0.05


action_primitives = StarterSemanticActionPrimitives(env, robot, enable_head_tracking=False)
right_eef_pos, right_eef_quat = th.tensor([ 1.579, -0.159,  0.807]), th.tensor([-0.629, -0.764, -0.030, -0.141])
target_pose = (right_eef_pos, right_eef_quat)
execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), 
                       env, 
                       robot)

for _ in range(40): og.sim.step()

# breakpoint()

action = action_primitives._empty_action()
action[robot.gripper_action_idx["right"]] = -1.0
env.step(action)
for _ in range(100): og.sim.step()

# breakpoint()

right_eef_pos, right_eef_quat = th.tensor([ 1.279, -0.159,  1.107]), th.tensor([-0.629, -0.764, -0.030, -0.141])
target_pose = (right_eef_pos, right_eef_quat)
execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), env, robot, grasp_action=-1.0)

for _ in range(100): og.sim.step()

breakpoint()
