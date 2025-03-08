import gc
import math
import os
import yaml
from collections import defaultdict

import torch as th
th.set_printoptions(precision=3, sci_mode=False)
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from scipy.spatial.transform import Rotation as R

import omnigibson as og
from omnigibson.action_primitives.curobo_vectorized import CuRoboMotionGenerator
from omnigibson.macros import gm
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.robots.locomotion_robot import LocomotionRobot
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo_vectorized import CuRoboEmbodimentSelection


# Set the number of envs here!
num_envs = 2

configs = []
for i in range(num_envs):
    # for some reason cfg is giving zero norm quaternion error, so using tiago_primitives.yaml
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    # if i == 0:
    #     obj_cfg = dict(
    #         type="DatasetObject",
    #         name="shelf",
    #         category="shelf",
    #         model="eniafz",
    #         position=[1.0, 0.0, 1.0],
    #         # scale=[0.4, 0.4, 1.0],
    #         # orientation=rot_quat,
    #         )
    #     config["objects"].append(obj_cfg)
    if i == 1:
        obj_cfg = dict(
            type="DatasetObject",
            name="shelf",
            category="shelf",
            model="eniafz",
            position=[0.685, 0.0, 1.0],
            # scale=[0.4, 0.4, 1.0],
            # orientation=rot_quat,
            )
        config["objects"].append(obj_cfg)
    config["scene"]["load_object_categories"] = ["floors"]
    configs.append(config)

vec_env = og.VectorEnvironment(num_envs, configs)
robots = []
objs = []
floor_touching_base_link_prim_paths_list = []


og.sim.viewer_camera.set_position_orientation(position=th.tensor([14.423, -4.440,  2.552]), orientation=th.tensor([ 0.589, -0.063, -0.085,  0.801]))

for env in vec_env.envs:
    robot = env.scene.robots[0]
    robots.append(robot) 
    robot.reset()

    # Open the gripper(s) to match cuRobo's default state
    for arm_name in robot.gripper_control_idx.keys():
        gripper_control_idx = robot.gripper_control_idx[arm_name]
        robot.set_joint_positions(th.ones_like(gripper_control_idx), indices=gripper_control_idx, normalized=True)

    robot.keep_still()

    for _ in range(5):
        og.sim.step()

    env.scene.update_initial_state()
    env.scene.reset()

base_yaw = 0
base_quat = R.from_euler("z", base_yaw, degrees=True).as_quat()
robots[0].set_position_orientation(position=th.tensor([1.0, 1.0, 0.0]), frame="scene", orientation=base_quat)
base_yaw = 0
base_quat = R.from_euler("z", base_yaw, degrees=True).as_quat()
robots[1].set_position_orientation(position=th.tensor([-0.1, 0.0, 0.0]), frame="scene", orientation=base_quat)
for _ in range(5):
    og.sim.step()

# breakpoint()

# Create CuRobo instance
batch_size = 1  # change later to 2 

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.geom.types import WorldConfig
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.base import TensorDeviceType
from curobo.types.file_path import ContentPath
import curobo

emb_sel = "arm"

cmg = CuRoboMotionGenerator(
    robots=robots,
    batch_size=batch_size,
    debug=False,
    use_cuda_graph=False, # change later to True
    collision_activation_distance=0.03,  # Use larger activation distance for better reproducibility
    use_default_embodiment_only=False,
    num_envs=num_envs,
    use_batch_env=True,
    warmup=False
)

if emb_sel == "arm":

    target_pos = [defaultdict(list) for _ in range(num_envs)]
    target_quat = [defaultdict(list) for _ in range(num_envs)]
    for env_idx, env in enumerate(vec_env.envs):
        robot = env.robots[0]
        for arm_name in robot.arm_names:
            eef_pos, eef_quat = robot.get_eef_pose(arm_name)
            # eef_pos += th.tensor([0.1, 0.0, 0.0], dtype=th.float32)
            if env_idx == 0:
                eef_pos += th.tensor([0.2, 0.0, 0.0], dtype=th.float32)
            else:
                eef_pos += th.tensor([0.2, 0.0, 0.0], dtype=th.float32)

            target_pos[env_idx][robot.eef_link_names[arm_name]].append(eef_pos)
            target_quat[env_idx][robot.eef_link_names[arm_name]].append(eef_quat)

            target_pos[env_idx][robot.eef_link_names[arm_name]] = th.stack(target_pos[env_idx][robot.eef_link_names[arm_name]], dim=0)
            target_quat[env_idx][robot.eef_link_names[arm_name]] = th.stack(target_quat[env_idx][robot.eef_link_names[arm_name]], dim=0)

    # Cast defaultdict to dict
    target_pos[env_idx] = dict(target_pos[env_idx])
    target_quat[env_idx] = dict(target_quat[env_idx])

if emb_sel == "base":
    target_pos = [
        {'base_footprint': robots[0].get_position_orientation()[0] + th.tensor([0.0, -2.0, 0.0])}, 
        {'base_footprint': robots[1].get_position_orientation()[0] + th.tensor([0.0, -2.0, 0.0])}, 
    ]

    target_quat= [
        {'base_footprint': th.tensor([    -0.000,     -0.000,      0.082,      0.997])}, 
        {'base_footprint': th.tensor([    -0.000,     -0.000,      0.056,      0.998])}, 
    ]

breakpoint()
successes, traj_paths = cmg.compute_trajectories(
    target_pos_list=target_pos.copy(),
    target_quat_list=target_quat.copy(),
    is_local=False,
    max_attempts=30,
    timeout=60.0,
    ik_fail_return=5,
    enable_finetune_trajopt=True,
    finetune_attempts=1,
    return_full_result=False,
    success_ratio=1.0,
    attached_obj=None,
    # For debugging ik only
    # ik_only=True,
    # ik_world_collision_check=False,
    emb_sel=emb_sel
)
print("successes: ", successes)

# # inefficient way of performing .permute(1,0)
# traj_paths_transposed = []
# for trial_idx in range(len(traj_paths[1])):
#     traj_path_all_envs = []
#     for env_idx in range(len(vec_env.envs)): 
#         traj_path_all_envs.append(traj_paths[env_idx][trial_idx])
#     traj_paths_transposed.append(traj_path_all_envs)
# traj_paths = traj_paths_transposed


q_trajs = []
for idx, success in enumerate(successes[0]):
    traj_path = traj_paths[idx][0]
    q_traj = cmg.path_to_joint_trajectory(
                    traj_path, get_full_js=True, emb_sel=emb_sel, env_idx=idx
                ).cpu()
    q_trajs.append(q_traj)

# padding
max_traj_len = max([q_traj.shape[0] for q_traj in q_trajs])
for i in range(len(q_trajs)):
    current_traj_len = q_trajs[i].shape[0]
    if current_traj_len < max_traj_len:
        num_repeats = max_traj_len - current_traj_len
        q_trajs[i] = th.cat([q_trajs[i], q_trajs[i][-1].repeat(num_repeats, 1)])

q_trajs = th.stack(q_trajs)

# breakpoint()
all_actions = []
for i in range(len(q_trajs[0])):
    actions_all_env = []
    for env_idx in range(len(q_trajs)):
        joint_pos = q_trajs[env_idx][i]
        action = robots[env_idx].q_to_action(joint_pos)
        actions_all_env.append(action)
    vec_env.step(actions_all_env)
    all_actions.append(actions_all_env)

if emb_sel == "arm":
    print("target_pos env 0: ", target_pos[0]["left_eef_link"])
    print("reached pos env 0: ", robots[0].get_eef_pose()[0])
    print("target_pos env 1: ", target_pos[1]["left_eef_link"])
    print("reached pos env 1: ", robots[1].get_eef_pose()[0])
elif emb_sel == "base":
    print("target_pos env 0: ", target_pos[0]["base_footprint"])
    print("reached pos env 0: ", robots[0].get_position_orientation()[0])
    print("target_pos env 1: ", target_pos[1]["base_footprint"])
    print("reached pos env 1: ", robots[1].get_position_orientation()[0])


breakpoint()



# Test cases:
# arm emb + base emb
# 1. robot0 and robot1 at origin
# 2. robot0 at origin, robot1 at non origin
# 3. robot0 at non origin, robot1 at origin
# 4. robot0 and robot1 at non origin

# base collisions
# 1. different obstacles in each env and same target pose (0.0, -0.2, 0.0)

# arm collisions
# 1. shelf in front of robot1 and robot0 free at (1.0, 1.0, 0.0)

