import gc
import math
import os
import yaml
from collections import defaultdict

import torch as th
import numpy as np

import omnigibson as og
from omnigibson.action_primitives.curobo_vectorized import CuRoboMotionGenerator
from omnigibson.macros import gm
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.robots.locomotion_robot import LocomotionRobot
import omnigibson.utils.transform_utils as T

# Set the number of envs here!
num_envs = 3

configs = []
for i in range(num_envs):
    # for some reason cfg is giving zero norm quaternion error, so using tiago_primitives.yaml
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config["scene"]["load_object_categories"] = ["floors"]
    configs.append(config)

vec_env = og.VectorEnvironment(num_envs, configs)
robots = []
objs = []
floor_touching_base_link_prim_paths_list = []

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

usd_help = UsdHelper()
# world_file = ["collision_test.yml", "collision_thin_walls.yml"]
world_file = ["collision_thin_walls.yml" for i in range(num_envs)]
world_cfg_list = []
for i in range(num_envs):
    world_cfg = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), world_file[i]))
    )  # .get_mesh_world()
    world_cfg.objects[0].pose[2] -= 0.02
    world_cfg.randomize_color(r=[0.2, 0.3], b=[0.0, 0.05], g=[0.2, 0.3])
    # usd_help.add_world_to_stage(world_cfg, base_frame="/World/world_" + str(i))
    world_cfg_list.append(world_cfg)
    

tensor_args = TensorDeviceType()
robot_cfg_path = '/home/arpit/test_projects/OmniGibson/omnigibson/data/assets/models/tiago/curobo/tiago_description_curobo_arm.yaml'
robot_usd_path = '/home/arpit/test_projects/OmniGibson/omnigibson/data/assets/models/tiago/usd/tiago.usda'
content_path = ContentPath(
                robot_config_absolute_path=robot_cfg_path, robot_usd_absolute_path=robot_usd_path
            )
robot_cfg_dict = curobo.cuda_robot_model.util.load_robot_yaml(content_path)["robot_cfg"]
robot_cfg_dict["kinematics"]["use_usd_kinematics"] = True
robot_cfg = curobo.types.robot.RobotConfig.from_dict(robot_cfg_dict, tensor_args)

motion_kwargs = dict(
    trajopt_tsteps=32,
    num_ik_seeds=128,
    num_batch_ik_seeds=128,
    num_batch_trajopt_seeds=1,
    num_trajopt_noisy_seeds=1,
    ik_opt_iters=100,
    optimize_dt=True,
    num_trajopt_seeds=4,
    num_graph_seeds=4,
    interpolation_dt=0.03,
    collision_activation_distance=0.005,
    self_collision_check=True,
    maximum_trajectory_dt=None,
    fixed_iters_trajopt=True,
    finetune_trajopt_iters=100,
    finetune_dt_scale=1.05,
    position_threshold=0.01,    # change later to 0.005
    rotation_threshold=0.1,     # change later to 0.05
)
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_cfg,
    world_cfg_list,
    tensor_args,
    collision_checker_type=CollisionCheckerType.MESH,
    use_cuda_graph=True,
    # interpolation_dt=0.03,
    collision_cache={"obb": 10, "mesh": 10},
    # collision_activation_distance=0.025,
    # maximum_trajectory_dt=0.25,
    **motion_kwargs
)
motion_gen = MotionGen(motion_gen_config)

cmg = CuRoboMotionGenerator(
    robots=robots,
    batch_size=batch_size,
    debug=False,
    use_cuda_graph=False, # change later to True
    collision_activation_distance=0.03,  # Use larger activation distance for better reproducibility
    use_default_embodiment_only=False,
    num_envs=num_envs
)

from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from curobo.types.math import Pose

plan_config = MotionGenPlanConfig(
        enable_graph=False, max_attempts=20, enable_finetune_trajopt=True
    )

sp_buffer = [robots[0].get_relative_eef_pose()[0].cpu().numpy()] * len(robots)
sq_buffer = [robots[0].get_relative_eef_pose()[1][[3, 0, 1, 2]].cpu().numpy()] * len(robots)
ik_goal = Pose(
    position=tensor_args.to_device(sp_buffer),
    quaternion=tensor_args.to_device(sq_buffer),
)

q_pos = th.stack([robots[0].get_joint_positions()] * batch_size, axis=0)
q_vel = th.stack([robots[0].get_joint_velocities()] * batch_size, axis=0)
q_eff = th.stack([robots[0].get_joint_efforts()] * batch_size, axis=0)
full_js = curobo.types.state.JointState(
    position=tensor_args.to_device(q_pos),
    velocity=tensor_args.to_device(q_vel) * 0.0,
    acceleration=tensor_args.to_device(q_eff) * 0.0,
    jerk=tensor_args.to_device(q_eff) * 0.0,
    joint_names=list(robots[0].joints.keys()),
)
for env_idx in range(1, len(robots)):
    cu_js = curobo.types.state.JointState(
        position=tensor_args.to_device(q_pos),
        velocity=tensor_args.to_device(q_vel) * 0.0,
        acceleration=tensor_args.to_device(q_eff) * 0.0,
        jerk=tensor_args.to_device(q_eff) * 0.0,
        joint_names=list(robots[0].joints.keys()),
    )
    full_js = full_js.stack(cu_js)

full_js = full_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

solve_state = motion_gen._get_solve_state(
    ReacherSolveType.BATCH_ENV, plan_config, ik_goal, full_js
)


# result = motion_gen._solve_ik_from_solve_state(
#     ik_goal,
#     solve_state,
#     full_js,
#     plan_config.use_nn_ik_seed,
#     plan_config.partial_ik_opt,
#     link_poses=None,
# )
# print("result.success: ", result.success)
# breakpoint()


result, success, joint_state = cmg.solve_ik_batch(full_js, ik_goal, plan_config, link_poses=None, emb_sel="arm")
print("result.success: ", result.success)
breakpoint()
