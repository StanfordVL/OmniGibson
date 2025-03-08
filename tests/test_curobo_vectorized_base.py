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


th.manual_seed(3)
np.random.seed(3)
# Set the number of envs here!
num_envs = 1

configs = []
for i in range(num_envs):
    # for some reason cfg is giving zero norm quaternion error, so using tiago_primitives.yaml
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config["scene"]["load_object_categories"] = ["floors"]
    config["objects"].append({
                "type": "PrimitiveObject",
                "name": "base_marker",
                "primitive_type": "Sphere",
                "radius": 0.05,
                "visual_only": True,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
                "rgba": [1, 0, 0, 1],
            })
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

# randomize initial base pose
base_pos_list = [[-3.0, 2.0], [1.0, -1.0]]
base_yaw_list = [-180, 90]
base_pos_per_env, base_quat_per_env = [], []
for idx, robot in enumerate(robots):
    # base_yaw = np.random.uniform(-math.pi, math.pi)
    # remove later
    base_yaw = base_yaw_list[idx]
    base_yaw = 0.0
    base_quat = R.from_euler("z", base_yaw, degrees=True).as_quat()
    # base_pos = np.random.uniform(-1.0, 1.0, size=2)
    # remove later
    base_pos = base_pos_list[idx]
    base_pos = [0.0, 0.0]    
    
    base_pos_per_env.append(base_pos)
    base_quat_per_env.append(base_quat)
    robot.set_position_orientation(position=th.tensor([base_pos[0], base_pos[1], 0.0]), frame="scene", orientation=base_quat)

for _ in range(5):
    og.sim.step()

# Curobo initialization
batch_size = 1
emb_sel = "base"
cmg = CuRoboMotionGenerator(
    robots=robots,
    batch_size=batch_size,
    debug=False,
    use_cuda_graph=False, # change later to True
    collision_activation_distance=0.03,  # Use larger activation distance for better reproducibility
    use_default_embodiment_only=False,
    num_envs=num_envs,
    use_batch_env=False,
    warmup=False
)


# Sample values for robot
n_samples = 20

rooms = list(vec_env.envs[0].scene._seg_map.room_ins_name_to_ins_id.keys())
n_rooms = len(rooms)
target_pos_list, target_quat_list = [], []
for _ in range(n_samples):
    target_pos_all_env, target_quat_all_env = [], []
    for env_idx in range(num_envs):
        robot = robots[env_idx]
        randint = np.random.randint(0, n_rooms)
        room = rooms[randint]
        random_pt = vec_env.envs[env_idx].scene._seg_map.get_random_point_by_room_instance(room)
        random_yaw = np.random.uniform(-math.pi, math.pi)
        random_quat = R.from_euler("z", random_yaw, degrees=False).as_quat()
        random_quat = th.tensor(random_quat, dtype=th.float32)
        # remove later
        random_quat = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)
        target_pos_dict = {
            robot.base_footprint_link_name: random_pt,
        }
        target_quat_dict = {
            robot.base_footprint_link_name: random_quat,
        }
        target_pos_all_env.append(target_pos_dict)
        target_quat_all_env.append(target_quat_dict)
    target_pos_list.append(target_pos_all_env)
    target_quat_list.append(target_quat_all_env)

for env_idx, robot in enumerate(robots):
    # If scene is reset, the robot also goes back to the init base pose.
    vec_env.envs[env_idx].scene.reset()
    robot.set_position_orientation(position=th.tensor([base_pos_per_env[env_idx][0], base_pos_per_env[env_idx][1], 0.0]), frame="scene", orientation=base_quat_per_env[env_idx])

    for _ in range(20): og.sim.step()
    
    # Make sure robot is kept still for better determinism before planning
    robot.keep_still()
    og.sim.step_physics()

# breakpoint()
og.sim.viewer_camera.set_position_orientation(
    position=th.tensor([-6.751,  0.592, 10.307]),
    orientation=th.tensor([ 0.238, -0.166, -0.546,  0.786])
)

all_traj_paths = []
all_successes = []
for idx in range(len(target_pos_list)):
    # for env_idx in range(num_envs):
    results, traj_paths = cmg.compute_trajectories(
        target_pos_list=target_pos_list[idx].copy(),
        target_quat_list=target_quat_list[idx].copy(),
        is_local=False,
        max_attempts=20,
        timeout=60.0,
        ik_fail_return=20,
        enable_finetune_trajopt=True,
        finetune_attempts=1,
        return_full_result=True,
        success_ratio=1.0,
        attached_obj=None,
        # For debugging ik only
        # ik_only=True,
        # ik_world_collision_check=False,
        emb_sel=emb_sel
    )
    successes = results[0].success
    print(f"Idx {idx}. successes: ", successes)
    print("pos_err, rot_err, feasible: ", results[0].position_error, results[0].rotation_error, results[0].feasible, results[0].status)

    for env in vec_env.envs:
        robot = env.scene.robots[0]
        marker = env.scene.object_registry("name", "base_marker")
        target_pos = target_pos_list[idx][env_idx][robot.base_footprint_link_name]
        print("robot.pos, target_pos: ", robot.get_position(), target_pos)
        marker.set_position_orientation(position=target_pos, frame="scene")
        for _ in range(20): og.sim.step()

    # breakpoint()

    all_traj_paths.append(traj_paths)
    all_successes.append(successes)

all_successes = th.stack(all_successes)
all_successes = all_successes.squeeze()
print("all_successes: ", all_successes)
print("all_successes.sum(dim=0): ", all_successes.sum(dim=0))

breakpoint()