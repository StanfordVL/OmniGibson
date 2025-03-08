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
num_envs = 2

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
    base_quat = R.from_euler("z", base_yaw, degrees=True).as_quat()
    # base_pos = np.random.uniform(-1.0, 1.0, size=2)
    # remove later
    base_pos = base_pos_list[idx]    
    
    base_pos_per_env.append(base_pos)
    base_quat_per_env.append(base_quat)
    robot.set_position_orientation(position=th.tensor([base_pos[0], base_pos[1], 0.0]), frame="scene", orientation=base_quat)

for _ in range(5):
    og.sim.step()


# Curobo initialization
batch_size = 1
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


# Sample values for robot
n_samples = 20
lo, hi = robot.joint_lower_limits.clone().view(1, -1), robot.joint_upper_limits.clone().view(1, -1)

if isinstance(robot, HolonomicBaseRobot):
    lo[0, :2] = -0.1
    lo[0, 2:5] = 0.0
    lo[0, 5] = -math.pi
    hi[0, :2] = 0.1
    hi[0, 2:5] = 0.0
    hi[0, 5] = math.pi

random_qs_list = []
for i in range(num_envs):
    robot = robots[i]
    random_qs = lo + th.rand((n_samples, robot.n_dof)) * (hi - lo)
    random_qs[:, robot.base_control_idx] = robot.get_joint_positions()[robot.base_control_idx]
    random_qs[:, robot.camera_control_idx] = robot.get_joint_positions()[robot.camera_control_idx]
    random_qs[:, robot.gripper_control_idx["left"]] = robot.get_joint_positions()[robot.gripper_control_idx["left"]]
    random_qs[:, robot.gripper_control_idx["right"]] = robot.get_joint_positions()[robot.gripper_control_idx["right"]]
    random_qs_list.append(random_qs)


# # Test collision with the environment (not including self-collisions)
# collision_results = cmg.check_collisions(qs=random_qs_list)

target_pos = [defaultdict(list) for _ in range(num_envs)]
target_quat = [defaultdict(list) for _ in range(num_envs)]
random_qs_list = th.stack(random_qs_list).permute(1,0,2)

for q in random_qs_list:
    for env_idx, robot in enumerate(robots):
        # Set robot to desired qpos
        robot.set_joint_positions(q[env_idx])
        robot.keep_still()
    og.sim.step_physics()

    # Sanity check in the GUI that the robot pose makes sense
    for _ in range(20):
        og.sim.render()

    for env_idx, robot in enumerate(robots):
        for arm_name in robot.arm_names:
            eef_pos, eef_quat = robot.get_eef_pose(arm_name)
            target_pos[env_idx][robot.eef_link_names[arm_name]].append(eef_pos)
            target_quat[env_idx][robot.eef_link_names[arm_name]].append(eef_quat)

# breakpoint()

for env_idx, robot in enumerate(robots):
    # If scene is reset, the robot also goes back to the init base pose.
    vec_env.envs[env_idx].scene.reset()
    robot.set_position_orientation(position=th.tensor([base_pos_per_env[env_idx][0], base_pos_per_env[env_idx][1], 0.0]), frame="scene", orientation=base_quat_per_env[env_idx])

    for _ in range(20): og.sim.step()
    
    for arm_name in robot.arm_names:
        target_pos[env_idx][robot.eef_link_names[arm_name]] = th.stack(target_pos[env_idx][robot.eef_link_names[arm_name]], dim=0)
        target_quat[env_idx][robot.eef_link_names[arm_name]] = th.stack(target_quat[env_idx][robot.eef_link_names[arm_name]], dim=0)

    # Cast defaultdict to dict
    target_pos[env_idx] = dict(target_pos[env_idx])
    target_quat[env_idx] = dict(target_quat[env_idx])

    print(f"Planning for {len(target_pos[env_idx][robot.eef_link_names[robot.default_arm]])} eef targets... for env {env_idx}")

    # Make sure robot is kept still for better determinism before planning
    robot.keep_still()
    og.sim.step_physics()

# Make target shape homogeneous
max_traj_len = max([len(t["left_eef_link"]) for t in target_pos])
for i in range(len(target_pos)):
    current_traj_len = len(target_pos[i]["left_eef_link"])
    if current_traj_len < max_traj_len:
        num_repeats = max_traj_len - current_traj_len
        for _ in range(num_repeats):
            target_pos[i]["left_eef_link"] = th.cat((target_pos[i]["left_eef_link"], target_pos[i]["left_eef_link"][-1:]))
            target_pos[i]["right_eef_link"] = th.cat((target_pos[i]["right_eef_link"], target_pos[i]["right_eef_link"][-1:]))
            target_quat[i]["left_eef_link"] = th.cat((target_quat[i]["left_eef_link"], target_quat[i]["left_eef_link"][-1:]))
            target_quat[i]["right_eef_link"] = th.cat((target_quat[i]["right_eef_link"], target_quat[i]["right_eef_link"][-1:]))

# inefficient way of performing .permute(1,0)
target_pos_transposed, target_quat_transposed = [], []
for trial_idx in range(len(target_pos[i]["left_eef_link"])):
    target_pos_all_envs = []
    target_quat_all_envs = []
    for env_idx in range(len(vec_env.envs)): 
        pos_dict = {
            "left_eef_link": target_pos[env_idx]["left_eef_link"][trial_idx],
            "right_eef_link": target_pos[env_idx]["right_eef_link"][trial_idx],
        }
        quat_dict = {
            "left_eef_link": target_quat[env_idx]["left_eef_link"][trial_idx],
            "right_eef_link": target_quat[env_idx]["right_eef_link"][trial_idx],
        }
        target_pos_all_envs.append(pos_dict)
        target_quat_all_envs.append(quat_dict)
    target_pos_transposed.append(target_pos_all_envs)
    target_quat_transposed.append(target_quat_all_envs)

# breakpoint()
emb_sel = "arm"
all_traj_paths = []
all_successes = []
for idx in range(len(target_pos_transposed)):
    # for env_idx in range(num_envs):
    successes, traj_paths = cmg.compute_trajectories(
        target_pos_list=target_pos_transposed[idx].copy(),
        target_quat_list=target_quat_transposed[idx].copy(),
        # target_pos_list=target_pos_transposed[idx][env_idx: env_idx+1].copy(),
        # target_quat_list=target_quat_transposed[idx][env_idx:env_idx+1].copy(),
        is_local=False,
        max_attempts=20,
        timeout=60.0,
        ik_fail_return=20,
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
    print(f"Idx {idx}. successes: ", successes)
    all_traj_paths.append(traj_paths)
    all_successes.append(successes)
all_successes = th.stack(all_successes)
all_successes = all_successes.squeeze()
print("all_successes: ", all_successes)
print("all_successes.sum(dim=0): ", all_successes.sum(dim=0))

breakpoint()