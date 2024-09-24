from argparse import ArgumentParser
from collections import Counter
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import torch as th
import yaml

import omnigibson as og
from omnigibson.action_primitives.pick_place_semantic_action_primitives import (
    PickPlaceSemanticActionPrimitives)
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.utils.video_logging_utils import VideoLogger


def custom_reset(env, robot, args, vid_logger):
    proprio = robot._get_proprioception_dict()
    # curr_right_arm_joints = th.tensor(proprio['arm_right_qpos'])
    reset_right_arm_joints = th.tensor(
        [0.85846, -0.44852, 1.81008, 1.63368, 0.43764, -1.32488, -0.68415])

    noise_1 = np.random.uniform(-0.2, 0.2, 3)
    noise_2 = np.random.uniform(-0.01, 0.01, 4)
    noise = th.tensor(np.concatenate((noise_1, noise_2)))
    right_hand_joints_pos = reset_right_arm_joints + noise
    # right_hand_joints_pos = curr_right_arm_joints + noise

    scene_initial_state = env.scene._initial_state
    # for manipulation
    base_pos = np.array([-0.05, -0.4, 0.0])
    base_x_noise = np.random.uniform(-0.15, 0.15)
    base_y_noise = np.random.uniform(-0.15, 0.15)
    base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    base_pos += base_noise 
    scene_initial_state['object_registry']['robot0']['root_link']['pos'] = base_pos
    
    base_yaw = -120
    base_yaw_noise = np.random.uniform(-15, 15)
    base_yaw += base_yaw_noise
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat

    default_head_joints = th.tensor([-0.5031718015670776, -0.9972541332244873])
    noise_1 = np.random.uniform(-0.1, 0.1, 1)
    noise_2 = np.random.uniform(-0.1, 0.1, 1)
    noise = th.tensor(np.concatenate((noise_1, noise_2)))
    head_joints = default_head_joints + noise

    # Reset environment and robot
    obs, info = env.reset()
    robot.reset(right_hand_joints_pos=right_hand_joints_pos, head_joints_pos=head_joints)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()

    is_contact = detect_robot_collision_in_sim(robot)


def main(args):
    np.random.seed(6)
    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
 
    config["objects"] = [
        {
            "type": "PrimitiveObject",
            "name": "box",
            "primitive_type": "Cube",
            "manipulable": True,
            # ^ Should the robot be allowed to interact w/ this object?
            # Used if need to randomly select object in DialogEnvironment
            "rgba": [1.0, 0, 0, 1.0],
            "scale": [0.1, 0.05, 0.1],
            # "size": 0.05,
            "position": [-0.5, -0.7, 0.5],
            "orientation": [
                0.0004835024010390043,
                -0.00029672126402147114,
                -0.11094563454389572,
                0.9938263297080994],
        },
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "manipulable": False,
            "scale": [0.3, 0.3, 0.3],
            "position": [-0.7, 0.5, 0.2],
            "orientation": [0, 0, 0, 1]
        },
        {
            "type": "PrimitiveObject",
            "name": "pad",
            "primitive_type": "Disk",
            "rgba": [0.0, 0, 1.0, 1.0],
            "radius": 0.08,
            "position": [-0.3, -0.8, 0.5],
        },
    ]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    vid_logger = VideoLogger(args, env)
    action_primitives = PickPlaceSemanticActionPrimitives(env, vid_logger, enable_head_tracking=False)

    subtask_success_counter = Counter()
    for i in range(args.num_trajs):
        print(f"---------------- Episode {i} ------------------")
        start_time = time.time()

        custom_reset(env, robot, args, vid_logger)

        obs, obs_info = env.get_obs()

        for _ in range(50):
            og.sim.step()
            vid_logger.save_im_text()

        pick_success = action_primitives._grasp("box")
        place_success = action_primitives._place_on_top("box", "pad")

        task_success = pick_success and place_success
        num_subtasks_completed = [int(pick_success), int(place_success), 0].index(0)
        vid_logger.make_video(prefix=f"rew{num_subtasks_completed}")
        subtask_success_counter["entire"] += int(task_success)
        subtask_success_counter["pick"] += int(pick_success)
        subtask_success_counter["place"] += int(place_success)

        print(f"num successes: {subtask_success_counter['entire']} / {i + 1}\n{subtask_success_counter}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Episode {i}: execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    """
    Example usage:
    python omnigibson/examples/action_primitives/pick_place_example.py --out-dir /home/albert/scratch/20240924 --num-trajs 1
    """
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-trajs", type=int, required=True)
    parser.add_argument("--vid-downscale-factor", type=float, default=2.0)
    parser.add_argument("--vid-speedup", type=float, default=2)
    args = parser.parse_args()
    main(args)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
