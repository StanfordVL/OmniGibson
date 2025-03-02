import os
import time

import numpy as np
import torch as th
import yaml

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.object_states import OnTop
from omnigibson.envs.primitives_env import PrimitivesEnv


def execute_controller(ctrl_gen, env):
    num_env_steps = 0
    for i, action in enumerate(ctrl_gen):
        _, r, _, _, _ = env.step(action)
        if i % 4 == 0:
            env.vid_logger.save_im_text()
        # print(f"reward: {r}")
        num_env_steps += 1
    return num_env_steps, r


def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in a crowded scene.

    It loads Rs_int with a robot, and the robot picks and places an apple.
    """
    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    configs = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    configs["scene"]["scene_model"] = "Rs_int"
    configs["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]
    # config["scene"]["not_load_object_categories"] = ["ceilings", "carpet"]
    configs["objects"] = [
        # {
        #     "type": "DatasetObject",
        #     "name": "apple",
        #     "category": "apple",
        #     "model": "agveuv",
        #     "position": [1.0, 0.5, 0.46],
        #     "orientation": [0, 0, 0, 1],
        #     "manipulable": True,
        # },
        {
            "type": "PrimitiveObject",
            "name": "box",
            "primitive_type": "Cube",
            "manipulable": True,
            "rgba": [1.0, 0, 0, 1.0],
            "scale": [0.15, 0.15, 0.05],
            # "size": 0.05,
            "position": [1.0, 0.5, 0.5],
            "orientation": [0, 0, 0, 1],
        },
        {
            "type": "PrimitiveObject",
            "name": "pad",
            "primitive_type": "Disk",
            "rgba": [0.0, 0, 1.0, 1.0],
            "radius": 0.12,
            "position": [-0.3, -0.9, 0.5],
        },
        {
            "type": "DatasetObject",
            "name": "coffee_table_pick",
            "category": "coffee_table",
            "model": "zisekv",
            "position": [1.2, 0.6, 0.4],
            "orientation": [0, 0, 0, 1],
        },
    ]

    # Load the environment
    # env = og.Environment(configs=config)
    # scene = env.scene
    # robot = env.robots[0]

    # og.sim.viewer_camera.set_position_orientation(
    #     th.tensor([1.8294, -10, 1.6885]), th.tensor([0.5770, 0.1719, 0.2280, 0.7652])
    # )
    # # Allow user to move camera more easily
    # og.sim.enable_viewer_camera_teleoperation()

    # breakfast_table = scene.object_registry("name", "breakfast_table_skczfi_0")
    # coffee_table = scene.object_registry("name", "coffee_table_fqluyq_0")
    # apple = scene.object_registry("name", "apple")
    # floor = scene.object_registry("name", "floors_ptwlei_0")
    st = time.time()
    env = PrimitivesEnv(configs, out_dir="/home/albert/dev/OmniGibson/out_videos", obj_to_grasp_name="box")
    print(f"time to init env: {time.time() - st}")
    st = time.time()
    robot = env.robots[0]
    controller = StarterSemanticActionPrimitives(env, robot, enable_head_tracking=False, skip_curobo_initilization=False)
    print(f"time to init Semantic Action primitives: {time.time() - st}")
    coffee_table_place = env.get_obj_by_name("coffee_table_fqluyq_0")
    coffee_table_pick = env.get_obj_by_name("coffee_table_pick")
    # apple = env.get_obj_by_name("apple")
    box = env.get_obj_by_name(env.obj_to_grasp_name)
    pad = env.get_obj_by_name("pad")

    total_num_env_steps_list = []
    rewards_list = []
    num_trials = 100
    for trial_idx in range(num_trials):
        st = time.time()
        for i in range(10):
            og.sim.step()
        print(f"time for 100 sim steps: {time.time() - st}")
        st = time.time()
        env.make_video()
        env.scene.reset()
        print(f"time for env reset: {time.time() - st}")

        # Randomize the rbot pose
        # robot.states[OnTop].set_value(floor, True)

        # Randomize the apple pose on top of the breakfast table
        # apple.states[OnTop].set_value(breakfast_table, True)

        try:
            # Grasp apple from breakfast table
            print("Start executing grasp")
            st = time.time()
            grasp_num_env_steps, _ = execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, box), env)
            print(f"Finish executing grasp. time: {time.time() - st}")

            # Place on coffee table
            print("Start executing place")
            st = time.time()
            place_num_env_steps, r = execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, pad), env)
            print(f"Finish executing place. time: {time.time() - st}")

            # TODO: collect success / failure statistics
            print("grasp_num_env_steps", grasp_num_env_steps)
            print("place_num_env_steps", place_num_env_steps)
            total_num_env_steps = grasp_num_env_steps + place_num_env_steps
            print("total_num_env_steps", total_num_env_steps)
        except:
            total_num_env_steps = None
            r = 0

        total_num_env_steps_list.append(total_num_env_steps)
        rewards_list.append(r)
        print(f"Number of successes so far:{np.sum(rewards_list)}/{trial_idx+1}")

    avg_num_ts = np.nanmean(np.array(total_num_env_steps_list, dtype=np.float64))
    print("total_num_env_steps_list", total_num_env_steps_list, f"Avg: {avg_num_ts}")
    print("rewards_list", rewards_list)


if __name__ == "__main__":
    main()
