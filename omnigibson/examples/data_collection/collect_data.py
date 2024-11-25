import os
import pdb

import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import BimanualKeyboardRobotController, choose_from_options

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False


def main():
    # config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    config_filename = os.path.join(og.example_config_path, "tiago_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # activity_name = "test_cabinet"
    activity_name = "test_tiago_cup"
    cfg["task"]["activity_name"] = activity_name
    cfg["task"]["online_object_sampling"] = False
    cfg["env"]["flatten_obs_space"] = True
    cfg["env"]["action_frequency"] = 30
    cfg["env"]["rendering_frequency"] = 30
    cfg["env"]["physics_frequency"] = 120
    cfg["robots"][0]["default_reset_mode"] = "untuck"

    collect_hdf5_path = f"{activity_name}.hdf5"

    # Load the environment
    env = og.Environment(configs=cfg)
    env = DataCollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
        optimize_sim=False,
    )
    robot = env.robots[0]
    coffee_cup = env.scene.object_registry("name", "coffee_cup")
    coffee_cup.links['base_link'].density = 100
    paper_cup = env.scene.object_registry("name", "paper_cup")
    paper_cup.links['base_link'].density = 100
    pdb.set_trace()

    # Create teleop controller
    action_generator = BimanualKeyboardRobotController(robot=robot)

    # Register custom binding to reset the environment
    # action_generator.register_custom_keymapping(
    #     key=lazy.carb.input.KeyboardInput.R,
    #     description="Reset the robot",
    #     callback_fn=lambda: env.reset(),
    # )
    action_generator.print_keyboard_teleop_info_bimanual()

    print("Getting ready")
    pdb.set_trace()
    n_episodes = 1
    for i in range(n_episodes):
        print(f"Episode {i} starts")
        env.reset()
        while True:
            action = action_generator.get_teleop_action_bimanual()
            next_obs, reward, terminated, truncated, info = env.step(action=action)
            success = info["done"]["success"]
            print("success:", success)
            if success:
                break

        # Take 5 more zero actions
        action[:] = 0.0
        for _ in range(5):
            next_obs, reward, terminated, truncated, info = env.step(action=action)
            success = info["done"]["success"]
        assert success

    print("Data saved")
    env.save_data()

    og.shutdown()


if __name__ == "__main__":
    main()
