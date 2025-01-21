import os
import yaml
import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options
import numpy as np


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select a type of scene and loads a turtlebot into it, generating a Point-Goal navigation
    task within the environment.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + (main.__doc__ or "") + "*" * 80)

    # Load the config
    # config_filename = os.path.join(og.example_config_path, "turtlebot_multi_nav.yaml")
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # check if we want to quick load or full load the scene
    load_options = {
        "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
        # "Full": "Load all interactive objects in the scene",
    }
    # load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
    # if load_mode == "Quick":
    #     config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Define load mode for convenience
    load_mode = "Quick"
    config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Load the environment
    env = og.Environment(configs=config)

    # Allow user to move camera more easily
    # og.sim.enable_viewer_camera_teleoperation()

    # Print robot names
    robots = env.robots
    og.log.info(f"Loaded robots: {[robot.name for robot in robots]}")

    # Save robot name
    robot_name = robots[0].name

    # Run a simple loop and reset periodically
    max_iterations = 10 if not short_exec else 1
    for j in range(max_iterations):
        og.log.info("Resetting environment")
        env.reset()
        for i in range(100):
            actions = {robot.name: robot.action_space.sample() for robot in robots}

            # states: dict
            states, rewards, terminated, truncated, infos = env.step(actions)

            print(f"states: {states}")
            print(f"states.keys: {states.keys()}")

            robot_state = states[robot_name]
            print(f"states[robot_name]: {robot_state}")

            # Get the format of the state (dict[Any, Any])
            # print(f"states: {states}")
            print(f"states.keys: {robot_state.keys()}")
            # print(f"states.values: {robot_state.values()}")
            print(f"states.items: {states.items()}")

            camera_key = f"{robot_name}:eyes:Camera:0"

            camera_output = robot_state[camera_key]
            print(f"camera output: {camera_output}")
            print(f"camera: {camera_output['rgb'].shape}")

            #  Print position and orientation of the robot (x,y,z, Yaw)
            proprio = robot_state["proprio"]
            print(f"proprio: {proprio}")
            robot_pos_2d = proprio[:2]
            robot_yaw = proprio[3]

            print(f"robot_pos_2d: {robot_pos_2d}")
            print(f"robot_yaw: {np.rad2deg(robot_yaw):.2f}")

            # print(f"state_shape: {states[robots[0].name].shape}")

            if terminated or truncated:
                og.log.info("Episode finished after {} timesteps".format(i + 1))
                break

    # Always close the environment at the end
    og.clear()


if __name__ == "__main__":
    main()
