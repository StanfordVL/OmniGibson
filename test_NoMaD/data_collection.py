import os
import pickle
import time
import numpy as np
from PIL import Image as PILImage
import yaml

import omnigibson as og
from omnigibson.macros import gm
import torch as th


# You can import or re-define array_to_pil from your NoMaD code:
def array_to_pil(rgb_tensor: th.Tensor) -> PILImage.Image:
    """Converts OmniGibson camera output (H, W, 4) to a PIL Image."""
    if rgb_tensor.shape[-1] == 4:
        rgb_tensor = rgb_tensor[..., :3]
    rgb_array = rgb_tensor.cpu().numpy().astype(np.uint8)
    return PILImage.fromarray(rgb_array)


def init_dataset(dataset_root="./omnigibson_dataset"):
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)
    return dataset_root


def start_new_trajectory(dataset_root, traj_id):
    traj_folder = os.path.join(dataset_root, f"trajectory_{traj_id}")
    os.makedirs(traj_folder, exist_ok=True)
    # Initialize counters and lists to store odometry
    data = {"image_counter": 0, "positions": [], "yaws": []}
    return traj_folder, data


def capture_step(traj_folder, data, robot_state, camera_key):
    # Extract the RGB image from the robot state
    rgb_tensor = robot_state[camera_key]["rgb"]
    img = array_to_pil(rgb_tensor)

    # Save image with sequential naming
    img_filename = os.path.join(traj_folder, f"{data['image_counter']}.jpg")
    img.save(img_filename)

    # Extract odometry (assuming robot_state["proprio"] holds [x, y, ... , yaw, ...])
    # Adjust indices based on your simulator’s output.
    proprio = robot_state["proprio"]
    pos = np.array(proprio[:2])
    yaw = float(proprio[3])  # or another index if yaw is stored elsewhere
    data["positions"].append(pos)
    data["yaws"].append(yaw)

    data["image_counter"] += 1


def end_trajectory(traj_folder, data):
    # Convert lists to numpy arrays and save as pickle
    traj_data = {"position": np.array(data["positions"]), "yaw": np.array(data["yaws"])}
    with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
        pickle.dump(traj_data, f)
    print(f"Saved trajectory data to {traj_folder}")


from omnigibson.utils.ui_utils import KeyboardRobotController


def run_control_loop(env, robot, camera_key, traj_folder, data, max_steps=1000):
    # Create a keyboard controller (or use your own method)
    controller = KeyboardRobotController(robot=robot)
    controller.print_keyboard_teleop_info()

    step = 0
    while step < max_steps:
        # Get control action from teleop
        action = controller.get_teleop_action()

        # Step the environment with the chosen action
        states, _, terminated, truncated, _ = env.step({robot.name: action})
        robot_state = states[robot.name]

        # Capture data from this step
        capture_step(traj_folder, data, robot_state, camera_key)
        print(
            f"Step {step}: Position: {robot_state['proprio'][:2]}, Yaw: {robot_state['proprio'][3]}"
        )

        step += 1

        if terminated or truncated:
            print("Episode ended.")
            break


def main():

    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    with open(config_filename, "r") as f:
        # config = yaml.safe_load(f)
        cfg = yaml.safe_load(f)
    cfg["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Set up OmniGibson (reuse config similar to your teleop example)
    # scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}
    # robot_cfg = {
    #     "type": "Turtlebot",
    #     "obs_modalities": ["rgb"],
    #     "action_type": "continuous",
    #     "action_normalize": True,
    # }
    # cfg = {"scene": scene_cfg, "robots": [robot_cfg]}
    env = og.Environment(configs=cfg)
    robot = env.robots[0]

    # Set camera key based on your robot configuration.
    # For example, it might be: "<robot_name>:eyes:Camera:0"
    camera_key = f"{robot.name}:eyes:Camera:0"

    # Initialize dataset directory and start a new trajectory
    dataset_root = init_dataset("./omnigibson_dataset")
    traj_id = int(time.time())  # or use a counter
    traj_folder, data = start_new_trajectory(dataset_root, traj_id)

    # Optionally, you can let the user control when to start/stop recording (e.g., using a key press)
    print(
        "Starting data collection. Use teleop to drive the robot. Press Ctrl+C to stop."
    )

    try:
        run_control_loop(env, robot, camera_key, traj_folder, data, max_steps=2000)
    except KeyboardInterrupt:
        print("Data collection interrupted by user.")

    # End the trajectory and save odometry data
    end_trajectory(traj_folder, data)

    # Clean up the environment
    og.clear()


if __name__ == "__main__":
    main()
