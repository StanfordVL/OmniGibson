from omnigibson.envs import DataPlaybackWrapper
from omnigibson.envs import EnvMetric
from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.utils.constants import STRUCTURE_CATEGORIES
import torch as th
from omnigibson.macros import gm
import os
import omnigibson as og
from omnigibson.macros import gm
import argparse
import sys
import json
from gello.robots.sim_robot.og_teleop_utils import optimize_sim_settings

RUN_QA = True

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128


class MotionMetric(EnvMetric):

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            # Record velocity (we'll derive accel -> jerk at the end of the episode)
            step_metrics[f"robot{i}::pos"] = robot.get_joint_positions()
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Derive acceleration -> jerk based on the recorded velocities
        episode_metrics = dict()
        for pos_key, positions in episode_info.items():
            positions = th.stack(positions, dim=0)
            vels = (positions[1:] - positions[:-1]) / og.sim.get_sim_step_dt()
            accs = (vels[1:] - vels[:-1]) / og.sim.get_sim_step_dt()
            jerks = (accs[1:] - accs[:-1]) / og.sim.get_sim_step_dt()
            episode_metrics[f"{pos_key}::max_vel"] = vels.max().item()
            episode_metrics[f"{pos_key}::max_acc"] = accs.max().item()
            episode_metrics[f"{pos_key}::max_jerk"] = jerks.max().item()

        return episode_metrics


class CollisionMetric(EnvMetric):
    def __init__(self):
        self.checks = dict()
        super().__init__()

    def add_check(self, name, check):
        """
        Adds a collision check to this metric, which can be queried by @name

        Args:
            name (str): name of the check
            check (function): Collision checker function, with the following signature:

                def check(env: Environment) -> bool

                which should return True if there is collision, else False
        """
        self.checks[name] = check

    def remove_check(self, name):
        """
        Removes check with corresponding @name

        Args:
            name (str): name of the check to remove
        """
        self.checks.pop(name)

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        for name, check in self.checks.items():
            step_metrics[f"{name}"] = check(env)
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Derive acceleration -> jerk based on the recorded velocities
        episode_metrics = dict()
        for name, collisions in episode_info.items():
            collisions = th.tensor(collisions)
            episode_metrics[f"{name}::n_collision"] = collisions.sum().item()

        return episode_metrics



class TaskSuccessMetric(EnvMetric):
    def __init__(self):
        super().__init__()

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        # Record whether task is done (terminated is true but not truncated)
        return {"done": terminated and not truncated}

    def _compute_episode_metrics(self, env, episode_info):
        # Derive acceleration -> jerk based on the recorded velocities
        return {"success": th.any(th.tensor(episode_info["done"])).item()}


def check_robot_self_collision(env):
    # TODO: What about gripper finger self collision?
    for robot in env.robots:
        link_paths = robot.link_prim_paths
        if RigidContactAPI.in_contact(link_paths, link_paths):
            return True
    return False


def check_robot_base_nonarm_nonfloor_collision(env):
    # TODO: How to check for wall collisions? They're kinematic only
    # # One solution: Make them non-kinematic only during QA checking
    # floor_link_paths = []
    # for structure_category in STRUCTURE_CATEGORIES:
    #     for structure in env.scene.object_registry("category", structure_category):
    #         floor_link_paths += structure.link_prim_paths
    # floor_link_col_idxs = {RigidContactAPI.get_body_col_idx(link_path) for link_path in floor_link_paths}

    for robot in env.robots:
        robot_link_paths = set(robot.link_prim_paths)
        for arm in robot.arm_names:
            robot_link_paths -= set(robot.arm_link_names[arm])
            robot_link_paths -= set(robot.gripper_link_names[arm])
            robot_link_paths -= set(robot.finger_link_names[arm])
    robot_link_idxs = [RigidContactAPI.get_body_col_idx(link_path) for link_path in robot_link_paths]
    robot_contacts = RigidContactAPI.get_all_impulses(env.scene.idx)[robot_link_idxs]

    # col_idxs = th.tensor(tuple(set(len(RigidContactAPI._COL_IDX_TO_PATH)) - floor_link_col_idxs))
    # robot_contacts = robot_contacts[:, col_idxs]

    return th.any(robot_contacts).item()


def replay_hdf5_file(hdf_input_path):
    """
    Replays a single HDF5 file and saves videos to a new folder
    
    Args:
        hdf_input_path: Path to the HDF5 file to replay
    """
    # Create folder with same name as HDF5 file (without extension)
    base_name = os.path.basename(hdf_input_path)
    folder_name = os.path.splitext(base_name)[0]
    folder_path = os.path.join(os.path.dirname(hdf_input_path), folder_name)
    
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Define output paths
    hdf_output_path = os.path.join(folder_path, f"{folder_name}_replay.hdf5")
    video_dir = folder_path

    # Metrics path
    metrics_output_path = os.path.join(folder_path, f"qa_metrics.json")

    # Move original HDF5 file to the new folder
    new_hdf_input_path = os.path.join(folder_path, base_name)
    if hdf_input_path != new_hdf_input_path:  # Avoid copying if already in target folder
        os.rename(hdf_input_path, new_hdf_input_path)
        hdf_input_path = new_hdf_input_path
    
    # Define resolution for consistency
    RESOLUTION_DEFAULT = 560
    RESOLUTION_WRIST = 240
    
    # This flag is needed to run data playback wrapper
    gm.ENABLE_TRANSITION_RULES = False
    
    # Define external camera positions and orientations
    external_camera_poses = [
        # Camera 1
        [[-0.4, 0, 2.0], [0.369, -0.369, -0.603, 0.603]],
        # # Camera 2
        # [[-0.2, 0.6, 2.0], [-0.1930, 0.4163, 0.8062, -0.3734]],
        # # Camera 3
        # [[-0.2, -0.6, 2.0], [0.4164, -0.1929, -0.3737, 0.8060]]
    ]
    
    # Robot sensor configuration
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_height": RESOLUTION_WRIST,
                "image_width": RESOLUTION_WRIST,
            },
        },
    }
    
    # Generate external sensors config automatically
    external_sensors_config = []
    for i, (position, orientation) in enumerate(external_camera_poses):
        external_sensors_config.append({
            "sensor_type": "VisionSensor",
            "name": f"external_sensor{i}",
            "relative_prim_path": f"/controllable__r1pro__robot_r1/base_link/external_sensor{i}",
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_height": RESOLUTION_DEFAULT,
                "image_width": RESOLUTION_DEFAULT,
            },
            "position": th.tensor(position, dtype=th.float32),
            "orientation": th.tensor(orientation, dtype=th.float32),
            "pose_frame": "parent",
        })

    # Replace normal head camera with custom config
    idx = len(external_sensors_config)
    external_sensors_config.append({
        "sensor_type": "VisionSensor",
        "name": f"external_sensor{idx}",
        "relative_prim_path": f"/controllable__r1pro__robot_r1/zed_link/external_sensor{idx}",
        "modalities": ["rgb"],
        "sensor_kwargs": {
            "image_height": RESOLUTION_DEFAULT,
            "image_width": RESOLUTION_DEFAULT,
            "horizontal_aperture": 40.0,
        },
        "position": th.tensor([0.06, 0, 0.4525], dtype=th.float32),
        "orientation": th.tensor([-0.98481, 0, 0, 0.17365], dtype=th.float32),
        "pose_frame": "parent",
    })

    # Create the environment
    additional_wrapper_configs = []
    if RUN_QA:
        additional_wrapper_configs.append({
            "type": "MetricsWrapper",
        })
    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=hdf_input_path,
        output_path=hdf_output_path,
        robot_obs_modalities=["rgb"],
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        exclude_sensor_names=["zed"],
        n_render_iterations=1,
        only_successes=False,
        additional_wrapper_configs=additional_wrapper_configs,
        include_task=False,
        include_task_obs=False,
        include_robot_control=False,
        include_contacts=True,
    )

    # Optimize rendering for faster speeds
    og.sim.add_callback_on_play("optimize_rendering", optimize_sim_settings)

    if RUN_QA:
        # Add QA metrics
        env.add_metric(name="success", metric=TaskSuccessMetric())
        env.add_metric(name="jerk", metric=MotionMetric())

        col_metric = CollisionMetric()
        col_metric.add_check(name="robot_self", check=check_robot_self_collision)
        col_metric.add_check(name="robot_nonarm_nonstructure", check=check_robot_base_nonarm_nonfloor_collision)
        env.add_metric(name="collision", metric=col_metric)
        env.reset()

    # Create a list to store video writers and RGB keys
    video_writers = []
    video_rgb_keys = []
    
    # Create video writer for robot cameras
    robot_camera_names = ['robot_r1::robot_r1:left_realsense_link:Camera:0::rgb', 
                        'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb']
    for robot_camera_name in robot_camera_names:
        video_writers.append(env.create_video_writer(fpath=f"{video_dir}/{robot_camera_name}.mp4"))
        video_rgb_keys.append(robot_camera_name)
    
    # Create video writers for external cameras
    for i in range(len(external_sensors_config)):
        camera_name = f"external_sensor{i}"
        video_writers.append(env.create_video_writer(fpath=f"{video_dir}/{camera_name}.mp4"))
        video_rgb_keys.append(f"external::{camera_name}::rgb")
    
    # Playback the dataset with all video writers
    # We avoid calling playback_dataset and call playback_episode individually in order to manually
    # aggregate per-episode metrics
    metrics = dict()
    for episode_id in range(env.input_hdf5["data"].attrs["n_episodes"]):
        env.playback_episode(
            episode_id=episode_id,
            record_data=False,
            video_writers=video_writers,
            video_rgb_keys=video_rgb_keys,
        )
        episode_metrics = env.aggregate_metrics(flatten=True)
        for k, v in episode_metrics.items():
            print(f"Metric [{k}]: {v}")
        metrics[f"episode_{episode_id}"] = episode_metrics
    
    # Close all video writers
    for writer in video_writers:
        writer.close()

    env.save_data()

    # Save metrics
    with open(metrics_output_path, "w+") as f:
        json.dump(metrics, f, indent=4)

    # Always clear the environment to free resources
    og.clear()
        
    print(f"Successfully processed {hdf_input_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--dir", help="Directory containing HDF5 files to process")
    parser.add_argument("--files", nargs="*", help="Individual HDF5 file(s) to process")
    
    args = parser.parse_args()
    
    if args.dir and os.path.isdir(args.dir):
        # Process all HDF5 files in the directory (non-recursively)
        hdf_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                    if f.lower().endswith('.hdf5') and os.path.isfile(os.path.join(args.dir, f))]
        
        if not hdf_files:
            print(f"No HDF5 files found in directory: {args.dir}")
        else:
            print(f"Found {len(hdf_files)} HDF5 files to process")
    elif args.files:
        # Process individual files specified
        hdf_files = args.files
    else:
        parser.print_help()
        print("\nError: Either --dir or --files must be specified", file=sys.stderr)
        return
    
    # Process each file
    for hdf_file in hdf_files:
        if not os.path.exists(hdf_file):
            print(f"Error: File {hdf_file} does not exist", file=sys.stderr)
            continue
            
        replay_hdf5_file(hdf_file)

    og.shutdown()


if __name__ == "__main__":
    main()