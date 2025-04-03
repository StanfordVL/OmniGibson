from omnigibson.envs import DataPlaybackWrapper
import torch as th
from omnigibson.macros import gm

HDF_INPUT_PATH = "/home/yhang/og-gello/data/example_branch_traj.hdf5"
HDF_OUTPUT_PATH = "/home/yhang/og-gello/data/example_branch_traj_replay.hdf5"
VIDEO_DIR = "/home/yhang/og-gello/data/example_branch_traj_video"

# Define resolution for consistency
RESOLUTION = 1080

# This flag is needed to run data playback wrapper
gm.ENABLE_TRANSITION_RULES = False

# Define external camera positions and orientations
# Format: [position_xyz, orientation_quaternion]
external_camera_poses = [
    # Camera 1
    [[-0.4, 0, 2.0], [0.369, -0.369, -0.603, 0.603]],
    # Camera 2
    [[-0.2, 0.6, 2.0], [-0.1930, 0.4163, 0.8062, -0.3734]],
    # Camera 3
    [[-0.2, -0.6, 2.0], [0.4164, -0.1929, -0.3737, 0.8060]]
]

# Robot sensor configuration
robot_sensor_config = {
    "VisionSensor": {
        "modalities": ["rgb"],
        "sensor_kwargs": {
            "image_height": RESOLUTION,
            "image_width": RESOLUTION,
        },
    },
}

# Generate external sensors config automatically
external_sensors_config = []
for i, (position, orientation) in enumerate(external_camera_poses):
    external_sensors_config.append({
        "sensor_type": "VisionSensor",
        "name": f"external_sensor{i}",
        "relative_prim_path": f"/controllable__r1__robot_r1/base_link/external_sensor{i}",
        "modalities": ["rgb"],
        "sensor_kwargs": {
            "image_height": RESOLUTION,
            "image_width": RESOLUTION,
        },
        "position": th.tensor(position, dtype=th.float32),
        "orientation": th.tensor(orientation, dtype=th.float32),
        "pose_frame": "parent",
    })

# Create the environment
env = DataPlaybackWrapper.create_from_hdf5(
    input_path=HDF_INPUT_PATH,
    output_path=HDF_OUTPUT_PATH,
    robot_obs_modalities=["rgb"],
    robot_sensor_config=robot_sensor_config,
    external_sensors_config=external_sensors_config,
    n_render_iterations=1,
    only_successes=False,
)

# Path prefix for video outputs
video_path_prefix = VIDEO_DIR

# Create a list to store video writers and RGB keys
video_writers = []
video_rgb_keys = []

# Create video writer for robot cameras
robot_camera_names = ['robot_r1::robot_r1:left_eef_link:Camera:0::rgb', 
                      'robot_r1::robot_r1:right_eef_link:Camera:0::rgb', 
                      'robot_r1::robot_r1:eyes:Camera:0::rgb']
for robot_camera_name in robot_camera_names:
    video_writers.append(env.create_video_writer(fpath=f"{video_path_prefix}/{robot_camera_name}.mp4"))
    video_rgb_keys.append(robot_camera_name)

# Create video writers for external cameras
for i in range(len(external_camera_poses)):
    camera_name = f"external_sensor{i}"
    video_writers.append(env.create_video_writer(fpath=f"{video_path_prefix}/{camera_name}.mp4"))
    video_rgb_keys.append(f"external::{camera_name}::rgb")

# Playback the dataset with all video writers
env.playback_dataset(record_data=False, video_writers=video_writers, video_rgb_keys=video_rgb_keys)

# Close all video writers
for writer in video_writers:
    writer.close()