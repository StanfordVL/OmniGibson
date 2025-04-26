from omnigibson.envs import DataPlaybackWrapper
import torch as th
from omnigibson.macros import gm
import os
import omnigibson as og
import argparse
import sys

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
    
    # Move original HDF5 file to the new folder
    new_hdf_input_path = os.path.join(folder_path, base_name)
    if hdf_input_path != new_hdf_input_path:  # Avoid copying if already in target folder
        os.rename(hdf_input_path, new_hdf_input_path)
        hdf_input_path = new_hdf_input_path
    
    # Define resolution for consistency
    RESOLUTION = 1080
    
    # This flag is needed to run data playback wrapper
    gm.ENABLE_TRANSITION_RULES = False
    
    # Define external camera positions and orientations
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
        input_path=hdf_input_path,
        output_path=hdf_output_path,
        robot_obs_modalities=["rgb"],
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        n_render_iterations=1,
        only_successes=False,
    )
    
    # Create a list to store video writers and RGB keys
    video_writers = []
    video_rgb_keys = []
    
    # Create video writer for robot cameras
    robot_camera_names = ['robot_r1::robot_r1:left_realsense_link:Camera:0::rgb', 
                        'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb', 
                        'robot_r1::robot_r1:zed_link:Camera:0::rgb']
    for robot_camera_name in robot_camera_names:
        video_writers.append(env.create_video_writer(fpath=f"{video_dir}/{robot_camera_name}.mp4"))
        video_rgb_keys.append(robot_camera_name)
    
    # Create video writers for external cameras
    for i in range(len(external_camera_poses)):
        camera_name = f"external_sensor{i}"
        video_writers.append(env.create_video_writer(fpath=f"{video_dir}/{camera_name}.mp4"))
        video_rgb_keys.append(f"external::{camera_name}::rgb")
    
    # Playback the dataset with all video writers
    env.playback_dataset(record_data=False, video_writers=video_writers, video_rgb_keys=video_rgb_keys)
    
    # Close all video writers
    for writer in video_writers:
        writer.close()

    env.save_data()        

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


if __name__ == "__main__":
    main()