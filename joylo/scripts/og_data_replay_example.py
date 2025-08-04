from omnigibson.envs import SceneGraphDataPlaybackWrapper
from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.scene_graph_utils import SceneGraphWriter
import torch as th
import os
import omnigibson as og
from omnigibson.macros import gm
import argparse
import sys
import json
from gello.robots.sim_robot.og_teleop_utils import optimize_sim_settings
from gello.utils.qa_utils import *
from gello.utils.b1k_utils import ALL_QA_METRICS, COMMON_QA_METRICS, TASK_QA_METRICS
import inspect

import cv2
import multiprocessing as mp

RUN_QA = False

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128


def extract_arg_names(func):
    return list(inspect.signature(func).parameters.keys())


def save_frame(args):
    '''worker function for multiprocessing'''
    frame, frame_id, output_dir = args
    filename = os.path.join(output_dir, f"{frame_id:05d}.png")
    cv2.imwrite(filename, frame)

def decompose_video_parallel(video_path, output_folder, base_frame_id=0, chunk_size=600):
    '''Decompose a video into PNG frames using parallel processing with chunking to avoid memory issues'''
    if base_frame_id is None:
        base_frame_id = 0
    base_frame_id = base_frame_id + 1
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Get video properties
    vid_capture = cv2.VideoCapture(video_path)
    if not vid_capture.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    total_frames = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames}")
    
    num_cores = mp.cpu_count() - 2
    print(f"Using {num_cores} cores for video decomposition with chunk size: {chunk_size}")

    frame_id = base_frame_id
    processed_frames = 0
    
    while True:
        # Process frames in chunks to avoid memory explosion
        chunk_tasks = []
        
        # Read a chunk of frames
        for _ in range(chunk_size):
            success, frame = vid_capture.read()
            if not success:
                break
            chunk_tasks.append((frame.copy(), frame_id, output_folder))  # .copy() to avoid reference issues
            frame_id += 1
        
        if not chunk_tasks:
            break
            
        # Process this chunk in parallel
        with mp.Pool(processes=num_cores) as pool:
            pool.map(save_frame, chunk_tasks)
        
        processed_frames += len(chunk_tasks)
        print(f"Processed {processed_frames}/{total_frames} frames ({processed_frames/total_frames*100:.1f}%)")
        
        # Clear the chunk from memory
        del chunk_tasks
    
    # close video capture
    vid_capture.release()
    
    print(f"Successfully decomposed {processed_frames} frames into {output_folder}")

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
    # folder_path = os.path.dirname(hdf_input_path)

    # # Create the folder if it doesn't exist
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
        [[-0.4, 0, 2.0], [0.2706, -0.2706, -0.6533,  0.6533]],
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
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {
                "image_height": RESOLUTION_DEFAULT,
                "image_width": RESOLUTION_DEFAULT,
                "horizontal_aperture": 40.0,
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
        "modalities": ["rgb", "seg_instance_id", "seg_instance"],
        "sensor_kwargs": {
            "image_height": RESOLUTION_DEFAULT,
            "image_width": RESOLUTION_DEFAULT,
            "horizontal_aperture": 40.0,
        },
        "position": th.tensor([0.06, 0.0, 0.01], dtype=th.float32),
        "orientation": th.tensor([-1.0, 0.0, 0.0, 0.0], dtype=th.float32),
        "pose_frame": "parent",
    })

    # Create the environment
    additional_wrapper_configs = []
    if RUN_QA:
        additional_wrapper_configs.append({
            "type": "MetricsWrapper",
        })
    env = SceneGraphDataPlaybackWrapper.create_from_hdf5(
        input_path=hdf_input_path,
        output_path=hdf_output_path,
        robot_obs_modalities=["rgb", "seg_instance"],
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        exclude_sensor_names=["zed"],
        n_render_iterations=1,
        only_successes=False,
        additional_wrapper_configs=additional_wrapper_configs,
        include_task=True,
        include_task_obs=False,
        include_robot_control=False,
        include_contacts=True,
    )

    # Optimize rendering for faster speeds
    og.sim.add_callback_on_play("optimize_rendering", optimize_sim_settings)

    if RUN_QA:
        # Add QA metrics
        metric_kwargs = dict(
            step_dt=1/30,
            vel_threshold=0.001,
            color_arms=False,           # For ghost hand
            default_color=(0.8235, 0.8235, 1.0000),
            head_camera=env.external_sensors[f"external_sensor{len(env.external_sensors) - 1}"],
            head_camera_link_name="torso_link4",
            navigation_window=3.0,
            translation_threshold=0.1,
            rotation_threshold=0.05,
            camera_tilt_threshold=0.4,
            gripper_link_paths={
                "left":
                    set([
                        '/World/scene_0/controllable__r1pro__robot_r1/left_realsense_link/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/left_gripper_link/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/left_gripper_finger_link1/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/left_gripper_finger_link2/visuals'
                    ]),
                "right":
                    set([
                        '/World/scene_0/controllable__r1pro__robot_r1/right_realsense_link/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/right_gripper_link/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/right_gripper_finger_link1/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/right_gripper_finger_link2/visuals'
                    ])
            },
        )
        active_metrics_info = {metric_name: ALL_QA_METRICS[metric_name] for metric_name in COMMON_QA_METRICS}
        for metric_name, metric_info in active_metrics_info.items():
            create_fcn = metric_info["cls"] if metric_info["init"] is None else metric_info["init"]
            init_kwargs = {arg: metric_kwargs[arg] for arg in extract_arg_names(create_fcn)}
            metric = create_fcn(**init_kwargs)
            env.add_metric(name=metric_name, metric=metric)
        env.reset()

    # Create a list to store video writers and RGB keys
    video_writers = []
    video_rgb_keys = []

    frame_writers = []
    frame_rgb_keys = []
    
    # Create video writer for robot cameras
    # robot_camera_names = ['robot_r1::robot_r1:left_realsense_link:Camera:0::rgb', 
    #                     'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb']
    # for robot_camera_name in robot_camera_names:
    #     # video_writers.append(env.create_video_writer(fpath=f"{video_dir}/{robot_camera_name}.mp4"))
    #     # video_rgb_keys.append(robot_camera_name)
    #     frame_writers.append(env.create_frame_writer(output_dir=f"{video_dir}/{robot_camera_name}/"))
    #     frame_rgb_keys.append(robot_camera_name)
    #     pass

    # Create video writers for external cameras
    for i in range(len(external_sensors_config)):
        camera_name = f"external_sensor{i}"
        video_writers.append(env.create_video_writer(fpath=f"{video_dir}/{camera_name}.mp4"))
        video_rgb_keys.append(f"external::{camera_name}::rgb")
        frame_writers.append(env.create_frame_writer(output_dir=f"{video_dir}/{camera_name}/"))
        frame_rgb_keys.append(f"external::{camera_name}::rgb")
    
    # Playback the dataset with all video writers
    # We avoid calling playback_dataset and call playback_episode individually in order to manually
    # aggregate per-episode metrics
    metrics = dict()

    assert len(env.input_hdf5["data"].keys()) == 1, f"Only one episode is supported for now, got {len(env.input_hdf5['data'].keys())} from {hdf_input_path}"

    replay_config = {
        "record_visibility": True,
        "record_rgb_keys": ["external::external_sensor0::rgb", "external::external_sensor1::rgb"],
        "sensors": ["external_sensor0", "external_sensor1"],
    }

    start_frame = None
    end_frame = None

    for episode_id in range(env.input_hdf5["data"].attrs["n_episodes"]):
        scene_graph_writer = SceneGraphWriter(output_path=os.path.join(folder_path, f"scene_graph_{episode_id}.json"), interval=200, buffer_size=200, write_full_graph_only=True)
        env.playback_episode(
            episode_id=episode_id,
            record_data=False,
            video_writers=video_writers,
            video_rgb_keys=video_rgb_keys,
            frame_writers=None,
            frame_rgb_keys=None,
            start_frame=start_frame,
            end_frame=end_frame,
            scene_graph_writer=scene_graph_writer,
            replay_config=replay_config,
        )
    # Close all video writers
    for writer in video_writers:
        writer.close()
    
    # Decompose videos into frames
    for video_writer in video_writers:
        video_path = video_writer._filename
        output_folder = os.path.splitext(video_path)[0]
        decompose_video_parallel(video_path, output_folder, base_frame_id=start_frame)




    # Always clear the environment to free resources
    og.clear()
        
    print(f"Successfully processed {hdf_input_path}")


def main():

    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--dir", help="Directory containing HDF5 files to process")
    parser.add_argument("--files", nargs="*", help="Individual HDF5 file(s) to process")

    default_files = ["/home/mll-laptop-1/01_projects/03_behavior_challenge/sampled_demo/cleaning_up_plates_and_food_1747631958405370_cleaning_up_plates_and_food.hdf5"]
    # default_dir = "/home/mll-laptop-1/01_projects/03_behavior_challenge/raw_demos/Jul_2_demos/cleaning_up_plates_and_food_1747365183765658_cleaning_up_plates_and_food.hdf5"
    
    args = parser.parse_args()

    # args.files = default_files
    # args.dir = default_dir
    
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
