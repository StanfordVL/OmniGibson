import os
import pickle
import shutil
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm


def process_trajectory(
    traj_folder, output_folder, target_spacing=0.05, original_spacing=0.005
):
    """
    Process a trajectory folder to increase the spacing between waypoints
    and save to a new output folder.

    Args:
        traj_folder: Path to the original trajectory folder
        output_folder: Path to save the processed trajectory
        target_spacing: Target spacing in meters
        original_spacing: Original spacing in meters
    """
    # Extract trajectory name from the path
    traj_name = os.path.basename(traj_folder)

    # Create the output trajectory folder
    output_traj_folder = os.path.join(output_folder, traj_name)
    os.makedirs(output_traj_folder, exist_ok=True)

    # Calculate the sampling rate (how many points to skip)
    sampling_rate = int(round(target_spacing / original_spacing))

    # Load the trajectory data
    pkl_path = os.path.join(traj_folder, "traj_data.pkl")
    try:
        with open(pkl_path, "rb") as f:
            traj_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return

    # Sample the position and yaw data
    original_positions = traj_data["position"]
    original_yaws = traj_data["yaw"]

    # Sample every n-th point (where n is the sampling rate)
    sampled_indices = list(range(0, len(original_positions), sampling_rate))
    sampled_positions = original_positions[sampled_indices]
    sampled_yaws = original_yaws[sampled_indices]

    # Create the new trajectory data
    new_traj_data = {"position": sampled_positions, "yaw": sampled_yaws}

    # Save the new trajectory data
    new_pkl_path = os.path.join(output_traj_folder, "traj_data.pkl")
    with open(new_pkl_path, "wb") as f:
        pickle.dump(new_traj_data, f)

    # Process the images
    image_counter = 0
    for i in tqdm(sampled_indices, desc="Processing images"):
        # Copy the image with a new sequential number
        src_img_path = os.path.join(traj_folder, f"{i}.jpg")
        if os.path.exists(src_img_path):
            dst_img_path = os.path.join(output_traj_folder, f"{image_counter}.jpg")
            shutil.copy(src_img_path, dst_img_path)
            image_counter += 1

    print(
        f"Processed {traj_name}: {len(original_positions)} waypoints -> {len(sampled_positions)} waypoints"
    )
    print(f"New spacing should be approximately {original_spacing * sampling_rate:.3f}m")


def process_dataset(
    dataset_root, output_root, target_spacing=0.05, original_spacing=0.005
):
    """
    Process all trajectory folders in the dataset root and save to a new output folder.

    Args:
        dataset_root: Path to the original dataset root folder
        output_root: Path to save the processed dataset
        target_spacing: Target spacing in meters
        original_spacing: Original spacing in meters
    """
    # Create the output root folder
    os.makedirs(output_root, exist_ok=True)

    # Get all trajectory folders
    traj_folders = [
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("trajectory_")
    ]

    for folder in traj_folders:
        folder_path = os.path.join(dataset_root, folder)
        process_trajectory(folder_path, output_root, target_spacing, original_spacing)


def main():
    parser = argparse.ArgumentParser(
        description="Process dataset to increase waypoint spacing."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./omnigibson_dataset",
        help="Root directory of the original dataset.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./omnigibson_dataset_sampled",
        help="Root directory to save the processed dataset.",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        default=0.05,
        help="Target spacing in meters.",
    )
    parser.add_argument(
        "--original_spacing",
        type=float,
        default=0.005,
        help="Original spacing in meters.",
    )
    args = parser.parse_args()

    process_dataset(
        args.dataset_root, args.output_root, args.target_spacing, args.original_spacing
    )
    print("\nDataset processing complete.")
    print(f"Original dataset: {args.dataset_root}")
    print(f"Processed dataset: {args.output_root}")
    print(f"Original spacing: {args.original_spacing}m")
    print(f"Target spacing: {args.target_spacing}m")
    print(
        "You can verify the new spacing by running estimate_waypoint_spacing.py on the processed dataset."
    )


if __name__ == "__main__":
    main()
