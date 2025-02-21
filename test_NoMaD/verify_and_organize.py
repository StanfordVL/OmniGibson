import os
import pickle
import argparse
import random
import numpy as np


def verify_trajectory(traj_folder):
    """
    Verifies that a trajectory folder has the expected structure:
      - Contains at least one .jpg image file.
      - Contains a 'traj_data.pkl' file.
      - The number of images should match the number of entries in the odometry data.

    Returns:
        (bool, str): Whether the trajectory is valid, and a message.
    """
    # Find all .jpg files
    jpg_files = sorted([f for f in os.listdir(traj_folder) if f.endswith(".jpg")])
    pkl_path = os.path.join(traj_folder, "traj_data.pkl")

    if not jpg_files:
        return False, "No image files found."
    if not os.path.exists(pkl_path):
        return False, "traj_data.pkl file not found."

    # Load the pickle file
    try:
        with open(pkl_path, "rb") as f:
            traj_data = pickle.load(f)
    except Exception as e:
        return False, f"Failed to load pickle: {e}"

    # Check required keys
    if "position" not in traj_data or "yaw" not in traj_data:
        return False, "traj_data.pkl missing required keys ('position' and 'yaw')."

    # Verify that the number of images matches the odometry data length
    num_images = len(jpg_files)
    num_positions = traj_data["position"].shape[0]
    num_yaws = traj_data["yaw"].shape[0]

    if num_images != num_positions or num_images != num_yaws:
        return False, (
            f"Mismatch in counts: {num_images} images, "
            f"{num_positions} positions, {num_yaws} yaws."
        )

    return True, f"Valid trajectory with {num_images} images."


def verify_dataset(dataset_root):
    """
    Iterates through all subdirectories in dataset_root (each expected to be a trajectory),
    verifies them, and returns a list of valid trajectory folder names.
    """
    traj_folders = [
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    valid_trajs = []
    for folder in traj_folders:
        folder_path = os.path.join(dataset_root, folder)
        valid, message = verify_trajectory(folder_path)
        print(f"Verifying {folder}: {message}")
        if valid:
            valid_trajs.append(folder)
    return valid_trajs


def organize_data_splits(
    dataset_root, valid_trajs, train_ratio=0.8, output_dir="./data/data_splits"
):
    """
    Shuffles and splits the list of valid trajectory folders into training and testing sets.
    Writes the names to 'traj_names.txt' under 'train' and 'test' subdirectories.
    """
    random.shuffle(valid_trajs)
    num_train = int(len(valid_trajs) * train_ratio)
    train_trajs = valid_trajs[:num_train]
    test_trajs = valid_trajs[num_train:]

    # Create output directories
    dataset_name = os.path.basename(os.path.abspath(dataset_root))
    split_dir = os.path.join(output_dir, dataset_name)
    train_dir = os.path.join(split_dir, "train")
    test_dir = os.path.join(split_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Write trajectory names to text files
    train_file = os.path.join(train_dir, "traj_names.txt")
    test_file = os.path.join(test_dir, "traj_names.txt")

    with open(train_file, "w") as f:
        for traj in train_trajs:
            f.write(traj + "\n")

    with open(test_file, "w") as f:
        for traj in test_trajs:
            f.write(traj + "\n")

    print(
        f"Organized {len(train_trajs)} training and {len(test_trajs)} testing trajectories."
    )
    print(f"Train list saved at: {train_file}")
    print(f"Test list saved at: {test_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and organize collected dataset for NoMaD training."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./scene_dataset",
        help="Root directory of the collected dataset (each subfolder is a trajectory).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of trajectories to use for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/data_splits",
        help="Directory where the data split files will be stored.",
    )
    args = parser.parse_args()

    print(f"Verifying dataset in {args.dataset_root}...")
    valid_trajs = verify_dataset(args.dataset_root)
    print(f"Found {len(valid_trajs)} valid trajectories out of total trajectories.")

    if not valid_trajs:
        print(
            "No valid trajectories found. Please check your dataset collection process."
        )
        return

    organize_data_splits(
        args.dataset_root, valid_trajs, args.train_ratio, args.output_dir
    )


if __name__ == "__main__":
    main()
