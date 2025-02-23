import os
import pickle
import argparse
import random
import shutil
import numpy as np


def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def verify_trajectory(traj_folder):
    """
    Verify that a trajectory folder has:
      - At least one JPEG image file.
      - A 'traj_data.pkl' file.
      - Consistency between the number of images and the odometry entries.
    """
    # List all JPEG files
    image_files = sorted([f for f in os.listdir(traj_folder) if f.endswith(".jpg")])
    pkl_file = os.path.join(traj_folder, "traj_data.pkl")

    if not image_files:
        return False, "No image files found."
    if not os.path.exists(pkl_file):
        return False, "traj_data.pkl file not found."

    try:
        with open(pkl_file, "rb") as f:
            traj_data = pickle.load(f)
    except Exception as e:
        return False, f"Failed to load traj_data.pkl: {e}"

    # Check if required keys exist
    if "position" not in traj_data or "yaw" not in traj_data:
        return False, "Missing required keys ('position' and 'yaw')."

    num_images = len(image_files)
    num_positions = traj_data["position"].shape[0]
    num_yaws = traj_data["yaw"].shape[0]

    if num_images != num_positions or num_images != num_yaws:
        return False, (
            f"Count mismatch: {num_images} images vs "
            f"{num_positions} positions and {num_yaws} yaws."
        )

    return True, f"Valid trajectory with {num_images} images."


def verify_dataset(dataset_root):
    """
    Iterate over each subfolder in dataset_root and verify its structure.
    Returns a list of valid trajectory folder names.
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
    Shuffles and splits the list of valid trajectories into training and testing sets.
    Writes the folder names into 'traj_names.txt' files under train and test directories.
    """
    random.shuffle(valid_trajs)
    split_index = int(len(valid_trajs) * train_ratio)
    train_trajs = valid_trajs[:split_index]
    test_trajs = valid_trajs[split_index:]

    # Create directories for splits following the repository's structure
    dataset_name = os.path.basename(os.path.abspath(dataset_root))
    train_dir = os.path.join(output_dir, dataset_name, "train")
    test_dir = os.path.join(output_dir, dataset_name, "test")

    for d in [train_dir, test_dir]:
        if os.path.exists(d):
            print(f"Clearing files from {d} for new data split")
            remove_files_in_dir(d)
        else:
            print(f"Creating directory: {d}")
            os.makedirs(d)

    # Write the trajectory folder names to text files
    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        for traj in train_trajs:
            f.write(traj + "\n")
    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        for traj in test_trajs:
            f.write(traj + "\n")

    print(
        f"Organized {len(train_trajs)} training and {len(test_trajs)} testing trajectories."
    )
    print(f"Train list saved at: {os.path.join(train_dir, 'traj_names.txt')}")
    print(f"Test list saved at: {os.path.join(test_dir, 'traj_names.txt')}")


def main():

    parser = argparse.ArgumentParser(
        description="Verify and organize collected trajectory data for NoMaD training."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./omnigibson_dataset",
        help="Root directory of the collected dataset (each subfolder is a trajectory).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of trajectories to use for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/data_splits",
        help="Directory where the train/test split files will be stored.",
    )
    args = parser.parse_args()

    print(f"Verifying dataset in {args.dataset_root}...")
    valid_trajs = verify_dataset(args.dataset_root)
    print(f"Found {len(valid_trajs)} valid trajectories out of total trajectories.")

    if not valid_trajs:
        print("No valid trajectories found. Please check your data collection process.")
        return

    organize_data_splits(
        args.dataset_root, valid_trajs, args.train_ratio, args.output_dir
    )


if __name__ == "__main__":
    main()
