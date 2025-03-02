import os
import pickle
import argparse
import numpy as np


def compute_waypoint_spacing(traj_folder):
    """
    Load the traj_data.pkl file from a trajectory folder and compute the average spacing
    between consecutive waypoints (positions).

    Returns:
        float: average spacing (meters) for the trajectory, or None if not computable.
    """
    pkl_path = os.path.join(traj_folder, "traj_data.pkl")
    try:
        with open(pkl_path, "rb") as f:
            traj_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None

    if "position" not in traj_data:
        print(f"traj_data.pkl in {traj_folder} is missing the 'position' key.")
        return None

    positions = traj_data["position"]  # Expected shape: [T, 2]
    if positions.shape[0] < 2:
        print(f"Not enough position data in {traj_folder} to compute spacing.")
        return None

    # Compute Euclidean distances between consecutive positions
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    for i, dist in enumerate(distances):
        print(f"Waypoint {i}: {dist:.3f} m")

    avg_spacing = np.mean(distances)
    return avg_spacing


def estimate_dataset_spacing(dataset_root):
    """
    Iterate through each trajectory folder in the dataset root, compute the average waypoint
    spacing for each, and then compute the overall average spacing.

    Returns:
        float: overall average spacing, or None if no valid trajectories found.
    """
    traj_folders = [
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]
    spacings = []
    for folder in traj_folders:
        folder_path = os.path.join(dataset_root, folder)
        spacing = compute_waypoint_spacing(folder_path)
        if spacing is not None:
            spacings.append(spacing)
            print(f"Trajectory {folder}: avg spacing = {spacing:.3f} m")
        else:
            print(f"Skipping trajectory {folder} due to insufficient data.")

    if len(spacings) == 0:
        return None

    overall_avg = np.mean(spacings)
    return overall_avg


def main():
    parser = argparse.ArgumentParser(
        description="Estimate metric_waypoint_spacing from collected trajectory data."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./omnigibson_dataset",
        help="Root directory of the collected dataset (each subfolder is a trajectory).",
    )
    args = parser.parse_args()

    overall_avg = estimate_dataset_spacing(args.dataset_root)
    if overall_avg is not None:
        print(
            "\nEstimated metric_waypoint_spacing for the dataset: {:.3f} m".format(
                overall_avg
            )
        )
        print("\nYou can update your data_config.yaml with the following entry:")
        print("omnigibson_dataset:")
        print("  metric_waypoint_spacing: {:.3f}".format(overall_avg))
    else:
        print("No valid trajectory data found to estimate spacing.")


if __name__ == "__main__":
    main()
