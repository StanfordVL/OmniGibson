import argparse
import os
import shutil
import random


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


def main(args: argparse.Namespace):
    # Get the names of the folders in the data directory that contain the file 'traj_data.pkl'
    folder_names = [
        f
        for f in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, f))
        and "traj_data.pkl" in os.listdir(os.path.join(args.data_dir, f))
    ]

    # Randomly shuffle the names of the folders
    random.shuffle(folder_names)

    # Split the names of the folders into train and test sets
    split_index = int(args.split * len(folder_names))
    train_folder_names = folder_names[:split_index]
    test_folder_names = folder_names[split_index:]

    # Create directories for the train and test sets
    train_dir = os.path.join(args.data_splits_dir, args.dataset_name, "train")
    test_dir = os.path.join(args.data_splits_dir, args.dataset_name, "test")
    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            print(f"Clearing files from {dir_path} for new data split")
            remove_files_in_dir(dir_path)
        else:
            print(f"Creating {dir_path}")
            os.makedirs(dir_path)

    # Write the names of the train and test folders to files
    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        for folder_name in train_folder_names:
            f.write(folder_name + "\n")

    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        for folder_name in test_folder_names:
            f.write(folder_name + "\n")


if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir", "-i", help="Directory containing the data", required=True
    )
    parser.add_argument(
        "--dataset-name", "-d", help="Name of the dataset", required=True
    )
    parser.add_argument(
        "--split", "-s", type=float, default=0.8, help="Train/test split (default: 0.8)"
    )
    parser.add_argument(
        "--data-splits-dir", "-o", default="vint_train/data/data_splits", help="Data splits directory"
    )
    args = parser.parse_args()
    main(args)
    print("Done")
