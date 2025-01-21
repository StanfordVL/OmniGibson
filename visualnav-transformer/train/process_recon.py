import h5py
import os
import pickle
from PIL import Image
import io
import argparse
import tqdm


def main(args: argparse.Namespace):
    recon_dir = os.path.join(args.input_dir, "recon_release")
    output_dir = args.output_dir

    # create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get all the folders in the recon dataset
    filenames = os.listdir(recon_dir)
    if args.num_trajs >= 0:
        filenames = filenames[: args.num_trajs]

    # processing loop
    for filename in tqdm.tqdm(filenames, desc="Trajectories processed"):
        # extract the name without the extension
        traj_name = filename.split(".")[0]
        # load the hdf5 file
        try:
            h5_f = h5py.File(os.path.join(recon_dir, filename), "r")
        except OSError:
            print(f"Error loading {filename}. Skipping...")
            continue
        # extract the position and yaw data
        position_data = h5_f["jackal"]["position"][:, :2]
        yaw_data = h5_f["jackal"]["yaw"][()]
        # save the data to a dictionary
        traj_data = {"position": position_data, "yaw": yaw_data}
        traj_folder = os.path.join(output_dir, traj_name)
        os.makedirs(traj_folder, exist_ok=True)
        with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)
        # make a folder for the file
        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)
        # save the image data to disk
        for i in range(h5_f["images"]["rgb_left"].shape[0]):
            img = Image.open(io.BytesIO(h5_f["images"]["rgb_left"][i]))
            img.save(os.path.join(traj_folder, f"{i}.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the recon_dataset",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="datasets/recon/",
        type=str,
        help="path for processed recon dataset (default: datasets/recon/)",
    )
    # number of trajs to process
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of trajectories to process (default: -1, all)",
    )

    args = parser.parse_args()
    print("STARTING PROCESSING RECON DATASET")
    main(args)
    print("FINISHED PROCESSING RECON DATASET")
