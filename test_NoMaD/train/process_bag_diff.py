
import os
import pickle
from PIL import Image
import io
import argparse
import tqdm
import yaml
import rosbag

# utils
from vint_train.process_data.process_data_utils import *


def main(args: argparse.Namespace):

    # load the config file
    with open("vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # iterate recurisively through all the folders and get the path of files with .bag extension in the args.input_dir
    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag") and "diff" in file:
                bag_files.append(os.path.join(root, file))
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]

    # processing loop
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            b = rosbag.Bag(bag_path)
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # name is that folders separated by _ and then the last part of the path
        traj_name = "_".join(bag_path.split("/")[-2:])[:-4]

        # load the bag file
        bag_img_data, bag_traj_data = get_images_and_odom_2(
            b,
            ['/usb_cam_front/image_raw', '/chosen_subgoal'],
            ['/odom'],
            rate=args.sample_rate,
        )
  
        if bag_img_data is None:
            print(
                f"{bag_path} did not have the topics we were looking for. Skipping..."
            )
            continue
        # remove backwards movement
        # cut_trajs = filter_backwards(bag_img_data, bag_traj_data)

        # for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
        #     traj_name_i = traj_name + f"_{i}"
        #     traj_folder_i = os.path.join(args.output_dir, traj_name_i)
        #     # make a folder for the traj
        #     if not os.path.exists(traj_folder_i):
        #         os.makedirs(traj_folder_i)
        #     with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
        #         pickle.dump(traj_data_i, f)
        #     # save the image data to disk
        #     for i, img in enumerate(img_data_i):
        #         img.save(os.path.join(traj_folder_i, f"{i}.jpg"))

        traj_folder = os.path.join(args.output_dir, traj_name)
        if not os.path.exists(traj_folder):
                os.makedirs(traj_folder)
        
        obs_images = bag_img_data["/usb_cam_front/image_raw"]
        diff_images = bag_img_data["/chosen_subgoal"]
        for i, img_data in enumerate(zip(obs_images, diff_images)):
            obs_image, diff_image = img_data
            # save the image data to disk
            # save the image data to disk
            obs_image.save(os.path.join(traj_folder, f"{i}.jpg"))
            diff_image.save(os.path.join(traj_folder, f"diff_{i}.jpg"))

        with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
                pickle.dump(bag_traj_data['/odom'], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    # add dataset name
    # parser.add_argument(
    #     "--dataset-name",
    #     "-d",
    #     type=str,
    #     help="name of the dataset (must be in process_config.yaml)",
    #     default="tartan_drive",
    #     required=True,
    # )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the datasets with rosbags",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../datasets/tartan_drive/",
        type=str,
        help="path for processed dataset (default: ../datasets/tartan_drive/)",
    )
    # number of trajs to process
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of bags to process (default: -1, all)",
    )
    # sampling rate
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )

    args = parser.parse_args()
    # all caps for the dataset name
    print(f"STARTING PROCESSING DIFF DATASET")
    main(args)
    print(f"FINISHED PROCESSING DIFF DATASET")
