from omnigibson.utils.mvs_utils import batch_np_matrix_to_pycolmap_wo_track, create_pixel_coordinate_grid, build_pycolmap_intri
import omnigibson.utils.camera_utils as ilutils
import omnigibson.utils.transform_utils as T
import os
import trimesh
import h5py
import numpy as np
import torch
import json
import tqdm

def hdf5_to_colmap(scene_dir):
  images_dir = os.path.join(scene_dir, "images")
  if not os.path.exists(images_dir):
      raise FileNotFoundError(f"Images directory {images_dir} does not exist. Please run the data collection script first.")

  # Open the HDF5 file in read mode
  print("Loading HDF5 file...")
  hdf5_file_path = os.path.join(scene_dir, "camera_data.hdf5")
  if not os.path.exists(hdf5_file_path):
      raise FileNotFoundError(f"HDF5 file {hdf5_file_path} does not exist. Please run the data collection script first.")
  with h5py.File(hdf5_file_path, 'r') as f:
      # Load each dataset into memory
      rgb_data = f['rgb'][:]                 # shape: (TOTAL_IMAGES, HEIGHT, WIDTH, 4), dtype: uint8
      depth_data = f['depth'][:]             # shape: (TOTAL_IMAGES, HEIGHT, WIDTH), dtype: float32
      segmentation_data = f['segmentation'][:]  # shape: (TOTAL_IMAGES, HEIGHT, WIDTH), dtype: int32
      
      camera_pose_data = f['camera_pose'][:]       # shape: (TOTAL_IMAGES, 4, 4), dtype: float32
      camera_intrinsics_data = f['camera_intrinsics'][:]  # shape: (TOTAL_IMAGES, 3, 3), dtype: float32

      # Load the segmentation labels (stored as a JSON string in the file attributes)
      segmentation_labels = json.loads(f.attrs['segmentation_labels'])

  # Find the image files in the images directory
  image_files = os.listdir(images_dir)
  image_files_by_idx = {int(f.split("_")[1]): f for f in image_files}

  # Convert the data to a point cloud format
  points_3d = []
  points_rgb = []
  image_paths = []
  extrinsics = []
  for i, (rgb, depth, seg, cam_pose, cam_intrinsics) in tqdm.tqdm(enumerate(zip(
      rgb_data, depth_data, segmentation_data, camera_pose_data, camera_intrinsics_data
  )), total=rgb_data.shape[0], desc="Processing images"):
    cam_pos, cam_orn = T.mat2pose(torch.as_tensor(cam_pose))
    quat_for_conversion = ilutils.convert_camera_frame_orientation_convention(
        cam_orn, "opengl", "ros"
    )[[3, 0, 1, 2]].reshape(4)
    points = ilutils.create_pointcloud_from_depth(
        torch.as_tensor(cam_intrinsics),
        torch.as_tensor(depth),
        True,
        position=cam_pos,
        orientation=quat_for_conversion,
    )
    points = points.reshape((rgb.shape[1], rgb.shape[0], 3)).permute(
        1, 0, 2
    )
    # cam_orn_opengl = ilutils.convert_camera_frame_orientation_convention(cam_orn, "opengl", "world")
    cam_orn_opengl = cam_orn
    extrinsics.append(T.pose2mat((cam_pos, cam_orn_opengl)))
    colors = rgb[:, :, :3]
    points_rgb.append(colors)
    points_3d.append(points.numpy())
    image_paths.append(image_files_by_idx[i])

  points_3d = np.stack(points_3d, axis=0)
  points_rgb = np.stack(points_rgb, axis=0)
  points_xyf = create_pixel_coordinate_grid(rgb_data.shape[0], rgb_data.shape[1], rgb_data.shape[2])
  extrinsics = np.stack(extrinsics, axis=0)

  # First find all the finite points in the point cloud
  finite_mask = np.isfinite(points_3d).all(axis=-1)
  points_3d = points_3d[finite_mask]
  points_rgb = points_rgb[finite_mask]
  points_xyf = points_xyf[finite_mask]

  # Limit the number of points to 100,000
  indices = np.random.choice(points_3d.shape[0], 100000, replace=False)
  points_3d = points_3d[indices]
  points_rgb = points_rgb[indices]
  points_xyf = points_xyf[indices]

  assert points_3d.shape == points_rgb.shape == points_xyf.shape == (100000, 3), \
      "Mismatch in number of points, colors, and pixel coordinates."

  print("Creating COLMAP reconstruction...")
  reconstruction = batch_np_matrix_to_pycolmap_wo_track(
      points_3d,
      points_xyf,
      points_rgb,
      extrinsics,
      camera_intrinsics_data,
      image_paths,
      rgb_data.shape[1],
      rgb_data.shape[2],
      shared_camera=False,
      camera_type="PINHOLE",
  )

  # reconstruction_resolution = vggt_fixed_resolution

  # reconstruction = rename_colmap_recons_and_rescale_camera(
  #     reconstruction,
  #     base_image_path_list,
  #     original_coords.cpu().numpy(),
  #     img_size=reconstruction_resolution,
  #     shift_point2d_to_original_res=True,
  #     shared_camera=shared_camera,
  # )

  print(f"Saving reconstruction to {scene_dir}/sparse")
  sparse_reconstruction_dir = os.path.join(scene_dir, "sparse")
  os.makedirs(sparse_reconstruction_dir, exist_ok=True)
  reconstruction.write(sparse_reconstruction_dir)

  # Save point cloud for fast visualization
  print(f"Exporting point cloud to {sparse_reconstruction_dir}/points.ply")
  trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(sparse_reconstruction_dir, "points.ply"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HDF5 data to COLMAP format.")
    parser.add_argument("scene_dir", type=str, help="Path to the scene directory containing the HDF5 file.")
    args = parser.parse_args()

    hdf5_to_colmap(args.scene_dir)