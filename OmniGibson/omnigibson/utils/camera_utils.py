# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions to project between pointcloud and depth images, borrowed from Isaac Lab: https://github.com/isaac-sim/IsaacLab."""

# needed to import for allowing type-hinting: torch.device | str | None
from __future__ import annotations
from collections.abc import Sequence
import math
from typing import Literal, Union

import numpy as np
import torch
import omnigibson.utils.transform_utils as T

# import warp as wp

TensorData = Union[np.ndarray, torch.Tensor]  # , wp.array]

"""
Math utils
"""


def convert_to_torch(
    array: TensorData,
    dtype: torch.dtype = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Converts a given array into a torch tensor.

    The function tries to convert the array to a torch tensor. If the array is a numpy/warp arrays, or python
    list/tuples, it is converted to a torch tensor. If the array is already a torch tensor, it is returned
    directly.

    If ``device`` is None, then the function deduces the current device of the data. For numpy arrays,
    this defaults to "cpu", for torch tensors it is "cpu" or "cuda", and for warp arrays it is "cuda".

    Note:
        Since PyTorch does not support unsigned integer types, unsigned integer arrays are converted to
        signed integer arrays. This is done by casting the array to the corresponding signed integer type.

    Args:
        array: The input array. It can be a numpy array, warp array, python list/tuple, or torch tensor.
        dtype: Target data-type for the tensor.
        device: The target device for the tensor. Defaults to None.

    Returns:
        The converted array as torch tensor.
    """
    # Convert array to tensor
    # if the datatype is not currently supported by torch we need to improvise
    # supported types are: https://pytorch.org/docs/stable/tensors.html
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint32:
            array = array.astype(np.int32)
        # need to deal with object arrays (np.void) separately
        tensor = torch.from_numpy(array)
    # elif isinstance(array, wp.array):
    #     if array.dtype == wp.uint32:
    #         array = array.view(wp.int32)
    #     tensor = wp.to_torch(array)
    else:
        tensor = torch.Tensor(array)
    # Convert tensor to the right device
    if device is not None and str(tensor.device) != str(device):
        tensor = tensor.to(device)
    # Convert dtype of tensor if requested
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.type(dtype)

    return tensor


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# @torch.jit.script
def math_utils_transform_points(
    points: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Transform input points in a given frame to a target frame.

    This function transform points from a source frame to a target frame. The transformation is defined by the
    position :math:`t` and orientation :math:`R` of the target frame in the source frame.

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    If the input `points` is a batch of points, the inputs `pos` and `quat` must be either a batch of
    positions and quaternions or a single position and quaternion. If the inputs `pos` and `quat` are
    a single position and quaternion, the same transformation is applied to all points in the batch.

    If either the inputs :attr:`pos` and :attr:`quat` are None, the corresponding transformation is not applied.

    Args:
        points: Points to transform. Shape is (N, P, 3) or (P, 3).
        pos: Position of the target frame. Shape is (N, 3) or (3,).
            Defaults to None, in which case the position is assumed to be zero.
        quat: Quaternion orientation of the target frame in (w, x, y, z). Shape is (N, 4) or (4,).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        Transformed points in the target frame. Shape is (N, P, 3) or (P, 3).

    Raises:
        ValueError: If the inputs `points` is not of shape (N, P, 3) or (P, 3).
        ValueError: If the inputs `pos` is not of shape (N, 3) or (3,).
        ValueError: If the inputs `quat` is not of shape (N, 4) or (4,).
    """
    points_batch = points.clone()
    # check if inputs are batched
    is_batched = points_batch.dim() == 3
    # -- check inputs
    if points_batch.dim() == 2:
        points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 2 or dim = 3: got shape {points.shape}")
    if not (pos is None or pos.dim() == 1 or pos.dim() == 2):
        raise ValueError(f"Expected pos to have dim = 1 or dim = 2: got shape {pos.shape}")
    if not (quat is None or quat.dim() == 1 or quat.dim() == 2):
        raise ValueError(f"Expected quat to have dim = 1 or dim = 2: got shape {quat.shape}")
    # -- rotation
    if quat is not None:
        # convert to batched rotation matrix
        rot_mat = matrix_from_quat(quat)
        if rot_mat.dim() == 2:
            rot_mat = rot_mat[None]  # (3, 3) -> (1, 3, 3)
        # convert points to matching batch size (N, P, 3) -> (N, 3, P)
        # and apply rotation
        points_batch = torch.matmul(rot_mat, points_batch.transpose_(1, 2))
        # (N, 3, P) -> (N, P, 3)
        points_batch = points_batch.transpose_(1, 2)
    # -- translation
    if pos is not None:
        # convert to batched translation vector
        if pos.dim() == 1:
            pos = pos[None, None, :]  # (3,) -> (1, 1, 3)
        else:
            pos = pos[:, None, :]  # (N, 3) -> (N, 1, 3)
        # apply translation
        points_batch += pos
    # -- return points in same shape as input
    if not is_batched:
        points_batch = points_batch.squeeze(0)  # (1, P, 3) -> (P, 3)

    return points_batch


"""
Depth <-> Pointcloud conversions.
"""


def transform_points(
    points: TensorData,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray | torch.Tensor:
    r"""Transform input points in a given frame to a target frame.

    This function transform points from a source frame to a target frame. The transformation is defined by the
    position ``t`` and orientation ``R`` of the target frame in the source frame.

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    If either the inputs `position` and `orientation` are None, the corresponding transformation is not applied.

    Args:
        points: a tensor of shape (p, 3) or (n, p, 3) comprising of 3d points in source frame.
        position: The position of source frame in target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of source frame in target frame.
            Defaults to None.
        device: The device for torch where the computation
            should be executed. Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        A tensor of shape (N, 3) comprising of 3D points in target frame.
        If the input is a numpy array, the output is a numpy array. Otherwise, it is a torch tensor.
    """
    # check if numpy
    is_numpy = isinstance(points, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert to torch
    points = convert_to_torch(points, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = points.device
    # apply rotation
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # apply translation
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    # apply transformation
    points = math_utils_transform_points(points, position, orientation)

    # return everything according to input type
    if is_numpy:
        return points.detach().cpu().numpy()
    else:
        return points


def create_pointcloud_from_depth(
    intrinsic_matrix: TensorData,
    depth: TensorData,
    keep_invalid: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray | torch.Tensor:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix: A (3, 3) array providing camera's calibration matrix.
        depth: An array of shape (H, W) with values encoding the depth measurement.
        keep_invalid: Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed.
            Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        An array/tensor of shape (N, 3) comprising of 3D coordinates of points.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    # convert to numpy matrix
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    depth = convert_to_torch(depth, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = depth.device
    # convert inputs to torch tensors
    intrinsic_matrix = convert_to_torch(intrinsic_matrix, dtype=torch.float32, device=device)
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # compute pointcloud
    depth_cloud = unproject_depth(depth, intrinsic_matrix)
    # convert 3D points to world frame
    depth_cloud = transform_points(depth_cloud, position, orientation)

    # keep only valid entries if flag is set
    if not keep_invalid:
        pts_idx_to_keep = torch.all(
            torch.logical_and(~torch.isnan(depth_cloud), ~torch.isinf(depth_cloud)),
            dim=1,
        )
        depth_cloud = depth_cloud[pts_idx_to_keep, ...]

    # return everything according to input type
    if is_numpy:
        return depth_cloud.detach().cpu().numpy()
    else:
        return depth_cloud


def create_pointcloud_from_rgbd(
    intrinsic_matrix: TensorData,
    depth: TensorData,
    rgb: TensorData | tuple[float, float, float] = None,
    normalize_rgb: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    num_channels: int = 3,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
    """Creates pointcloud from input depth image and camera transformation matrix.

    This function provides the same functionality as :meth:`create_pointcloud_from_depth` but also allows
    to provide the RGB values for each point.

    The ``rgb`` attribute is used to resolve the corresponding point's color:

    - If a ``np.array``/``wp.array``/``torch.tensor`` of shape (H, W, 3), then the corresponding channels encode RGB values.
    - If a tuple, then the point cloud has a single color specified by the values (r, g, b).
    - If None, then default color is white, i.e. (0, 0, 0).

    If the input ``normalize_rgb`` is set to :obj:`True`, then the RGB values are normalized to be in the range [0, 1].

    Args:
        intrinsic_matrix: A (3, 3) array/tensor providing camera's calibration matrix.
        depth: An array/tensor of shape (H, W) with values encoding the depth measurement.
        rgb: Color for generated point cloud. Defaults to None.
        normalize_rgb: Whether to normalize input rgb. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation `(w, x, y, z)` of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed. Defaults to None, in which case
            it takes the device that matches the depth image.
        num_channels: Number of channels in RGB pointcloud. Defaults to 3.

    Returns:
        A tuple of (N, 3) arrays or tensors containing the 3D coordinates of points and their RGB color respectively.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.

    Raises:
        ValueError:  When rgb image is a numpy array but not of shape (H, W, 3) or (H, W, 4).
    """
    # check valid inputs
    if rgb is not None and not isinstance(rgb, tuple):
        if len(rgb.shape) == 3:
            if rgb.shape[2] not in [3, 4]:
                raise ValueError(f"Input rgb image of invalid shape: {rgb.shape} != (H, W, 3) or (H, W, 4).")
        else:
            raise ValueError(f"Input rgb image not three-dimensional. Received shape: {rgb.shape}.")
    if num_channels not in [3, 4]:
        raise ValueError(f"Invalid number of channels: {num_channels} != 3 or 4.")

    # check if input depth is numpy array
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    if is_numpy:
        depth = torch.from_numpy(depth).to(device=device)
    # retrieve XYZ pointcloud
    points_xyz = create_pointcloud_from_depth(intrinsic_matrix, depth, True, position, orientation, device=device)

    # get image height and width
    im_height, im_width = depth.shape[:2]
    # total number of points
    num_points = im_height * im_width
    # extract color value
    if rgb is not None:
        if isinstance(rgb, (np.ndarray, torch.Tensor)):  # , wp.array)):
            # copy numpy array to preserve
            rgb = convert_to_torch(rgb, device=device, dtype=torch.float32)
            rgb = rgb[:, :, :3]
            # convert the matrix to (W, H, 3) from (H, W, 3) since depth processing
            # is done in the order (u, v) where u: (0, W-1) and v: (0 - H-1)
            points_rgb = rgb.permute(1, 0, 2).reshape(-1, 3)
        elif isinstance(rgb, (tuple, list)):
            # same color for all points
            points_rgb = torch.Tensor((rgb,) * num_points, device=device, dtype=torch.uint8)
        else:
            # default color is white
            points_rgb = torch.Tensor(((0, 0, 0),) * num_points, device=device, dtype=torch.uint8)
    else:
        points_rgb = torch.Tensor(((0, 0, 0),) * num_points, device=device, dtype=torch.uint8)
    # normalize color values
    if normalize_rgb:
        points_rgb = points_rgb.float() / 255

    # remove invalid points
    # pts_idx_to_keep = torch.all(torch.logical_and(~torch.isnan(points_xyz), ~torch.isinf(points_xyz)), dim=1)
    # points_rgb = points_rgb[pts_idx_to_keep, ...]
    # points_xyz = points_xyz[pts_idx_to_keep, ...]

    # add additional channels if required
    if num_channels == 4:
        points_rgb = torch.nn.functional.pad(points_rgb, (0, 1), mode="constant", value=1.0)

    # return everything according to input type
    if is_numpy:
        return points_xyz.cpu().numpy(), points_rgb.cpu().numpy()
    else:
        return points_xyz, points_rgb


def save_images_to_file(images: torch.Tensor, file_path: str):
    """Save images to file.

    Args:
        images: A tensor of shape (N, H, W, C) containing the images.
        file_path: The path to save the images to.
    """
    from torchvision.utils import make_grid, save_image

    save_image(
        make_grid(
            torch.swapaxes(images.unsqueeze(1), 1, -1).squeeze(-1),
            nrow=round(images.shape[0] ** 0.5),
        ),
        file_path,
    )


@torch.jit.script
def orthogonalize_perspective_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Converts perspective depth image to orthogonal depth image.

    Perspective depth images contain distances measured from the camera's optical center.
    Meanwhile, orthogonal depth images provide the distance from the camera's image plane.
    This method uses the camera geometry to convert perspective depth to orthogonal depth image.

    The function assumes that the width and height are both greater than 1.

    Args:
        depth: The perspective depth images. Shape is (H, W) or or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        intrinsics: The camera's calibration matrix. If a single matrix is provided, the same
            calibration matrix is used across all the depth images in the batch.
            Shape is (3, 3) or (N, 3, 3).

    Returns:
        The orthogonal depth images. Shape matches the input shape of depth images.

    Raises:
        ValueError: When depth is not of shape (H, W) or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        ValueError: When intrinsics is not of shape (3, 3) or (N, 3, 3).
    """
    # Clone inputs to avoid in-place modifications
    perspective_depth_batch = depth.clone()
    intrinsics_batch = intrinsics.clone()

    # Check if inputs are batched
    is_batched = perspective_depth_batch.dim() == 4 or (
        perspective_depth_batch.dim() == 3 and perspective_depth_batch.shape[-1] != 1
    )

    # Track whether the last dimension was singleton
    add_last_dim = False
    if perspective_depth_batch.dim() == 4 and perspective_depth_batch.shape[-1] == 1:
        add_last_dim = True
        perspective_depth_batch = perspective_depth_batch.squeeze(dim=3)  # (N, H, W, 1) -> (N, H, W)
    if perspective_depth_batch.dim() == 3 and perspective_depth_batch.shape[-1] == 1:
        add_last_dim = True
        perspective_depth_batch = perspective_depth_batch.squeeze(dim=2)  # (H, W, 1) -> (H, W)

    if perspective_depth_batch.dim() == 2:
        perspective_depth_batch = perspective_depth_batch[None]  # (H, W) -> (1, H, W)

    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]  # (3, 3) -> (1, 3, 3)

    if is_batched and intrinsics_batch.shape[0] == 1:
        intrinsics_batch = intrinsics_batch.expand(perspective_depth_batch.shape[0], -1, -1)  # (1, 3, 3) -> (N, 3, 3)

    # Validate input shapes
    if perspective_depth_batch.dim() != 3:
        raise ValueError(f"Expected depth images to have 2, 3, or 4 dimensions; got {depth.shape}.")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3); got {intrinsics.shape}.")

    # Image dimensions
    im_height, im_width = perspective_depth_batch.shape[1:]

    # Get the intrinsics parameters
    fx = intrinsics_batch[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics_batch[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics_batch[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics_batch[:, 1, 2].view(-1, 1, 1)

    # Create meshgrid of pixel coordinates
    u_grid = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    v_grid = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    u_grid, v_grid = torch.meshgrid(u_grid, v_grid, indexing="xy")

    # Expand the grids for batch processing
    u_grid = u_grid.unsqueeze(0).expand(perspective_depth_batch.shape[0], -1, -1)
    v_grid = v_grid.unsqueeze(0).expand(perspective_depth_batch.shape[0], -1, -1)

    # Compute the squared terms for efficiency
    x_term = ((u_grid - cx) / fx) ** 2
    y_term = ((v_grid - cy) / fy) ** 2

    # Calculate the orthogonal (normal) depth
    orthogonal_depth = perspective_depth_batch / torch.sqrt(1 + x_term + y_term)

    # Restore the last dimension if it was present in the input
    if add_last_dim:
        orthogonal_depth = orthogonal_depth.unsqueeze(-1)

    # Return to original shape if input was not batched
    if not is_batched:
        orthogonal_depth = orthogonal_depth.squeeze(0)

    return orthogonal_depth


@torch.jit.script
def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor, is_ortho: bool = True) -> torch.Tensor:
    r"""Un-project depth image into a pointcloud.

    This function converts orthogonal or perspective depth images into points given the calibration matrix
    of the camera. It uses the following transformation based on camera geometry:

    .. math::
        p_{3D} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`p_{3D}` is the 3D point, :math:`d` is the depth value (measured from the image plane),
    :math:`u` and :math:`v` are the pixel coordinates and :math:`K` is the intrinsic matrix.

    The function assumes that the width and height are both greater than 1. This makes the function
    deal with many possible shapes of depth images and intrinsics matrices.

    .. note::
        If :attr:`is_ortho` is False, the input depth images are transformed to orthogonal depth images
        by using the :meth:`orthogonalize_perspective_depth` method.

    Args:
        depth: The depth measurement. Shape is (H, W) or or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        intrinsics: The camera's calibration matrix. If a single matrix is provided, the same
            calibration matrix is used across all the depth images in the batch.
            Shape is (3, 3) or (N, 3, 3).
        is_ortho: Whether the input depth image is orthogonal or perspective depth image. If True, the input
            depth image is considered as the *orthogonal* type, where the measurements are from the camera's
            image plane. If False, the depth image is considered as the *perspective* type, where the
            measurements are from the camera's optical center. Defaults to True.

    Returns:
        The 3D coordinates of points. Shape is (P, 3) or (N, P, 3).

    Raises:
        ValueError: When depth is not of shape (H, W) or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        ValueError: When intrinsics is not of shape (3, 3) or (N, 3, 3).
    """
    # clone inputs to avoid in-place modifications
    intrinsics_batch = intrinsics.clone()
    # convert depth image to orthogonal if needed
    if not is_ortho:
        depth_batch = orthogonalize_perspective_depth(depth, intrinsics)
    else:
        depth_batch = depth.clone()

    # check if inputs are batched
    is_batched = depth_batch.dim() == 4 or (depth_batch.dim() == 3 and depth_batch.shape[-1] != 1)
    # make sure inputs are batched
    if depth_batch.dim() == 3 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=2)  # (H, W, 1) -> (H, W)
    if depth_batch.dim() == 2:
        depth_batch = depth_batch[None]  # (H, W) -> (1, H, W)
    if depth_batch.dim() == 4 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=3)  # (N, H, W, 1) -> (N, H, W)
    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]  # (3, 3) -> (1, 3, 3)
    # check shape of inputs
    if depth_batch.dim() != 3:
        raise ValueError(f"Expected depth images to have dim = 2 or 3 or 4: got shape {depth.shape}")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3): got shape {intrinsics.shape}")

    # get image height and width
    im_height, im_width = depth_batch.shape[1:]
    # create image points in homogeneous coordinates (3, H x W)
    indices_u = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    indices_v = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    img_indices = torch.stack(torch.meshgrid([indices_u, indices_v], indexing="ij"), dim=0).reshape(2, -1)
    pixels = torch.nn.functional.pad(img_indices, (0, 0, 0, 1), mode="constant", value=1.0)
    pixels = pixels.unsqueeze(0)  # (3, H x W) -> (1, 3, H x W)

    # unproject points into 3D space
    points = torch.matmul(torch.inverse(intrinsics_batch), pixels)  # (N, 3, H x W)
    points = points / points[:, -1, :].unsqueeze(1)  # normalize by last coordinate
    # flatten depth image (N, H, W) -> (N, H x W)
    depth_batch = depth_batch.transpose_(1, 2).reshape(depth_batch.shape[0], -1).unsqueeze(2)
    depth_batch = depth_batch.expand(-1, -1, 3)
    # scale points by depth
    points_xyz = points.transpose_(1, 2) * depth_batch  # (N, H x W, 3)

    # return points in same shape as input
    if not is_batched:
        points_xyz = points_xyz.squeeze(0)

    return points_xyz

def _axis_angle_rotation(axis: Literal["X", "Y", "Z"], angle: torch.Tensor) -> torch.Tensor:
    """Return the rotation matrices for one of the rotations about an axis of which Euler angles describe,
    for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: Euler angles in radians of any shape.

    Returns:
        Rotation matrices. Shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L164-L191
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def matrix_from_euler(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians. Shape is (..., 3).
        convention: Convention string of three uppercase letters from {"X", "Y", and "Z"}.
            For example, "XYZ" means that the rotations should be applied first about x,
            then y, then z.

    Returns:
        Rotation matrices. Shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L194-L220
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def convert_camera_frame_orientation_convention(
    orientation: torch.Tensor,
    origin: Literal["opengl", "ros", "world"] = "opengl",
    target: Literal["opengl", "ros", "world"] = "ros",
) -> torch.Tensor:
    r"""Converts a quaternion representing a rotation from one convention to another.

    In USD, the camera follows the ``"opengl"`` convention. Thus, it is always in **Y up** convention.
    This means that the camera is looking down the -Z axis with the +Y axis pointing up , and +X axis pointing right.
    However, in ROS, the camera is looking down the +Z axis with the +Y axis pointing down, and +X axis pointing right.
    Thus, the camera needs to be rotated by :math:`180^{\circ}` around the X axis to follow the ROS convention.

    .. math::

        T_{ROS} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} T_{USD}

    On the other hand, the typical world coordinate system is with +X pointing forward, +Y pointing left,
    and +Z pointing up. The camera can also be set in this convention by rotating the camera by :math:`90^{\circ}`
    around the X axis and :math:`-90^{\circ}` around the Y axis.

    .. math::

        T_{WORLD} = \begin{bmatrix} 0 & 0 & -1 & 0 \\ -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} T_{USD}

    Thus, based on their application, cameras follow different conventions for their orientation. This function
    converts a quaternion from one convention to another.

    Possible conventions are:

    - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
    - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
    - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

    Args:
        orientation: Quaternion of form `(x, y, z, w)` with shape (..., 4) in source convention.
        origin: Convention to convert from. Defaults to "opengl".
        target: Convention to convert to. Defaults to "ros".

    Returns:
        Quaternion of form `(x, y, z, w)` with shape (..., 4) in target convention
    """
    if target == origin:
        return orientation.clone()

    # -- unify input type
    if origin == "ros":
        # convert from ros to opengl convention
        rotm = T.quat2mat(orientation)
        rotm[:, 2] = -rotm[:, 2]
        rotm[:, 1] = -rotm[:, 1]
        # convert to opengl convention
        quat_gl = T.mat2quat(rotm)
    elif origin == "world":
        # convert from world (x forward and z up) to opengl convention
        rotm = T.quat2mat(orientation)
        rotm = torch.matmul(
            rotm,
            matrix_from_euler(torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ"),
        )
        # convert to isaac-sim convention
        quat_gl = T.mat2quat(rotm)
    else:
        quat_gl = orientation

    # -- convert to target convention
    if target == "ros":
        # convert from opengl to ros convention
        rotm = T.quat2mat(quat_gl)
        rotm[:, 2] = -rotm[:, 2]
        rotm[:, 1] = -rotm[:, 1]
        return T.mat2quat(rotm)
    elif target == "world":
        # convert from opengl to world (x forward and z up) convention
        rotm = T.quat2mat(quat_gl)
        rotm = torch.matmul(
            rotm,
            matrix_from_euler(torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ").T,
        )
        return T.mat2quat(rotm)
    else:
        return quat_gl.clone()
