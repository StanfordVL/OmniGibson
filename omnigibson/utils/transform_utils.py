"""
Utility functions of matrix and vector transformations.

NOTE: convention for quaternions is (x, y, z, w)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch as th

PI = math.pi
EPS = torch.finfo(torch.float32).eps * 4.0

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}


@torch.compile
def copysign(a, b):
    # type: (float, torch.Tensor) -> torch.Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.compile
def anorm(x: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    """Compute L2 norms along specified axes."""
    return torch.norm(x, dim=dim, keepdim=keepdim)


@torch.compile
def normalize(v: torch.Tensor, dim: Optional[int] = None, eps: float = 1e-10) -> torch.Tensor:
    """L2 Normalize along specified axes."""
    norm = anorm(v, dim=dim, keepdim=True)
    return v / torch.where(norm < eps, torch.full_like(norm, eps), norm)


@torch.compile
def dot(v1, v2, dim=-1, keepdim=False):
    """
    Computes dot product between two vectors along the provided dim, optionally keeping the dimension

    Args:
        v1 (tensor): (..., N, ...) arbitrary vector
        v2 (tensor): (..., N, ...) arbitrary vector
        dim (int): Dimension to sum over for dot product
        keepdim (bool): Whether to keep dimension over which dot product is calculated

    Returns:
        tensor: (..., [1,] ...) dot product of vectors, with optional dimension kept if @keepdim is True
    """
    # type: (Tensor, Tensor, int, bool) -> Tensor
    return torch.sum(v1 * v2, dim=dim, keepdim=keepdim)


@torch.compile
def unit_vector(data: torch.Tensor, dim: Optional[int] = None, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Returns tensor normalized by length, i.e. Euclidean norm, along axis.

    Args:
        data (torch.Tensor): data to normalize
        dim (Optional[int]): If specified, determines specific dimension along data to normalize
        out (Optional[torch.Tensor]): If specified, will store computation in this variable

    Returns:
        torch.Tensor: Normalized vector
    """
    if out is None:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        else:
            data = data.clone().to(torch.float32)

        if data.ndim == 1:
            return data / torch.sqrt(torch.dot(data, data))
    else:
        if out is not data:
            out.copy_(data)
        data = out

    if dim is None:
        dim = -1

    length = torch.sum(data * data, dim=dim, keepdim=True).sqrt()
    data = data / (length + 1e-8)  # Add small epsilon to avoid division by zero

    return data


@torch.compile
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Apply a quaternion rotation to a vector (equivalent to R.from_quat(x).apply(y))
    Args:
        quat (torch.Tensor): (4,) or (N, 4) or (N, 1, 4) quaternion in (x, y, z, w) format
        vec (torch.Tensor): (3,) or (M, 3) or (1, M, 3) vector to rotate
    Returns:
        torch.Tensor: (M, 3) or (N, M, 3) rotated vector
    """
    assert quat.shape[-1] == 4, "Quaternion must have 4 components in last dimension"
    assert vec.shape[-1] == 3, "Vector must have 3 components in last dimension"

    # Ensure quat is at least 2D and vec is at least 2D
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)

    # Ensure quat is (N, 1, 4) and vec is (1, M, 3)
    if quat.dim() == 2:
        quat = quat.unsqueeze(1)
    if vec.dim() == 2:
        vec = vec.unsqueeze(0)

    # Extract quaternion components
    qx, qy, qz, qw = quat.unbind(-1)

    # Compute the quaternion multiplication
    t = torch.stack(
        [
            2 * (qy * vec[..., 2] - qz * vec[..., 1]),
            2 * (qz * vec[..., 0] - qx * vec[..., 2]),
            2 * (qx * vec[..., 1] - qy * vec[..., 0]),
        ],
        dim=-1,
    )

    # Compute the final rotated vector
    result = vec + qw.unsqueeze(-1) * t + torch.cross(quat[..., :3], t, dim=-1)

    # Remove any extra dimensions
    return result.squeeze()


@torch.compile
def convert_quat(q: torch.Tensor, to: str = "xyzw") -> torch.Tensor:
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q (torch.Tensor): a 4-dim array corresponding to a quaternion
        to (str): either 'xyzw' or 'wxyz', determining which convention to convert to.

    Returns:
        torch.Tensor: The converted quaternion
    """
    if to == "xyzw":
        return torch.stack([q[1], q[2], q[3], q[0]], dim=0)
    elif to == "wxyz":
        return torch.stack([q[3], q[0], q[1], q[2]], dim=0)
    else:
        raise ValueError("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


@torch.compile
def quat_multiply(quaternion1: torch.Tensor, quaternion0: torch.Tensor) -> torch.Tensor:
    """
    Return multiplication of two quaternions (q1 * q0).

    Args:
        quaternion1 (torch.Tensor): (x,y,z,w) quaternion
        quaternion0 (torch.Tensor): (x,y,z,w) quaternion

    Returns:
        torch.Tensor: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    x1, y1, z1, w1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]

    return torch.stack(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ],
        dim=0,
    )


@torch.compile
def quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Return conjugate of quaternion.

    Args:
        quaternion (torch.Tensor): (x,y,z,w) quaternion

    Returns:
        torch.Tensor: (x,y,z,w) quaternion conjugate
    """
    return torch.cat([-quaternion[:3], quaternion[3:]])


@torch.compile
def quat_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Return inverse of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> torch.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True

    Args:
        quaternion (torch.tensor): (x,y,z,w) quaternion

    Returns:
        torch.tensor: (x,y,z,w) quaternion inverse
    """
    return quat_conjugate(quaternion) / torch.dot(quaternion, quaternion)


@torch.compile
def quat_distance(quaternion1, quaternion0):
    """
    Returns distance between two quaternions, such that distance * quaternion0 = quaternion1

    Args:
        quaternion1 (torch.tensor): (x,y,z,w) quaternion
        quaternion0 (torch.tensor): (x,y,z,w) quaternion

    Returns:
        torch.tensor: (x,y,z,w) quaternion distance
    """
    return quat_multiply(quaternion1, quat_inverse(quaternion0))


@torch.compile
def quat_slerp(quat0, quat1, frac, shortestpath=True, eps=1.0e-15):
    """
    Return spherical linear interpolation between two quaternions.

    Args:
        quat0 (tensor): (..., 4) tensor where the final dim is (x,y,z,w) initial quaternion
        quat1 (tensor): (..., 4) tensor where the final dim is (x,y,z,w) final quaternion
        frac (tensor): Values in [0.0, 1.0] representing fraction of interpolation
        shortestpath (bool): If True, will calculate shortest path
        eps (float): Value to check for singularities
    Returns:
        tensor: (..., 4) Interpolated
    """
    # type: (Tensor, Tensor, Tensor, bool, float) -> Tensor
    # reshape quaternion
    quat_shape = quat0.shape
    quat0 = unit_vector(quat0.reshape(-1, 4), dim=-1)
    quat1 = unit_vector(quat1.reshape(-1, 4), dim=-1)

    # Check for endpoint cases
    where_start = frac <= 0.0
    where_end = frac >= 1.0

    d = dot(quat0, quat1, dim=-1, keepdim=True)
    if shortestpath:
        quat1 = torch.where(d < 0.0, -quat1, quat1)
        d = torch.abs(d)
    angle = torch.acos(torch.clip(d, -1.0, 1.0))

    # Check for small quantities (i.e.: q0 = q1)
    where_small_diff = torch.abs(torch.abs(d) - 1.0) < eps
    where_small_angle = abs(angle) < eps

    isin = 1.0 / torch.sin(angle)
    val = quat0 * torch.sin((1.0 - frac) * angle) * isin + quat1 * torch.sin(frac * angle) * isin

    # Filter edge cases
    val = torch.where(
        where_small_diff | where_small_angle | where_start,
        quat0,
        torch.where(
            where_end,
            quat1,
            val,
        ),
    )

    # Reshape and return values
    return val.reshape(list(quat_shape))


@torch.compile
def random_axis_angle(angle_limit: float = 2.0 * math.pi):
    """
    Samples an axis-angle rotation by first sampling a random axis
    and then sampling an angle. If @angle_limit is provided, the size
    of the rotation angle is constrained.

    Args:
        angle_limit (float): Determines magnitude limit of angles to generate

    Raises:
        AssertionError: [Invalid RNG]
    """
    # sample random axis using a normalized sample from spherical Gaussian.
    # see (http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/)
    # for why it works.
    random_axis = torch.randn(3)
    random_axis /= torch.norm(random_axis)
    random_angle = torch.rand(1) * angle_limit
    return random_axis, random_angle.item()


@torch.compile
def quat2mat(quaternion):
    """
    Convert quaternions into rotation matrices.

    Args:
        quaternion (torch.Tensor): A tensor of shape (..., 4) representing batches of quaternions (w, x, y, z).

    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing batches of rotation matrices.
    """
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    outer = quaternion.unsqueeze(-1) * quaternion.unsqueeze(-2)

    # Extract the necessary components
    xx = outer[..., 0, 0]
    yy = outer[..., 1, 1]
    zz = outer[..., 2, 2]
    xy = outer[..., 0, 1]
    xz = outer[..., 0, 2]
    yz = outer[..., 1, 2]
    xw = outer[..., 0, 3]
    yw = outer[..., 1, 3]
    zw = outer[..., 2, 3]

    rotation_matrix = torch.empty(quaternion.shape[:-1] + (3, 3), dtype=quaternion.dtype, device=quaternion.device)

    rotation_matrix[..., 0, 0] = 1 - 2 * (yy + zz)
    rotation_matrix[..., 0, 1] = 2 * (xy - zw)
    rotation_matrix[..., 0, 2] = 2 * (xz + yw)

    rotation_matrix[..., 1, 0] = 2 * (xy + zw)
    rotation_matrix[..., 1, 1] = 1 - 2 * (xx + zz)
    rotation_matrix[..., 1, 2] = 2 * (yz - xw)

    rotation_matrix[..., 2, 0] = 2 * (xz - yw)
    rotation_matrix[..., 2, 1] = 2 * (yz + xw)
    rotation_matrix[..., 2, 2] = 1 - 2 * (xx + yy)

    return rotation_matrix


@torch.compile
def mat2quat(rmat: torch.Tensor) -> torch.Tensor:
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat (torch.Tensor): (3, 3) or (..., 3, 3) rotation matrix
    Returns:
        torch.Tensor: (4,) or (..., 4) (x,y,z,w) float quaternion angles
    """
    # Check if input is a single matrix or a batch
    is_single = rmat.dim() == 2
    if is_single:
        rmat = rmat.unsqueeze(0)

    batch_shape = rmat.shape[:-2]
    mat_flat = rmat.reshape(-1, 3, 3)

    m00, m01, m02 = mat_flat[:, 0, 0], mat_flat[:, 0, 1], mat_flat[:, 0, 2]
    m10, m11, m12 = mat_flat[:, 1, 0], mat_flat[:, 1, 1], mat_flat[:, 1, 2]
    m20, m21, m22 = mat_flat[:, 2, 0], mat_flat[:, 2, 1], mat_flat[:, 2, 2]

    trace = m00 + m11 + m22

    trace_positive = trace > 0
    cond1 = (m00 > m11) & (m00 > m22) & ~trace_positive
    cond2 = (m11 > m22) & ~(trace_positive | cond1)
    cond3 = ~(trace_positive | cond1 | cond2)

    # Trace positive condition
    sq = torch.where(trace_positive, torch.sqrt(trace + 1.0) * 2.0, torch.zeros_like(trace))
    qw = torch.where(trace_positive, 0.25 * sq, torch.zeros_like(trace))
    qx = torch.where(trace_positive, (m21 - m12) / sq, torch.zeros_like(trace))
    qy = torch.where(trace_positive, (m02 - m20) / sq, torch.zeros_like(trace))
    qz = torch.where(trace_positive, (m10 - m01) / sq, torch.zeros_like(trace))

    # Condition 1
    sq = torch.where(cond1, torch.sqrt(1.0 + m00 - m11 - m22) * 2.0, sq)
    qw = torch.where(cond1, (m21 - m12) / sq, qw)
    qx = torch.where(cond1, 0.25 * sq, qx)
    qy = torch.where(cond1, (m01 + m10) / sq, qy)
    qz = torch.where(cond1, (m02 + m20) / sq, qz)

    # Condition 2
    sq = torch.where(cond2, torch.sqrt(1.0 + m11 - m00 - m22) * 2.0, sq)
    qw = torch.where(cond2, (m02 - m20) / sq, qw)
    qx = torch.where(cond2, (m01 + m10) / sq, qx)
    qy = torch.where(cond2, 0.25 * sq, qy)
    qz = torch.where(cond2, (m12 + m21) / sq, qz)

    # Condition 3
    sq = torch.where(cond3, torch.sqrt(1.0 + m22 - m00 - m11) * 2.0, sq)
    qw = torch.where(cond3, (m10 - m01) / sq, qw)
    qx = torch.where(cond3, (m02 + m20) / sq, qx)
    qy = torch.where(cond3, (m12 + m21) / sq, qy)
    qz = torch.where(cond3, 0.25 * sq, qz)

    quat = torch.stack([qx, qy, qz, qw], dim=-1)

    # Normalize the quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    # Reshape to match input batch shape
    quat = quat.reshape(batch_shape + (4,))

    # If input was a single matrix, remove the batch dimension
    if is_single:
        quat = quat.squeeze(0)

    return quat


@torch.compile
def mat2pose(hmat):
    """
    Converts a homogeneous 4x4 matrix into pose.

    Args:
        hmat (torch.tensor): a 4x4 homogeneous matrix

    Returns:
        2-tuple:
            - (torch.tensor) (x,y,z) position array in cartesian coordinates
            - (torch.tensor) (x,y,z,w) orientation array in quaternion form
    """
    assert torch.allclose(hmat[:3, :3].det(), torch.tensor(1.0)), "Rotation matrix must not be scaled"
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn


@torch.compile
def vec2quat(vec: torch.Tensor, up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])) -> torch.Tensor:
    """
    Converts given 3d-direction vector @vec to quaternion orientation with respect to another direction vector @up

    Args:
        vec (torch.Tensor): (x,y,z) direction vector (possibly non-normalized)
        up (torch.Tensor): (x,y,z) direction vector representing the canonical up direction (possibly non-normalized)

    Returns:
        torch.Tensor: (x,y,z,w) quaternion
    """
    # Ensure inputs are 2D
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    if up.dim() == 1:
        up = up.unsqueeze(0)

    vec_n = torch.nn.functional.normalize(vec, dim=-1)
    up_n = torch.nn.functional.normalize(up, dim=-1)

    s_n = torch.cross(up_n, vec_n, dim=-1)
    u_n = torch.cross(vec_n, s_n, dim=-1)

    rotation_matrix = torch.stack([vec_n, s_n, u_n], dim=-1)

    return mat2quat(rotation_matrix)


@torch.compile
def euler2quat(euler: torch.Tensor) -> torch.Tensor:
    """
    Converts euler angles into quaternion form

    Args:
        euler (torch.Tensor): (..., 3) (r,p,y) angles

    Returns:
        torch.Tensor: (..., 4) (x,y,z,w) float quaternion angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    assert euler.shape[-1] == 3, "Invalid input shape"

    # Unpack roll, pitch, yaw
    roll, pitch, yaw = euler.unbind(-1)

    # Compute sines and cosines of half angles
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    # Compute quaternion components
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    # Stack and return
    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.compile
def quat2euler(q):

    single_dim = q.dim() == 1

    if single_dim:
        q = q.unsqueeze(0)

    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(math.pi / 2.0, sinp), torch.asin(sinp))
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    euler = torch.stack([roll, pitch, yaw], dim=-1) % (2 * math.pi)
    euler[euler > math.pi] -= 2 * math.pi

    if single_dim:
        euler = euler.squeeze(0)

    return euler


@torch.compile
def euler2mat(euler):
    """
    Converts euler angles into rotation matrix form

    Args:
        euler (torch.tensor): (r,p,y) angles

    Returns:
        torch.tensor: 3x3 rotation matrix

    Raises:
        AssertionError: [Invalid input shape]
    """
    euler = torch.as_tensor(euler, dtype=torch.float32)
    assert euler.shape[-1] == 3, f"Invalid shaped euler {euler}"

    # Convert Euler angles to quaternion
    quat = euler2quat(euler)

    # Convert quaternion to rotation matrix
    return quat2mat(quat)


@torch.compile
def mat2euler(rmat):
    """
    Converts given rotation matrix to euler angles in radian.

    Args:
        rmat (torch.tensor): 3x3 rotation matrix

    Returns:
        torch.tensor: (r,p,y) converted euler angles in radian vec3 float
    """
    M = torch.as_tensor(rmat, dtype=torch.float32)[:3, :3]

    # Convert rotation matrix to quaternion
    # Note: You'll need to implement mat2quat function
    quat = mat2quat(M)

    # Convert quaternion to Euler angles
    euler = quat2euler(quat)
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    return torch.stack([roll, pitch, yaw], dim=-1)


@torch.compile
def pose2mat(pose: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    pos, orn = pose

    # Ensure pos and orn are the expected shape and dtype
    pos = pos.to(dtype=torch.float32).reshape(3)
    orn = orn.to(dtype=torch.float32).reshape(4)

    homo_pose_mat = torch.eye(4, dtype=torch.float32)
    homo_pose_mat[:3, :3] = quat2mat(orn)
    homo_pose_mat[:3, 3] = pos

    return homo_pose_mat


@torch.compile
def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.
    Args:
        quat (tensor): (..., 4) tensor where the final dim is (x,y,z,w) quaternion
    Returns:
        tensor: (..., 3) axis-angle exponential coordinates
    """
    # reshape quaternion
    quat_shape = quat.shape[:-1]  # ignore last dim
    quat = quat.reshape(-1, 4)
    # clip quaternion
    quat[:, 3] = torch.clip(quat[:, 3], -1.0, 1.0)
    # Calculate denominator
    den = torch.sqrt(1.0 - quat[:, 3] * quat[:, 3])
    # Map this into a mask

    # Create return array
    ret = torch.zeros_like(quat)[:, :3]
    idx = torch.nonzero(den).reshape(-1)
    ret[idx, :] = (quat[idx, :3] * 2.0 * torch.acos(quat[idx, 3]).unsqueeze(-1)) / den[idx].unsqueeze(-1)

    # Reshape and return output
    ret = ret.reshape(
        list(quat_shape)
        + [
            3,
        ]
    )
    return ret


@torch.compile
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape, dtype=torch.int)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps  # torch.nonzero(angle).reshape(-1)
    quat[idx, :] = torch.cat(
        [vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :], torch.cos(angle[idx, :] / 2.0)], dim=-1
    )

    # Reshape and return output
    quat = quat.reshape(
        list(input_shape)
        + [
            4,
        ]
    )
    return quat


@torch.compile
def pose_in_A_to_pose_in_B(pose_A, pose_A_in_B):
    """
    Converts a homogenous matrix corresponding to a point C in frame A
    to a homogenous matrix corresponding to the same point C in frame B.

    Args:
        pose_A (torch.tensor): 4x4 matrix corresponding to the pose of C in frame A
        pose_A_in_B (torch.tensor): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        torch.tensor: 4x4 matrix corresponding to the pose of C in frame B
    """

    # pose of A in B takes a point in A and transforms it to a point in C.

    # pose of C in B = pose of A in B * pose of C in A
    # take a point in C, transform it to A, then to B
    # T_B^C = T_A^C * T_B^A
    return torch.matmul(pose_A_in_B, pose_A)


@torch.compile
def pose_inv(pose_mat):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose_mat (torch.tensor): 4x4 matrix for the pose to inverse

    Returns:
        torch.tensor: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = torch.zeros((4, 4))
    pose_inv[:3, :3] = pose_mat[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3] @ pose_mat[:3, 3]
    pose_inv[3, 3] = 1.0
    return pose_inv


@torch.compile
def pose_transform(pos1, quat1, pos0, quat0):
    """
    Conducts forward transform from pose (pos0, quat0) to pose (pos1, quat1):

    pose1 @ pose0, NOT pose0 @ pose1

    Args:
        pos1: (x,y,z) position to transform
        quat1: (x,y,z,w) orientation to transform
        pos0: (x,y,z) initial position
        quat0: (x,y,z,w) initial orientation

    Returns:
        2-tuple:
            - (torch.tensor) (x,y,z) position array in cartesian coordinates
            - (torch.tensor) (x,y,z,w) orientation array in quaternion form
    """
    # Get poses
    mat0 = pose2mat((pos0, quat0))
    mat1 = pose2mat((pos1, quat1))

    # Multiply and convert back to pos, quat
    return mat2pose(mat1 @ mat0)


@torch.compile
def invert_pose_transform(pos, quat):
    """
    Inverts a pose transform

    Args:
        pos: (x,y,z) position to transform
        quat: (x,y,z,w) orientation to transform

    Returns:
        2-tuple:
            - (torch.tensor) (x,y,z) position array in cartesian coordinates
            - (torch.tensor) (x,y,z,w) orientation array in quaternion form
    """
    # Get pose
    mat = pose2mat((pos, quat))

    # Invert pose and convert back to pos, quat
    return mat2pose(pose_inv(mat))


@torch.compile
def relative_pose_transform(pos1, quat1, pos0, quat0):
    """
    Computes relative forward transform from pose (pos0, quat0) to pose (pos1, quat1), i.e.: solves:

    pose1 = pose0 @ transform

    Args:
        pos1: (x,y,z) position to transform
        quat1: (x,y,z,w) orientation to transform
        pos0: (x,y,z) initial position
        quat0: (x,y,z,w) initial orientation

    Returns:
        2-tuple:
            - (torch.tensor) (x,y,z) position array in cartesian coordinates
            - (torch.tensor) (x,y,z,w) orientation array in quaternion form
    """
    # Get poses
    mat0 = pose2mat((pos0, quat0))
    mat1 = pose2mat((pos1, quat1))

    # Invert pose0 and calculate transform
    return mat2pose(pose_inv(mat0) @ mat1)


@torch.compile
def _skew_symmetric_translation(pos_A_in_B):
    """
    Helper function to get a skew symmetric translation matrix for converting quantities
    between frames.

    Args:
        pos_A_in_B (torch.tensor): (x,y,z) position of A in frame B

    Returns:
        torch.tensor: 3x3 skew symmetric translation matrix
    """
    return torch.tensor(
        [
            [0.0, -pos_A_in_B[2].item(), pos_A_in_B[1].item()],
            [pos_A_in_B[2].item(), 0.0, -pos_A_in_B[0].item()],
            [-pos_A_in_B[1].item(), pos_A_in_B[0].item(), 0.0],
        ],
        dtype=torch.float32,
        device=pos_A_in_B.device,
    )


@torch.compile
def vel_in_A_to_vel_in_B(vel_A, ang_vel_A, pose_A_in_B):
    """
    Converts linear and angular velocity of a point in frame A to the equivalent in frame B.

    Args:
        vel_A (torch.tensor): (vx,vy,vz) linear velocity in A
        ang_vel_A (torch.tensor): (wx,wy,wz) angular velocity in A
        pose_A_in_B (torch.tensor): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        2-tuple:

            - (torch.tensor) (vx,vy,vz) linear velocities in frame B
            - (torch.tensor) (wx,wy,wz) angular velocities in frame B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    vel_B = rot_A_in_B @ vel_A + skew_symm @ (rot_A_in_B @ ang_vel_A)
    ang_vel_B = rot_A_in_B @ ang_vel_A
    return vel_B, ang_vel_B


@torch.compile
def force_in_A_to_force_in_B(force_A, torque_A, pose_A_in_B):
    """
    Converts linear and rotational force at a point in frame A to the equivalent in frame B.

    Args:
        force_A (torch.tensor): (fx,fy,fz) linear force in A
        torque_A (torch.tensor): (tx,ty,tz) rotational force (moment) in A
        pose_A_in_B (torch.tensor): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        2-tuple:

            - (torch.tensor) (fx,fy,fz) linear forces in frame B
            - (torch.tensor) (tx,ty,tz) moments in frame B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    force_B = rot_A_in_B.T @ force_A
    torque_B = -(rot_A_in_B.T @ (skew_symm @ force_A)) + rot_A_in_B.T @ torque_A
    return force_B, torque_B


@torch.compile
def rotation_matrix(angle: float, direction: torch.Tensor) -> torch.Tensor:
    """
    Returns a 3x3 rotation matrix to rotate about the given axis.

    Args:
        angle (float): Magnitude of rotation in radians
        direction (torch.Tensor): (ax,ay,az) axis about which to rotate

    Returns:
        torch.Tensor: 3x3 rotation matrix
    """
    sina = torch.sin(torch.tensor(angle, dtype=torch.float32))
    cosa = torch.cos(torch.tensor(angle, dtype=torch.float32))

    direction = direction / torch.norm(direction)  # Normalize direction vector

    # Create rotation matrix
    R = torch.eye(3, dtype=torch.float32, device=direction.device)
    R *= cosa
    R += torch.outer(direction, direction) * (1.0 - cosa)
    direction *= sina

    # Create the skew-symmetric matrix
    skew_matrix = torch.zeros(3, 3, dtype=torch.float32, device=direction.device)
    skew_matrix[0, 1] = -direction[2]
    skew_matrix[0, 2] = direction[1]
    skew_matrix[1, 0] = direction[2]
    skew_matrix[1, 2] = -direction[0]
    skew_matrix[2, 0] = -direction[1]
    skew_matrix[2, 1] = direction[0]

    R += skew_matrix

    return R


@torch.compile
def transformation_matrix(angle: float, direction: torch.Tensor, point: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Returns a 4x4 homogeneous transformation matrix to rotate about axis defined by point and direction.

    Args:
        angle (float): Magnitude of rotation in radians
        direction (torch.Tensor): (ax,ay,az) axis about which to rotate
        point (Optional[torch.Tensor]): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        torch.Tensor: 4x4 homogeneous transformation matrix
    """
    R = rotation_matrix(angle, direction)

    M = torch.eye(4, dtype=torch.float32, device=direction.device)
    M[:3, :3] = R

    if point is not None:
        # Rotation not about origin
        point = point.to(dtype=torch.float32)
        M[:3, 3] = point - R @ point
    return M


@torch.compile
def clip_translation(dpos, limit):
    """
    Limits a translation (delta position) to a specified limit

    Scales down the norm of the dpos to 'limit' if norm(dpos) > limit, else returns immediately

    Args:
        dpos (n-array): n-dim Translation being clipped (e,g.: (x, y, z)) -- numpy array
        limit (float): Value to limit translation by -- magnitude (scalar, in same units as input)

    Returns:
        2-tuple:

            - (torch.tensor) Clipped translation (same dimension as inputs)
            - (bool) whether the value was clipped or not
    """
    input_norm = torch.norm(dpos)
    return (dpos * limit / input_norm, True) if input_norm > limit else (dpos, False)


@torch.compile
def clip_rotation(quat, limit):
    """
    Limits a (delta) rotation to a specified limit

    Converts rotation to axis-angle, clips, then re-converts back into quaternion

    Args:
        quat (torch.tensor): (x,y,z,w) rotation being clipped
        limit (float): Value to limit rotation by -- magnitude (scalar, in radians)

    Returns:
        2-tuple:

            - (torch.tensor) Clipped rotation quaternion (x, y, z, w)
            - (bool) whether the value was clipped or not
    """
    clipped = False

    # First, normalize the quaternion
    quat = quat / torch.norm(quat)

    den = math.sqrt(max(1 - quat[3] * quat[3], 0))
    if den == 0:
        # This is a zero degree rotation, immediately return
        return quat, clipped
    else:
        # This is all other cases
        x = quat[0] / den
        y = quat[1] / den
        z = quat[2] / den
        a = 2 * math.acos(quat[3])

    # Clip rotation if necessary and return clipped quat
    if abs(a) > limit:
        a = limit * torch.sign(a) / 2
        sa = math.sin(a)
        ca = math.cos(a)
        quat = torch.tensor([x * sa, y * sa, z * sa, ca])
        clipped = True

    return quat, clipped


@torch.compile
def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.

    Args:
        translation (torch.tensor): (x,y,z) translation value
        rotation (torch.tensor): a 3x3 matrix representing rotation

    Returns:
        pose (torch.tensor): a 4x4 homogeneous matrix
    """
    pose = torch.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


@torch.compile
def get_orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector, where inputs are quaternions

    Args:
        desired (tensor): (..., 4) where final dim is (x,y,z,w) quaternion
        current (tensor): (..., 4) where final dim is (x,y,z,w) quaternion
    Returns:
        tensor: (..., 3) where final dim is (ax, ay, az) axis-angle representing orientation error
    """
    # convert input shapes
    input_shape = desired.shape[:-1]
    desired = desired.reshape(-1, 4)
    current = current.reshape(-1, 4)

    cc = quat_conjugate(current)
    q_r = quat_multiply(desired, cc)
    return (q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)).reshape(list(input_shape) + [3])


@torch.compile
def get_orientation_diff_in_radian(orn0: torch.Tensor, orn1: torch.Tensor) -> torch.Tensor:
    """
    Returns the difference between two quaternion orientations in radians.

    Args:
        orn0 (torch.Tensor): (x, y, z, w) quaternion
        orn1 (torch.Tensor): (x, y, z, w) quaternion

    Returns:
        orn_diff (torch.Tensor): orientation difference in radians
    """
    # Compute the difference quaternion
    diff_quat = quat_distance(orn0, orn1)

    # Convert to axis-angle representation
    axis_angle = quat2axisangle(diff_quat)

    # The magnitude of the axis-angle vector is the rotation angle
    return torch.norm(axis_angle)


@torch.compile
def get_pose_error(target_pose, current_pose):
    """
    Computes the error corresponding to target pose - current pose as a 6-dim vector.
    The first 3 components correspond to translational error while the last 3 components
    correspond to the rotational error.

    Args:
        target_pose (torch.tensor): a 4x4 homogenous matrix for the target pose
        current_pose (torch.tensor): a 4x4 homogenous matrix for the current pose

    Returns:
        torch.tensor: 6-dim pose error.
    """
    error = torch.zeros(6)

    # compute translational error
    target_pos = target_pose[:3, 3]
    current_pos = current_pose[:3, 3]
    pos_err = target_pos - current_pos

    # compute rotational error
    r1 = current_pose[:3, 0]
    r2 = current_pose[:3, 1]
    r3 = current_pose[:3, 2]
    r1d = target_pose[:3, 0]
    r2d = target_pose[:3, 1]
    r3d = target_pose[:3, 2]
    rot_err = 0.5 * (torch.linalg.cross(r1, r1d) + torch.linalg.cross(r2, r2d) + torch.linalg.cross(r3, r3d))

    error[:3] = pos_err
    error[3:] = rot_err
    return error


@torch.compile
def matrix_inverse(matrix):
    """
    Helper function to have an efficient matrix inversion function.

    Args:
        matrix (torch.tensor): 2d-array representing a matrix

    Returns:
        torch.tensor: 2d-array representing the matrix inverse
    """
    return torch.linalg.inv_ex(matrix).inverse


@torch.compile
def vecs2axisangle(vec0, vec1):
    """
    Converts the angle from unnormalized 3D vectors @vec0 to @vec1 into an axis-angle representation of the angle

    Args:
        vec0 (torch.tensor): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        vec1 (torch.tensor): (..., 3) (x,y,z) 3D vector, possibly unnormalized
    """
    # Normalize vectors
    vec0 = normalize(vec0, dim=-1)
    vec1 = normalize(vec1, dim=-1)

    # Get cross product for direction of angle, and multiply by arcos of the dot product which is the angle
    return torch.linalg.cross(vec0, vec1) * torch.arccos((vec0 * vec1).sum(-1, keepdim=True))


@torch.compile
def vecs2quat(vec0: torch.Tensor, vec1: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    """
    Converts the angle from unnormalized 3D vectors @vec0 to @vec1 into a quaternion representation of the angle
    Args:
        vec0 (torch.Tensor): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        vec1 (torch.Tensor): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        normalized (bool): If True, @vec0 and @vec1 are assumed to already be normalized and we will skip the
            normalization step (more efficient)
    Returns:
        torch.Tensor: (..., 4) Normalized quaternion representing the rotation from vec0 to vec1
    """
    # Normalize vectors if requested
    if not normalized:
        vec0 = normalize(vec0, dim=-1)
        vec1 = normalize(vec1, dim=-1)

    # Half-way Quaternion Solution -- see https://stackoverflow.com/a/11741520
    cos_theta = torch.sum(vec0 * vec1, dim=-1, keepdim=True)

    # Create a tensor for the case where cos_theta == -1
    batch_shape = vec0.shape[:-1]
    fallback = torch.zeros(batch_shape + (4,), device=vec0.device, dtype=vec0.dtype)
    fallback[..., 0] = 1.0

    # Compute the quaternion
    quat_unnormalized = torch.where(
        cos_theta == -1,
        fallback,
        torch.cat([torch.linalg.cross(vec0, vec1), 1 + cos_theta], dim=-1),
    )

    return quat_unnormalized / torch.norm(quat_unnormalized, dim=-1, keepdim=True)


@torch.compile
def align_vector_sets(vec_set1: torch.Tensor, vec_set2: torch.Tensor) -> torch.Tensor:
    """
    Computes a single quaternion representing the rotation that best aligns vec_set1 to vec_set2.

    Args:
        vec_set1 (torch.Tensor): (N, 3) tensor of N 3D vectors
        vec_set2 (torch.Tensor): (N, 3) tensor of N 3D vectors

    Returns:
        torch.Tensor: (4,) Normalized quaternion representing the overall rotation
    """
    # Compute the cross-covariance matrix
    H = vec_set2.T @ vec_set1

    # Compute the elements for the quaternion
    trace = H.trace()
    w = trace + 1
    x = H[1, 2] - H[2, 1]
    y = H[2, 0] - H[0, 2]
    z = H[0, 1] - H[1, 0]

    # Construct the quaternion
    quat = torch.stack([x, y, z, w])

    # Handle the case where w is close to zero
    if quat[3] < 1e-4:
        quat[3] = 0
        max_idx = torch.argmax(quat[:3].abs()) + 1
        quat[max_idx] = 1

    # Normalize the quaternion
    quat = quat / (torch.norm(quat) + 1e-8)  # Add epsilon to avoid division by zero

    return quat


@torch.compile
def l2_distance(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Returns the L2 distance between vector v1 and v2."""
    return torch.norm(v1 - v2)


@torch.compile
def cartesian_to_polar(x, y):
    """Convert cartesian coordinate to polar coordinate"""
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.arctan2(y, x)
    return rho, phi


def deg2rad(deg):
    return deg * math.pi / 180.0


def rad2deg(rad):
    return rad * 180.0 / math.pi


@torch.compile
def check_quat_right_angle(quat: torch.Tensor, atol: float = 5e-2) -> torch.Tensor:
    """
    Check by making sure the quaternion is some permutation of +/- (1, 0, 0, 0),
    +/- (0.707, 0.707, 0, 0), or +/- (0.5, 0.5, 0.5, 0.5)
    Because orientations are all normalized (same L2-norm), every orientation should have a unique L1-norm
    So we check the L1-norm of the absolute value of the orientation as a proxy for verifying these values

    Args:
        quat (torch.Tensor): (x,y,z,w) quaternion orientation to check
        atol (float): Absolute tolerance permitted

    Returns:
        torch.Tensor: Boolean tensor indicating whether the quaternion is a right angle or not
    """
    l1_norm = torch.abs(quat).sum(dim=-1)
    reference_norms = torch.tensor([1.0, 1.414, 2.0], device=quat.device, dtype=quat.dtype)
    return torch.any(torch.abs(l1_norm.unsqueeze(-1) - reference_norms) < atol, dim=-1)


@torch.compile
def z_angle_from_quat(quat):
    """Get the angle around the Z axis produced by the quaternion."""
    rotated_X_axis = quat_apply(quat, torch.tensor([1, 0, 0], dtype=torch.float32))
    return torch.arctan2(rotated_X_axis[1], rotated_X_axis[0])


@torch.compile
def integer_spiral_coordinates(n: int) -> Tuple[int, int]:
    """A function to map integers to 2D coordinates in a spiral pattern around the origin."""
    # Map integers from Z to Z^2 in a spiral pattern around the origin.
    # Sources:
    # https://www.reddit.com/r/askmath/comments/18vqorf/find_the_nth_coordinate_of_a_square_spiral/
    # https://oeis.org/A174344
    m = math.floor(math.sqrt(n))
    x = ((-1) ** m) * ((n - m * (m + 1)) * (math.floor(2 * math.sqrt(n)) % 2) - math.ceil(m / 2))
    y = ((-1) ** (m + 1)) * ((n - m * (m + 1)) * (math.floor(2 * math.sqrt(n) + 1) % 2) + math.ceil(m / 2))
    return int(x), int(y)


@torch.compile
def random_quaternion(num_quaternions: int = 1) -> torch.Tensor:
    """
    Generate random rotation quaternions, uniformly distributed over SO(3).

    Arguments:
        num_quaternions: int, number of quaternions to generate (default: 1)

    Returns:
        torch.Tensor: A tensor of shape (num_quaternions, 4) containing random unit quaternions.
    """
    # Generate four random numbers between 0 and 1
    rand = torch.rand(num_quaternions, 4)

    # Use the formula from Ken Shoemake's "Uniform Random Rotations"
    r1 = torch.sqrt(1.0 - rand[:, 0])
    r2 = torch.sqrt(rand[:, 0])
    t1 = 2 * torch.pi * rand[:, 1]
    t2 = 2 * torch.pi * rand[:, 2]

    quaternions = torch.stack([r1 * torch.sin(t1), r1 * torch.cos(t1), r2 * torch.sin(t2), r2 * torch.cos(t2)], dim=1)

    return quaternions


@torch.compile
def transform_points(points: torch.Tensor, matrix: torch.Tensor, translate: bool = True) -> torch.Tensor:
    """
    Returns points rotated by a homogeneous
    transformation matrix.
    If points are (n, 2) matrix must be (3, 3)
    If points are (n, 3) matrix must be (4, 4)

    Arguments:
        points : (n, dim) torch.Tensor
        Points where `dim` is 2 or 3.
        matrix : (3, 3) or (4, 4) torch.Tensor
        Homogeneous rotation matrix.
        translate : bool
        Apply translation from matrix or not.

    Returns:
        transformed : (n, dim) torch.Tensor
        Transformed points.
    """
    if len(points) == 0 or matrix is None:
        return points.clone()

    count, dim = points.shape
    # Check if the matrix is close to an identity matrix
    identity = torch.eye(dim + 1, device=points.device)
    if torch.abs(matrix - identity[: dim + 1, : dim + 1]).max() < 1e-8:
        return points.clone().contiguous()

    if translate:
        stack = torch.cat((points, torch.ones(count, 1, device=points.device)), dim=1)
        return torch.mm(matrix, stack.t()).t()[:, :dim]
    else:
        return torch.mm(matrix[:dim, :dim], points.t()).t()


@torch.compile
def quaternions_close(q1: torch.Tensor, q2: torch.Tensor, atol: float = 1e-3) -> bool:
    """
    Whether two quaternions represent the same rotation,
    allowing for the possibility that one is the negative of the other.

    Arguments:
        q1: torch.Tensor
            First quaternion
        q2: torch.Tensor
            Second quaternion
        atol: float
            Absolute tolerance for comparison

    Returns:
        bool
            Whether the quaternions are close
    """
    return torch.allclose(q1, q2, atol=atol) or torch.allclose(q1, -q2, atol=atol)
