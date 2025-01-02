from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from omegaconf import MISSING
import torch as th


@dataclass
class BaseControllerConfig:
    """Base configuration for all controllers"""

    control_freq: float = MISSING
    control_limits: Dict[str, Any] = MISSING  # Dict with position, velocity, effort limits and has_limit
    dof_idx: List[int] = MISSING
    command_input_limits: Optional[Union[str, Tuple[float, float], Tuple[List[float], List[float]]]] = "default"
    command_output_limits: Optional[Union[str, Tuple[float, float], Tuple[List[float], List[float]]]] = "default"


@dataclass
class JointControllerConfig(BaseControllerConfig):
    """Configuration for joint controllers"""

    motor_type: str = MISSING  # One of {position, velocity, effort}
    pos_kp: Optional[float] = None
    pos_damping_ratio: Optional[float] = None
    vel_kp: Optional[float] = None
    use_impedances: bool = False
    use_gravity_compensation: bool = False
    use_cc_compensation: bool = True
    use_delta_commands: bool = False
    compute_delta_in_quat_space: Optional[List[Tuple[int, int, int]]] = None


@dataclass
class VisionSensorConfig:
    """Configuration for vision sensors"""

    enabled: bool = True
    noise_type: Optional[str] = None
    noise_kwargs: Optional[Dict[str, Any]] = None
    modalities: Optional[List[str]] = None
    sensor_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "image_height": 128,
            "image_width": 128,
        }
    )


@dataclass
class ScanSensorConfig:
    """Configuration for scan sensors"""

    enabled: bool = True
    noise_type: Optional[str] = None
    noise_kwargs: Optional[Dict[str, Any]] = None
    modalities: Optional[List[str]] = None
    sensor_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "min_range": 0.05,
            "max_range": 10.0,
            "horizontal_fov": 360.0,
            "vertical_fov": 1.0,
            "yaw_offset": 0.0,
            "horizontal_resolution": 1.0,
            "vertical_resolution": 1.0,
            "rotation_rate": 0.0,
            "draw_points": False,
            "draw_lines": False,
            "occupancy_grid_resolution": 128,
            "occupancy_grid_range": 5.0,
            "occupancy_grid_inner_radius": 0.5,
            "occupancy_grid_local_link": None,
        }
    )


@dataclass
class DifferentialDriveControllerConfig(BaseControllerConfig):
    """Configuration for differential drive controllers"""

    wheel_radius: float = MISSING
    wheel_axle_length: float = MISSING
    wheel_control_idx: List[int] = MISSING  # [left_wheel_idx, right_wheel_idx]


@dataclass
class InverseKinematicsControllerConfig(BaseControllerConfig):
    """Configuration for inverse kinematics controllers"""

    control_freq: float = MISSING
    control_limits: Dict[str, Any] = MISSING
    dof_idx: List[int] = MISSING
    command_input_limits: Optional[Union[str, Tuple[float, float], Tuple[List[float], List[float]]]] = "default"
    command_output_limits: Optional[Union[str, Tuple[float, float], Tuple[List[float], List[float]]]] = "default"
    rest_poses: Optional[th.Tensor] = None
    position_limits: Optional[Tuple[th.Tensor, th.Tensor]] = None
    rotation_limits: Optional[Tuple[th.Tensor, th.Tensor]] = None
    position_gain: float = 1.0
    rotation_gain: float = 0.5
    position_threshold: float = 0.005
    rotation_threshold: float = 0.05
    num_ik_seeds: int = 10
    num_ik_solutions: int = 1
    regularization_weight: float = 0.01
    collision_checking: bool = True


@dataclass
class MultiFingerGripperControllerConfig(BaseControllerConfig):
    """Configuration for multi-finger gripper controllers"""

    mode: str = "binary"  # One of {binary, continuous}
    grasp_thresh: float = 0.5
    release_thresh: float = -0.5


@dataclass
class SensorConfig:
    """Configuration for all sensors"""

    VisionSensor: VisionSensorConfig = field(default_factory=VisionSensorConfig)
    ScanSensor: ScanSensorConfig = field(default_factory=ScanSensorConfig)
