from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from omegaconf import MISSING
from omnigibson.utils.constants import PrimType
from omnigibson.configs.controller_config import (
    BaseControllerConfig,
    JointControllerConfig,
    DifferentialDriveControllerConfig,
    InverseKinematicsControllerConfig,
    MultiFingerGripperControllerConfig,
    SensorConfig,
)


@dataclass
class PrimConfig:
    """Base configuration for all prims"""

    name: str = MISSING
    relative_prim_path: Optional[str] = None
    prim_type: PrimType = PrimType.RIGID
    position: Optional[List[float]] = None
    orientation: Optional[List[float]] = None
    scale: Optional[List[float]] = None
    fixed_base: bool = False
    visible: bool = True
    visual_only: bool = False
    kinematic_only: Optional[bool] = None
    self_collisions: bool = False
    load_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectConfig(PrimConfig):
    """Configuration for objects"""

    category: str = "object"
    abilities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    include_default_states: bool = True


@dataclass
class USDObjectConfig(ObjectConfig):
    """Configuration for USD-based objects"""

    usd_path: str = MISSING
    encrypted: bool = False


@dataclass
class DatasetObjectConfig(ObjectConfig):
    """Configuration for dataset objects"""

    model: Optional[str] = None
    dataset_type: str = "BEHAVIOR"
    bounding_box: Optional[List[float]] = None
    in_rooms: Optional[List[str]] = None


@dataclass
class PrimitiveObjectConfig(ObjectConfig):
    """Configuration for primitive objects"""

    primitive_type: str = MISSING
    rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    radius: Optional[float] = None
    height: Optional[float] = None
    size: Optional[float] = None


@dataclass
class LightObjectConfig(ObjectConfig):
    """Configuration for light objects"""

    light_type: str = MISSING
    radius: float = 1.0
    intensity: float = 50000.0


@dataclass
class ControllableObjectConfig(ObjectConfig):
    """Configuration for controllable objects"""

    control_freq: Optional[float] = None
    controller_config: Optional[Union[BaseControllerConfig, JointControllerConfig]] = None
    action_type: str = "continuous"
    action_normalize: bool = True
    reset_joint_pos: Optional[List[float]] = None


@dataclass
class RobotConfig(ControllableObjectConfig):
    """Configuration for robots"""

    type: str = MISSING
    obs_modalities: List[str] = field(default_factory=lambda: ["rgb", "proprio"])
    proprio_obs: str = "default"
    sensor_config: Optional[SensorConfig] = field(default_factory=SensorConfig)
    grasping_mode: str = "physical"
    grasping_direction: str = "lower"
    disable_grasp_handling: bool = False
    default_reset_mode: str = "untuck"
    default_arm_pose: str = "vertical"
    controllers: Dict[
        str,
        Union[
            BaseControllerConfig,
            JointControllerConfig,
            DifferentialDriveControllerConfig,
            InverseKinematicsControllerConfig,
            MultiFingerGripperControllerConfig,
        ],
    ] = field(default_factory=dict)
