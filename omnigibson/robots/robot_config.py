from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Literal
import yaml
import torch as th


@dataclass
class ArmConfig:
    """Configuration for a robot arm"""

    name: str
    joints: List[str]
    links: List[str]
    eef_link: str
    finger_links: List[str]
    finger_joints: List[str]
    workspace_range: List[float]  # [min_angle, max_angle] in degrees
    teleop_rotation_offset: Optional[List[float]] = None  # [rx, ry, rz] in radians
    default_poses: Optional[Dict[str, List[float]]] = None  # name -> joint positions


@dataclass
class ManipulationConfig:
    """Configuration for manipulation capability"""

    enabled: bool = False
    grasping_mode: Literal["physical", "assisted", "sticky"] = "physical"
    grasping_direction: Literal["lower", "upper"] = "lower"
    disable_grasp_handling: bool = False
    arms: List[ArmConfig] = None


@dataclass
class LocomotionConfig:
    """Configuration for locomotion capability"""

    enabled: bool = False
    type: Literal["two_wheel", "holonomic"] = "two_wheel"
    base_footprint_link: str = None
    base_joints: List[str] = None
    floor_touching_links: List[str] = None
    non_floor_touching_links: List[str] = None
    # Two-wheel specific config
    wheel_radius: Optional[float] = None
    wheel_axle_length: Optional[float] = None
    # Holonomic specific config
    max_linear_velocity: Optional[float] = None
    max_angular_velocity: Optional[float] = None


@dataclass
class TrunkConfig:
    """Configuration for articulated trunk capability"""

    enabled: bool = False
    joints: List[str] = None
    links: List[str] = None


@dataclass
class CameraConfig:
    """Configuration for active camera capability"""

    enabled: bool = False
    joints: List[str] = None


@dataclass
class ControllerConfig:
    """Configuration for a controller"""

    name: str
    control_freq: float
    motor_type: Optional[str] = None
    control_limits: Optional[Dict[str, List[float]]] = None
    command_output_limits: Optional[Union[List[float], None]] = None
    mode: Optional[str] = None
    smoothing_filter_size: Optional[int] = None
    limit_tolerance: Optional[float] = None
    inverted: Optional[bool] = None
    use_delta_commands: Optional[bool] = None


@dataclass
class ResetConfig:
    """Configuration for robot reset behavior"""

    mode: Literal["tuck", "untuck"] = "untuck"
    arm_pose: Optional[str] = "vertical"
    joint_positions: Optional[Dict[str, float]] = None


@dataclass
class ProprioObsConfig:
    """Configuration for proprioception observations"""

    enabled_keys: List[str]


@dataclass
class CollisionConfig:
    """Configuration for collision handling"""

    disabled_links: List[str]
    disabled_pairs: List[List[str]]


@dataclass
class RobotConfig:
    """Main configuration class for robots"""

    name: str
    model_name: str
    usd_path: str
    urdf_path: str
    scale: float = 1.0

    # Feature configurations
    manipulation: ManipulationConfig = ManipulationConfig()
    locomotion: LocomotionConfig = LocomotionConfig()
    trunk: TrunkConfig = TrunkConfig()
    camera: CameraConfig = CameraConfig()

    # Controller configurations
    controllers: Dict[str, ControllerConfig] = None

    # Other configurations
    reset: ResetConfig = ResetConfig()
    proprio_obs: ProprioObsConfig = None
    collisions: CollisionConfig = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RobotConfig":
        """Load robot configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert arm configurations
        if "manipulation" in config_dict and config_dict["manipulation"]["enabled"]:
            arms = []
            for arm_dict in config_dict["manipulation"]["arms"]:
                # Convert workspace range from degrees to radians
                if "workspace_range" in arm_dict:
                    arm_dict["workspace_range"] = [th.deg2rad(th.tensor(x)).item() for x in arm_dict["workspace_range"]]
                arms.append(ArmConfig(**arm_dict))
            config_dict["manipulation"]["arms"] = arms

        # Convert controller configurations
        if "controllers" in config_dict:
            controllers = {}
            for name, ctrl_dict in config_dict["controllers"].items():
                controllers[name] = ControllerConfig(**ctrl_dict)
            config_dict["controllers"] = controllers

        # Create the config object
        return cls(**config_dict)

    def validate(self):
        """Validate the configuration"""
        # Validate manipulation config
        if self.manipulation.enabled:
            assert self.manipulation.arms is not None, "Manipulation enabled but no arms configured"
            for arm in self.manipulation.arms:
                assert all(
                    k in self.controllers for k in [f"arm_{arm.name}", f"gripper_{arm.name}"]
                ), f"Missing controller configuration for arm {arm.name}"

        # Validate locomotion config
        if self.locomotion.enabled:
            assert "base" in self.controllers, "Missing base controller configuration"
            assert self.locomotion.base_joints is not None, "Missing base joint configuration"
            if self.locomotion.type == "two_wheel":
                assert self.locomotion.wheel_radius is not None, "Missing wheel radius for two-wheel robot"
                assert self.locomotion.wheel_axle_length is not None, "Missing wheel axle length for two-wheel robot"
            elif self.locomotion.type == "holonomic":
                assert (
                    self.locomotion.max_linear_velocity is not None
                ), "Missing max linear velocity for holonomic robot"
                assert (
                    self.locomotion.max_angular_velocity is not None
                ), "Missing max angular velocity for holonomic robot"

        # Validate trunk config
        if self.trunk.enabled:
            assert "trunk" in self.controllers, "Missing trunk controller configuration"
            assert self.trunk.joints is not None, "Missing trunk joint configuration"

        # Validate camera config
        if self.camera.enabled:
            assert "camera" in self.controllers, "Missing camera controller configuration"
            assert self.camera.joints is not None, "Missing camera joint configuration"

        return True
