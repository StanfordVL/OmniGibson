from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from omegaconf import MISSING

from omnigibson.configs.object_config import ControllableObjectConfig


@dataclass
class ControllerConfig:
    """Base configuration for controllers"""

    name: str = MISSING
    control_freq: Optional[float] = None
    control_limits: Optional[Dict[str, Any]] = None
    command_input_limits: Optional[List[float]] = None
    command_output_limits: Optional[List[float]] = None
    use_delta_commands: bool = False
    use_impedances: bool = False


@dataclass
class JointControllerConfig(ControllerConfig):
    """Configuration for joint controllers"""

    motor_type: str = "position"
    dof_idx: Optional[List[int]] = None
    pos_kp: Optional[float] = None
    pos_kd: Optional[float] = None


@dataclass
class IKControllerConfig(ControllerConfig):
    """Configuration for IK controllers"""

    task_name: str = MISSING
    mode: str = "pose_delta_ori"
    smoothing_filter_size: int = 2
    workspace_pose_limiter: Optional[Dict[str, Any]] = None
    reset_joint_pos: Optional[List[float]] = None


@dataclass
class OSCControllerConfig(ControllerConfig):
    """Configuration for operational space controllers"""

    task_name: str = MISSING
    mode: str = "pose_delta_ori"
    workspace_pose_limiter: Optional[Dict[str, Any]] = None
    reset_joint_pos: Optional[List[float]] = None


@dataclass
class DifferentialDriveConfig(ControllerConfig):
    """Configuration for differential drive controllers"""

    wheel_radius: float = MISSING
    wheel_axle_length: float = MISSING


@dataclass
class GripperControllerConfig(ControllerConfig):
    """Configuration for gripper controllers"""

    mode: str = "binary"
    limit_tolerance: float = 0.001
    inverted: bool = False


@dataclass
class RobotConfig(ControllableObjectConfig):
    """Base configuration for robots"""

    type: str = MISSING
    obs_modalities: List[str] = field(default_factory=lambda: ["rgb", "proprio"])
    proprio_obs: str = "default"
    sensor_config: Optional[Dict[str, Any]] = None
    controllers: Dict[str, ControllerConfig] = field(default_factory=dict)


@dataclass
class ManipulationRobotConfig(RobotConfig):
    """Configuration for manipulation robots"""

    grasping_mode: str = "physical"
    grasping_direction: str = "lower"
    disable_grasp_handling: bool = False


@dataclass
class MobileManipulationRobotConfig(ManipulationRobotConfig):
    """Configuration for mobile manipulation robots"""

    default_reset_mode: str = "untuck"


@dataclass
class UntuckedArmPoseRobotConfig(MobileManipulationRobotConfig):
    """Configuration for robots with untucked arm poses"""

    default_arm_pose: str = "vertical"


@dataclass
class BehaviorRobotConfig(ManipulationRobotConfig):
    """Configuration for behavior robots"""

    use_ghost_hands: bool = True


@dataclass
class FrankaConfig(ManipulationRobotConfig):
    """Configuration for Franka robots"""

    end_effector: str = "gripper"


@dataclass
class A1Config(ManipulationRobotConfig):
    """Configuration for A1 robot"""

    end_effector: str = "inspire"


@dataclass
class TiagoConfig(UntuckedArmPoseRobotConfig):
    """Configuration for Tiago robot"""

    variant: str = "default"


@dataclass
class FetchConfig(UntuckedArmPoseRobotConfig):
    """Configuration for Fetch robot"""

    pass


@dataclass
class R1Config(MobileManipulationRobotConfig):
    """Configuration for R1 robot"""

    pass


@dataclass
class StretchConfig(ManipulationRobotConfig):
    """Configuration for Stretch robot"""

    pass


@dataclass
class VX300SConfig(ManipulationRobotConfig):
    """Configuration for VX300S robot"""

    pass


@dataclass
class TwoWheelRobotConfig(RobotConfig):
    """Configuration for two wheel robots"""

    wheel_radius: float = MISSING
    wheel_axle_length: float = MISSING


@dataclass
class HuskyConfig(TwoWheelRobotConfig):
    """Configuration for Husky robot"""

    pass


@dataclass
class TurtlebotConfig(TwoWheelRobotConfig):
    """Configuration for Turtlebot robot"""

    pass


@dataclass
class FreightConfig(TwoWheelRobotConfig):
    """Configuration for Freight robot"""

    pass


@dataclass
class LocobotConfig(TwoWheelRobotConfig):
    """Configuration for Locobot robot"""

    pass


@dataclass
class ManipulationRobotConfig(RobotConfig):
    """Configuration for manipulation robots"""

    n_arms: int = 1
    arm_names: List[str] = field(default_factory=lambda: ["default"])
    finger_lengths: Dict[str, float] = field(default_factory=dict)
    assisted_grasp_points: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


@dataclass
class LocomotionRobotConfig(RobotConfig):
    """Configuration for locomotion robots"""

    base_joint_names: List[str] = field(default_factory=list)
    base_control_idx: List[int] = field(default_factory=list)


@dataclass
class TwoWheelRobotConfig(LocomotionRobotConfig):
    """Configuration for two wheel robots"""

    wheel_radius: float = MISSING
    wheel_axle_length: float = MISSING


@dataclass
class HolonomicBaseRobotConfig(LocomotionRobotConfig):
    """Configuration for holonomic base robots"""

    base_footprint_link_name: str = MISSING
