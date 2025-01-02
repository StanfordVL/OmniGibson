from omnigibson.configs.base_config import (
    ObjectConfig,
    USDObjectConfig,
    DatasetObjectConfig,
    PrimitiveObjectConfig,
    LightObjectConfig,
    ControllableObjectConfig,
)

__all__ = [
    "ObjectConfig",
    "USDObjectConfig", 
    "DatasetObjectConfig",
    "PrimitiveObjectConfig",
    "LightObjectConfig",
    "ControllableObjectConfig",
]


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
