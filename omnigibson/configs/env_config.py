from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import MISSING

from omnigibson.configs.object_config import ObjectConfig, RobotConfig


@dataclass
class RenderConfig:
    viewer_width: int = 1280
    viewer_height: int = 720


@dataclass
class EnvConfig:
    action_frequency: int = 60
    rendering_frequency: int = 60
    physics_frequency: int = 60
    device: Optional[str] = None
    automatic_reset: bool = False
    flatten_action_space: bool = False
    flatten_obs_space: bool = False
    initial_pos_z_offset: float = 0.1
    external_sensors: Optional[List[Dict[str, Any]]] = None


@dataclass
class SceneConfig:
    type: str = MISSING
    model: str = MISSING
    waypoint_resolution: float = 0.2
    num_waypoints: int = 10
    trav_map_resolution: float = 0.1
    default_erosion_radius: float = 0.0
    trav_map_with_objects: bool = True
    scene_instance: Optional[str] = None
    scene_file: Optional[str] = None


@dataclass
class TaskConfig:
    type: str = "DummyTask"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WrapperConfig:
    type: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OmniGibsonConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    robots: List[RobotConfig] = field(default_factory=list)
    objects: List[ObjectConfig] = field(default_factory=list)
    task: TaskConfig = field(default_factory=TaskConfig)
    wrapper: WrapperConfig = field(default_factory=WrapperConfig)
