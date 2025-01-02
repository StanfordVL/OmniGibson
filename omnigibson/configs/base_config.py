from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from omegaconf import MISSING

@dataclass 
class BaseConfig:
    """Base configuration class that all other configs inherit from"""
    name: str = MISSING

@dataclass
class BasePrimConfig(BaseConfig):
    """Base configuration for all prims"""
    relative_prim_path: Optional[str] = None
    load_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EntityConfig(BasePrimConfig):
    """Configuration for entities"""
    scale: Optional[List[float]] = None
    visible: bool = True
    visual_only: bool = False
    kinematic_only: Optional[bool] = None
    self_collisions: bool = False
    prim_type: str = MISSING

@dataclass
class ObjectConfig(EntityConfig):
    """Configuration for all objects"""
    category: str = "object"
    fixed_base: bool = False
    abilities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    include_default_states: bool = True

@dataclass
class StatefulObjectConfig(ObjectConfig):
    """Configuration for stateful objects"""
    abilities: Optional[Dict[str, Dict[str, Any]]] = None
    include_default_states: bool = True

@dataclass
class USDObjectConfig(StatefulObjectConfig):
    """Configuration for USD-based objects"""
    usd_path: str = MISSING
    encrypted: bool = False

@dataclass
class DatasetObjectConfig(StatefulObjectConfig):
    """Configuration for dataset objects"""
    model: Optional[str] = None
    dataset_type: str = "BEHAVIOR"
    bounding_box: Optional[List[float]] = None
    in_rooms: Optional[List[str]] = None

@dataclass
class PrimitiveObjectConfig(StatefulObjectConfig):
    """Configuration for primitive objects"""
    primitive_type: str = MISSING
    rgba: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    radius: Optional[float] = None
    height: Optional[float] = None
    size: Optional[float] = None

@dataclass
class LightObjectConfig(StatefulObjectConfig):
    """Configuration for light objects"""
    light_type: str = MISSING
    radius: float = 1.0
    intensity: float = 50000.0

@dataclass
class ControllableObjectConfig(StatefulObjectConfig):
    """Configuration for controllable objects"""
    control_freq: Optional[float] = None
    controller_config: Optional[Dict[str, Any]] = None
    action_type: str = "continuous"
    action_normalize: bool = True
    reset_joint_pos: Optional[List[float]] = None
