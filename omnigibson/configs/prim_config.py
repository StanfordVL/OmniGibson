from dataclasses import dataclass
from typing import Optional, Any
from omegaconf import MISSING
from omnigibson.configs.base_config import BasePrimConfig

@dataclass
class XFormPrimConfig(BasePrimConfig):
    """Configuration for XForm prims"""
    pass

@dataclass
class GeomPrimConfig(XFormPrimConfig):
    """Configuration for geometry prims"""
    pass

@dataclass
class CollisionGeomPrimConfig(GeomPrimConfig):
    """Configuration for collision geometry prims"""
    pass

@dataclass
class VisualGeomPrimConfig(GeomPrimConfig):
    """Configuration for visual geometry prims"""
    pass

@dataclass
class CollisionVisualGeomPrimConfig(CollisionGeomPrimConfig, VisualGeomPrimConfig):
    """Configuration for combined collision and visual geometry prims"""
    pass

@dataclass
class RigidPrimConfig(XFormPrimConfig):
    """Configuration for rigid body prims"""
    kinematic_only: Optional[bool] = None
    belongs_to_articulation: Optional[bool] = None
    visual_only: Optional[bool] = None
    mass: Optional[float] = None
    density: Optional[float] = None

@dataclass
class JointPrimConfig(BasePrimConfig):
    """Configuration for joint prims"""
    joint_type: Optional[str] = None
    body0: Optional[str] = None 
    body1: Optional[str] = None
    articulation_view: Optional[Any] = None

@dataclass
class EntityPrimConfig(XFormPrimConfig):
    """Configuration for entity prims"""
    visual_only: Optional[bool] = None
    prim_type: str = MISSING
