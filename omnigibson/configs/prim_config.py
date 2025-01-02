from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from omegaconf import MISSING

@dataclass
class PrimConfig:
    """Base configuration for all prims"""
    relative_prim_path: str = MISSING
    name: str = MISSING
    load_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class XFormPrimConfig(PrimConfig):
    """Configuration for XForm prims"""
    scale: Optional[List[float]] = None

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
class JointPrimConfig(PrimConfig):
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
