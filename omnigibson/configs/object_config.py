from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import MISSING

@dataclass
class PrimConfig:
    """Base configuration for all prims"""
    name: str = MISSING
    prim_type: str = MISSING  
    position: Optional[List[float]] = None
    orientation: Optional[List[float]] = None
    scale: Optional[List[float]] = None
    fixed_base: bool = False
    visible: bool = True
    visual_only: bool = False
    self_collisions: bool = True
    load_config: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ObjectConfig(PrimConfig):
    """Configuration for objects"""
    category: Optional[str] = None
    model: Optional[str] = None
    abilities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    controller_config: Optional[Dict[str, Any]] = None
    
@dataclass
class RobotConfig(ObjectConfig):
    """Configuration for robots"""
    controller_type: Optional[str] = None
    controller_params: Dict[str, Any] = field(default_factory=dict)
    default_joint_pos: Optional[Dict[str, float]] = None
    default_arm_pose: Optional[str] = None
    default_gripper_pose: Optional[str] = None
    base_threshold: float = 0.2
    arm_threshold: float = 0.2
    gripper_threshold: float = 0.2
