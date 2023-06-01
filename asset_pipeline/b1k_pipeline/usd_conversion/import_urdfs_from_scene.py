import xml.etree.ElementTree as ET
from copy import deepcopy
from os.path import exists

import omni
import omni.client
import omni.kit.commands
import omnigibson
from omni.client._omniclient import Result
from omni.isaac.urdf import URDFCreateImportConfig, URDFParseAndImportFile
from omni.physx.scripts import utils
from omnigibson import app
from omnigibson.simulator import Simulator
from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdLux, UsdPhysics
from pxr.Sdf import ValueTypeNames as VT

from b1k_pipeline.usd_conversion.expand_collision_obj_and_urdf import split_objs_in_urdf
from b1k_pipeline.usd_conversion.preprocess_urdf_for_metalinks import (
    update_obj_urdf_with_metalinks,
)


def create_import_config():
    # Set up import configuration
    import_config = URDFCreateImportConfig().do()
    drive_mode = (
        import_config.default_drive_type.__class__
    )  # Hacky way to get class for default drive type, options are JOINT_DRIVE_{NONE / POSITION / VELOCITY}

    import_config.set_merge_fixed_joints(False)
    import_config.set_convex_decomp(True)
    import_config.set_fix_base(False)
    import_config.set_import_inertia_tensor(False)
    import_config.set_distance_scale(1.0)
    import_config.set_density(0.0)
    import_config.set_default_drive_type(drive_mode.JOINT_DRIVE_NONE)
    import_config.set_default_drive_strength(0.0)
    import_config.set_default_position_drive_damping(0.0)
    import_config.set_self_collision(False)
    import_config.set_up_vector(0, 0, 1)
    import_config.set_make_default_prim(True)
    import_config.set_create_physics_scene(True)
    return import_config


def import_obj_urdf(obj_category, obj_model, dataset_root, skip_if_exist=False):
    # Preprocess input URDF to account for metalinks
    urdf_path = update_obj_urdf_with_metalinks(
        obj_category=obj_category, obj_model=obj_model, dataset_root=dataset_root
    )
    # Import URDF
    cfg = create_import_config()
    # Check if filepath exists
    usd_path = f"{dataset_root}/objects/{obj_category}/{obj_model}/usd/{obj_model}.usd"
    if not (skip_if_exist and exists(usd_path)):
        print(f"Converting collision meshes from {obj_category}, {obj_model}...")
        urdf_path = split_objs_in_urdf(urdf_fpath=urdf_path, name_suffix="split")
        print(f"Importing {obj_category}, {obj_model}...")
        # Only import if it doesn't exist
        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=cfg,
            dest_path=usd_path,
        )
