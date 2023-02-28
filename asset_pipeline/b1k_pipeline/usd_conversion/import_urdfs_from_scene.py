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
from b1k_pipeline.usd_conversion.utils import DATASET_ROOT


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


def import_objects_from_scene_urdf(urdf):
    tree = ET.parse(urdf)
    root = tree.getroot()
    import_nested_objs_from_element(root)


def import_nested_objs_from_element(element):
    obj_infos = set()
    # Check if this element is a link
    for ele in element:
        if ele.tag == "link":
            # This is a valid object, import the model
            category = ele.get("category")
            model = ele.get("model")
            name = ele.get("name").replace("-", "_")
            print(f"Link: name: {name}, category: {category}, model: {model}")
            obj_info = (category, model)
            # Skip world link
            if name == "world":
                pass
            elif obj_info not in obj_infos:
                import_obj_urdf(
                    obj_category=category, obj_model=model, skip_if_exist=False
                )
                obj_infos.add(obj_info)
        # If there's children nodes, we iterate over those
        for child in ele:
            import_nested_objs_from_element(child)


def import_obj_urdf(obj_category, obj_model, skip_if_exist=False):
    # Preprocess input URDF to account for metalinks
    urdf_path = update_obj_urdf_with_metalinks(
        obj_category=obj_category, obj_model=obj_model
    )
    # Import URDF
    cfg = create_import_config()
    # Check if filepath exists
    usd_path = f"{DATASET_ROOT}/objects/{obj_category}/{obj_model}/usd/{obj_model}.usd"
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
