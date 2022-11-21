#!/usr/bin/env python3

"""
Script to import scene and objects
"""
import json
import traceback
from import_urdfs_from_scene import import_obj_urdf
from import_metadata import import_obj_metadata
import os


OUTPUT_FILENAME = "usdify_objs.json"
SUCCESS_FILENAME = "usdify_objs.success"


def main():
    success = True
    error_msg = ""
    try:
        metadata_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/pipeline")
        object_inventory_path = os.path.join(os.path.dirname(__file__), "../artifacts/pipeline/object_inventory.json")
        objects_root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/aggregate/objects")

        with open(object_inventory_path, "r") as f:
            object_inventory = json.load(f)

        assert object_inventory["success"], "Object inventory was unsuccessful."

        for obj in object_inventory["providers"].keys():
            obj_category, obj_model = obj.split("-")
            import_obj_urdf(obj_category=obj_category, obj_model=obj_model, skip_if_exist=False)
            import_obj_metadata(obj_category=obj_category, obj_model=obj_model, import_render_channels=True)
    except Exception:
        success = False
        error_msg = traceback.format_exc()

    filename = os.path.join(metadata_output_dir, OUTPUT_FILENAME)
    with open(filename, "w") as f:
        json.dump({"success": success, "error_msg": error_msg}, f, indent=4)

    if success:
        with open(os.path.join(metadata_output_dir, SUCCESS_FILENAME), "w") as f:
            pass

    app.close()
