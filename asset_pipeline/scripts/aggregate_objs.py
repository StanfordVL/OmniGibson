import json
import os
import shutil
import sys
import traceback

OUTPUT_FILENAME = "aggregate_objs.json"
SUCCESS_FILENAME = "aggregate_objs.success"


def main():
    metadata_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/pipeline")
    object_inventory_path = os.path.join(os.path.dirname(__file__), "../artifacts/pipeline/object_inventory.json")
    objects_root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/aggregate/objects")
    os.makedirs(objects_root_dir, exist_ok=True)

    success = True
    error_msg = ""
    try:
        with open(object_inventory_path, "r") as f:
            object_inventory = json.load(f)
    
        assert object_inventory["success"], "Object inventory was unsuccessful."

        copy_list = []  # Tuples in the form of (from, to)
        for obj, provider in object_inventory["providers"].items():
            obj = obj.replace("-", "/")

            src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", provider, "artifacts/objects", obj)
            assert os.path.exists(src_path), f"Could not resolve path {src_path} for object {obj} provided by {provider}."

            dst_path = os.path.join(objects_root_dir, obj)
            copy_list.append((src_path, dst_path))

        # We were able to find everything we are looking for, so let's actually copy things now.
        for src_path, dst_path in copy_list:
            shutil.copytree(src_path, dst_path, symlinks=True, dirs_exist_ok=False)
    except Exception as e:
        success = False
        error_msg = traceback.format_exc()

    filename = os.path.join(metadata_output_dir, OUTPUT_FILENAME)
    with open(filename, "w") as f:
        json.dump({"success": success, "error_msg": error_msg}, f, indent=4)

    if success:
        with open(os.path.join(metadata_output_dir, SUCCESS_FILENAME), "w") as f:
            pass

if __name__ == "__main__":
    main()