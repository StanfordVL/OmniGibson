from collections import Counter, defaultdict
import glob
import json
import os


OBJECT_FILE_GLOB = os.path.join(os.path.dirname(__file__), "../cad/*/*/artifacts/object_list.json")
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/pipeline/object_inventory.json")
SUCCESS_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/pipeline/object_inventory.success")

RELPATH_BASE = os.path.join(os.path.dirname(__file__), "../cad")

def main():
    needed = set()
    providers = defaultdict(list)
    skipped_files = []

    # Merge the object lists.
    for object_file in glob.glob(OBJECT_FILE_GLOB):
        with open(object_file, "r") as f:
            object_list = json.load(f)

        if not object_list["success"]:
            skipped_files.append(object_file)
            continue

        scene_or_obj_dir = os.path.dirname(os.path.dirname(object_file))
        path_to_record = os.path.relpath(scene_or_obj_dir, RELPATH_BASE).replace("\\", "/")
        needed |= set(object_list["needed_objects"])
        for provided in object_list["provided_objects"]:
            providers[provided].append(path_to_record)

    # Check the multiple-provided copies.
    multiple_provided = {k: v for k, v in providers.items() if len(v) > 1}
    single_provider = {k: v[0] for k, v in providers.items()}

    provided_objects = set(single_provider.keys())
    missing_objects = needed - provided_objects

    id_occurrences = defaultdict(list)
    for obj in provided_objects:
        id_occurrences[obj.split("-")[1]].append(id_occurrences)
    id_collisions = {obj_id: obj_names for obj_id, obj_names in id_occurrences.items() if len(obj_names) > 1}

    success = len(skipped_files) == 0 and len(multiple_provided) == 0 and len(missing_objects) == 0 and len(id_collisions) == 0
    with open(DEFAULT_PATH, "w") as f:
        json.dump({
            "success": success,
            "providers": single_provider,
            "error_skipped_files": sorted(skipped_files),
            "error_multiple_provided": multiple_provided,
            "error_missing_objects": sorted(missing_objects),
            "error_id_collisions": id_collisions,
        }, f, indent=4)

    if success:
        with open(SUCCESS_PATH, "w") as f:
            pass

if __name__ == "__main__":
    main()