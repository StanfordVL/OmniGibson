from collections import defaultdict
import glob
import json


OBJECT_FILE_GLOB = "*/artifacts/object_list.json"
DEFAULT_PATH = "artifacts/object_inventory.json"

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

        needed |= object_list["needed_objects"]
        for provided in object_list["provided_objects"]:
            providers[provided].append(object_file)

    # Check the multiple-provided copies.
    multiple_provided = {k: v for k, v in providers.items() if len(v) > 1}
    single_provider = {k: v[0] for k, v in providers.items()}

    provided_objects = set(single_provider.keys())
    missing_objects = needed - provided_objects

    success = len(skipped_files) == 0 and len(multiple_provided) == 0 and len(missing_objects) == 0
    with open(DEFAULT_PATH, "w") as f:
        json.dump({
            "success": success,
            "providers": single_provider,
            "error_skipped_files": skipped_files,
            "error_multiple_provided": multiple_provided,
            "error_missing_objects": missing_objects
        }, f, indent=4)

if __name__ == "__main__":
    main()