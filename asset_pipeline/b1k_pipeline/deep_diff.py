import json
import pathlib
from deepdiff import DeepDiff

from .utils import parse_name

def model_ids_from_objects(objs):
    model_ids = set()
    for obj in objs:
        pn = parse_name(obj)
        if not pn:
            continue
        model_ids.add(pn.group("model_id"))

    return model_ids

def main():
    deep_glob = "cad/*/*/artifacts/file_manifest.json"  # maybe use _deep ?
    base_files = pathlib.Path("base").glob(deep_glob)
    base_manifest_by_target = {"/".join(f.parts[-4:-2]): f for f in base_files}
    pr_files = pathlib.Path("pr").glob(deep_glob)
    pr_manifest_by_target = {"/".join(f.parts[-4:-2]): f for f in pr_files}
    all_targets = set(base_manifest_by_target.keys()) | set(pr_manifest_by_target.keys())

    full_diffs = {}
    object_diffs = {}
    for target in sorted(all_targets):
        print(f"\n\n-------------------------\n{target}")
        
        # Load the manifests
        base_manifest = json.loads(base_manifest_by_target[target].read_text()) if target in base_manifest_by_target else []
        pr_manifest = json.loads(pr_manifest_by_target[target].read_text()) if target in pr_manifest_by_target else []

        # Convert the manifests to dicts based on the object name
        base_manifest_dict = {v["name"]: v for v in base_manifest}
        pr_manifest_dict = {v["name"]: v for v in pr_manifest}

        # Discard the mtl_hash for now. Should not matter until we start material work.
        for val in base_manifest_dict.values():
            del val["mtl_hash"]
        for val in pr_manifest_dict.values():
            del val["mtl_hash"]
        
        diff = DeepDiff(base_manifest_dict, pr_manifest_dict)
        object_diffs[target] = sorted(diff.affected_root_keys)
        full_diffs[target] = diff.pretty()

    print("-------------------------------------------------")
    print("OBJECT DIFFS")
    print("Unique edited obj names:", sum(len(vals) for vals in object_diffs.values()))
    print("Unique edited model IDs:", len({mid for objs in object_diffs.values() for mid in model_ids_from_objects(objs)}))
    print("-------------------------------------------------")
    for target in sorted(all_targets):
        target_objs = sorted(object_diffs[target])
        target_mids = sorted(model_ids_from_objects(target_mids))

        print(f"\n\n-------------------------\n{target}")
        print(f"Objects ({len(target_objs)})")
        for root in target_objs:
            print("    " + str(root))

        print(f"Models ({len(target_mids)})")
        for mid in target_mids:
            print("    " + str(mid))

        print()

    print("-------------------------------------------------")
    print("FULL DIFFS")
    print("-------------------------------------------------")
    for target in sorted(all_targets):
        print(f"\n\n-------------------------\n{target}")
        print("    " + full_diffs[target].replace("\n", "\n    "))
        print()

if __name__ == "__main__":
    main()