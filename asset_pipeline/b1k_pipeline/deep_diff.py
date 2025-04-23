import json
import pathlib
from deepdiff import DeepDiff

def main():
    deep_glob = "cad/*/*/artifacts/file_manifest.json"  # maybe use _deep ?
    base_files = pathlib.Path("base").glob(deep_glob)
    base_manifest_by_target = {"/".join(f.parts[-4:-2]): f for f in base_files}
    pr_files = pathlib.Path("pr").glob(deep_glob)
    pr_manifest_by_target = {"/".join(f.parts[-4:-2]): f for f in pr_files}
    all_targets = set(base_manifest_by_target.keys()) | set(pr_manifest_by_target.keys())

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
        print("  Affected objects:")
        for root in sorted(diff.affected_root_keys):
            print("    " + str(root))

        print()
        print(diff.pretty())

if __name__ == "__main__":
    main()