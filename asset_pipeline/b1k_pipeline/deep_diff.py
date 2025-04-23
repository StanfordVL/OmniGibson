import json
import pathlib
from deepdiff import DeepDiff

def main():
    deep_glob = "cad/*/*/artifacts/file_manifest_deep.json"
    base_files = pathlib.Path("base").glob(deep_glob)
    base_manifest_by_target = {"/".join(f.parts[-4:-2]): f for f in base_files}
    pr_files = pathlib.Path("pr").glob(deep_glob)
    pr_manifest_by_target = {"/".join(f.parts[-4:-2]): f for f in pr_files}
    all_targets = set(base_manifest_by_target.keys()) | set(pr_manifest_by_target.keys())

    for target in all_targets:
        print(f"\n\n{target}")
        base_manifest = json.loads(base_manifest_by_target[target].read_text()) if target in base_manifest_by_target else []
        pr_manifest = json.loads(pr_manifest_by_target[target].read_text()) if target in pr_manifest_by_target else []
        print(DeepDiff(base_manifest, pr_manifest).pretty())

if __name__ == "__main__":
    main()