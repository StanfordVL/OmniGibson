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
        print(DeepDiff(base_manifest_by_target[target], pr_manifest_by_target[target]).pretty())

if __name__ == "__main__":
    main()