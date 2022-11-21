import json
import os
import shutil
import sys
import traceback
import b1k_pipeline.utils
import yaml


PARAMS_FILE = b1k_pipeline.utils.PIPELINE_ROOT / "params.yaml"
OUTPUT_FILENAME = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "pipeline" / "aggregate_scenes.json"
SUCCESS_FILENAME = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "pipeline" / "aggregate_scenes.success"
SCENE_ROOT_DIR = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "aggregate" / "scenes"

def main():
    os.makedirs(SCENE_ROOT_DIR, exist_ok=True)

    success = True
    error_msg = ""
    try:
        # Get the scene list.
        with open(PARAMS_FILE, "r") as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
            targets = params["final_scenes"]

        # Copy over the stuff
        for target in targets:
            scene_name = target.split("/")[1]
            source_dir = b1k_pipeline.utils.PIPELINE_ROOT / "cad" / target / "artifacts" / "scene"
            destination_dir = SCENE_ROOT_DIR / scene_name
            shutil.copytree(source_dir, destination_dir, symlinks=True, dirs_exist_ok=True)

    except Exception as e:
        success = False
        error_msg = traceback.format_exc()

    with open(OUTPUT_FILENAME, "w") as f:
        json.dump({"success": success, "error_msg": error_msg}, f, indent=4)

    if success:
        with open(SUCCESS_FILENAME, "w") as f:
            pass

if __name__ == "__main__":
    main()