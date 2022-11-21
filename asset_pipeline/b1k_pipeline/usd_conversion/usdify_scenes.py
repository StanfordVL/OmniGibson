#!/usr/bin/env python3

import json
import os
import shutil
import sys
import traceback

import yaml

from import_scene_template import import_models_template_from_scene


PARAMS_FILE = os.path.join(os.path.dirname(__file__), "../params.yaml")
OUTPUT_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/pipeline/usdify_scenes.json")
SUCCESS_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/pipeline/usdify_scenes.success")
SCENE_ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/aggregate/scenes")

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
            scene_dir = os.path.join(SCENE_ROOT_DIR, scene_name)
            scene_urdf = os.path.join(scene_dir, f"{scene_name}_best.urdf")
            assert os.path.exists(scene_urdf), f"Scene {scene_name} URDF does not exist."
            scene_usd = os.path.join(scene_dir, f"{scene_name}_best_template.usd")
            import_models_template_from_scene(urdf=scene_urdf, usd_out=scene_usd)
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