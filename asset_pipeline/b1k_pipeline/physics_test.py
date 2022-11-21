"""
This script is used to load a newly-exported scene and check it for physics stability.

To run this script, you need to have a valid version of iGibson 2.0 installed in your environment.
"""

import os
import sys

from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene

OUTPUT_FILENAME = "physics_test.json"
SUCCESS_FILENAME = "physics_test.success"
DATASET_PATH = ""



def main():
    scene_path = sys.argv[1]
    scene_name = os.path.basename(scene_path)

    s = Simulator(mode="headless")

    try:
        scene = InteractiveIndoorScene(
            scene_name,
            build_graph=True,
        )
        s.import_scene(scene)

        output_dir = os.path.join(rt.maxFilePath, "artifacts")
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, OUTPUT_FILENAME)
        with open(filename, "w") as f:
            json.dump({"success": success, "needed_objects": needed, "provided_objects": provided, "object_counts": counts, "error_invalid_name": sorted(nomatch)}, f, indent=4)

        if success:
            with open(os.path.join(output_dir, SUCCESS_FILENAME), "w") as f:
                pass

    finally:
        s.disconnect()


if __name__ == "__main__":
    main()