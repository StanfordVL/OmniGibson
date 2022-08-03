
"""Regenerate the params.yaml file to contain all of the object and scenes."""

import os
import yaml

OBJS_DIR = "../cad/objects"
SCENES_DIR = "../cad/scenes"
OUT_PATH = "../params.yaml"

def main():
    objects_path = os.path.join(os.path.dirname(__file__), OBJS_DIR)
    objects = [] # sorted(["objects/" + x for x in os.listdir(objects_path)])

    scenes_path = os.path.join(os.path.dirname(__file__), OBJS_DIR)
    scenes = ["scenes/restaurant_hotel"] # sorted(["scenes/" + x for x in os.listdir(scenes_path)])

    combined = objects + scenes

    out_path = os.path.join(os.path.dirname(__file__), OUT_PATH)
    with open(out_path, "w") as f:
        yaml.dump({"objects": objects, "scenes": scenes, "combined": combined}, f)

    print("Params updated successfully.")

if __name__ == "__main__":
    main()