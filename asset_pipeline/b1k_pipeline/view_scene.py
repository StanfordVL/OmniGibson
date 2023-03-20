import sys


import json
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p
import tqdm

from b1k_pipeline.utils import PIPELINE_ROOT

import igibson
igibson.ignore_visual_shape = False
igibson.ig_dataset_path = PIPELINE_ROOT / "artifacts/aggregate"

from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.external.pybullet_tools import utils

def main():
    scene_name = sys.argv[1]
    target = "scenes/" + scene_name
    input_dir = PIPELINE_ROOT / "artifacts" / "aggregate" / target
    scene_filename = input_dir / f"urdf/{scene_name}_best.urdf"

    # Load the scene into iGibson 2
    s = Simulator(mode="headless", use_pb_gui=True)
    try:
        scene = InteractiveIndoorScene(scene_name, urdf_path=str(scene_filename), load_object_categories=["walls", "floors", "door"])
        s.import_scene(scene)

        print("Stepping")
        while True:
            s.step()
    finally:
        s.disconnect()

if __name__ == "__main__":
    main()