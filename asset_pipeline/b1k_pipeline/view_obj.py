import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm

import igibson
igibson.ig_dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/aggregate")

from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.empty_scene import EmptyScene
from igibson.objects.articulated_object import URDFObject
from igibson.external.pybullet_tools import utils


def main():
    # Load the scene into iGibson 2
    s = Simulator(mode="headless", use_pb_gui=True, image_height=1080, image_width=1920)
    scene = EmptyScene()
    s.import_scene(scene)

    obj = URDFObject(os.path.join(igibson.ig_dataset_path, "objects", "fridge", "hivvdf", "hivvdf.urdf"))
    s.import_object(obj)

    # Step the simulation by 5 seconds.
    while True:
        s.step()

if __name__ == "__main__":
    main()