"""
Script to import scene and objects
"""
import os

import tqdm
from omnigibson.macros import gm

# Set some macros. Is this kosher?
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True

import omnigibson as og
from omnigibson import app

from b1k_pipeline.usd_conversion.convert_scene_urdfs_to_json_templates import (
    convert_scene_urdf_to_json,
)
from b1k_pipeline.usd_conversion.utils import DATASET_ROOT

IMPORT_RENDER_CHANNELS = True


if __name__ == "__main__":
    scenes = list(os.listdir(os.path.join(DATASET_ROOT, "scenes")))
    for scene in tqdm.tqdm(scenes):
        urdf_path = f"{DATASET_ROOT}/scenes/{scene}/urdf/{scene}_best.urdf"
        json_path = f"{DATASET_ROOT}/scenes/{scene}/json/{scene}_best.json"

        # Convert URDF to USD
        convert_scene_urdf_to_json(urdf=urdf_path, json_path=json_path)

        # Clear the sim
        og.sim.clear()

    og.shutdown()
