"""
Script to import scene and objects
"""
import sys

from omnigibson.macros import gm

# Set some macros. Is this kosher?
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True

import omnigibson as og
from omnigibson import app
from omnigibson.systems import REGISTERED_SYSTEMS, FluidSystem

from b1k_pipeline.usd_conversion.convert_scene_urdfs_to_json_templates import (
    convert_scene_urdf_to_json,
)

IMPORT_RENDER_CHANNELS = True


if __name__ == "__main__":
    dataset_root = sys.argv[1]
    gm.DATASET_PATH = str(dataset_root)

    # Generate systems
    cats = ['water', 'dust', 'dirt', 'debris', 'bunchgrass', 'mud', 'mold', 'mildew', 'baby_oil', 'coconut_oil', 'cooking_oil', 'essential_oil', 'linseed_oil', 'olive_oil', 'sesame_oil', 'stain', 'ink', 'alga', 'spray_paint', 'house_paint', 'rust', 'patina', 'incision', 'tarnish', 'calcium_carbonate', 'wrinkle']
    for cat in cats:
        if cat not in REGISTERED_SYSTEMS:
            FluidSystem.create(
                name=cat.replace("-", "_"),
                particle_contact_offset=0.012,
                particle_density=500.0,
                is_viscous=False,
                material_mtl_name="DeepWater",
            )

    scene = sys.argv[2]
    urdf_path = f"{dataset_root}/scenes/{scene}/urdf/{scene}_best.urdf"
    json_path = f"{dataset_root}/scenes/{scene}/json/{scene}_best.json"

    # Convert URDF to USD
    convert_scene_urdf_to_json(urdf=urdf_path, json_path=json_path)

    # Clear the sim
    og.sim.clear()

    og.shutdown()
