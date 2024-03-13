"""
Script to validate a scene for physics
"""
import json
import sys
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from omnigibson.macros import gm

# Set some macros. Is this kosher?
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False

import omnigibson as og
from omnigibson import app
from omnigibson.systems import REGISTERED_SYSTEMS, FluidSystem

MAX_POS_DELTA = 0.1  # 10cm
MAX_ORN_DELTA = np.deg2rad(10)  # 10 degrees


if __name__ == "__main__":
    dataset_root = sys.argv[1]
    scene = sys.argv[2]
    out_path = os.path.join(sys.argv[3], f"{scene}.json")
    gm.DATASET_PATH = str(dataset_root)

    # Generate systems
    # cats = ['water', 'dust', 'dirt', 'debris', 'bunchgrass', 'mud', 'mold', 'mildew', 'baby_oil', 'coconut_oil', 'cooking_oil', 'essential_oil', 'linseed_oil', 'olive_oil', 'sesame_oil', 'stain', 'ink', 'alga', 'spray_paint', 'house_paint', 'rust', 'patina', 'incision', 'tarnish', 'calcium_carbonate', 'wrinkle']
    # for cat in cats:
    #     if cat not in REGISTERED_SYSTEMS:
    #         FluidSystem.create(
    #             name=cat.replace("-", "_"),
    #             particle_contact_offset=0.012,
    #             particle_density=500.0,
    #             is_viscous=False,
    #             material_mtl_name="DeepWater",
    #         )

    # Load the sim and do stuff
    # If the scene type is interactive, also check if we want to quick load or full load the scene
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene,
        },
    }

    # Load the environment
    env = og.Environment(configs=cfg)

    # Get all the RigidPrims from the scene
    objs = env.scene.objects
    links = {obj.name + "-" + link_name: link for obj in objs for link_name, link in obj.links.items()}

    # Store their poses
    initial_poses = {link_name: link.get_position_orientation() for link_name, link in links.items()}

    # Run the simulation
    print("Stepping simulation.")
    for _ in range(150):
        env.step([])
    print("Done stepping simulation.")

    # Get their new poses
    final_poses = {link_name: link.get_position_orientation() for link_name, link in links.items()}

    # Compare the poses
    mismatches = []
    for link_name in links:
        old_pos, old_orn = initial_poses[link_name]
        new_pos, new_orn = final_poses[link_name]

        delta_pos = np.linalg.norm(np.array(new_pos) - np.array(old_pos))
        if delta_pos > MAX_POS_DELTA:
            mismatches.append(f"{link_name} position changed by {delta_pos} meters from {old_pos} to {new_pos}.")
        delta_orn_mag = (R.from_quat(new_orn) * R.from_quat(old_orn).inv()).magnitude()
        if delta_orn_mag > MAX_ORN_DELTA:
            mismatches.append(f"{link_name} orientation changed by {delta_orn_mag} rads from {old_orn} to {new_orn}.")

    # Save the results
    with open(out_path, "w") as f:
        json.dump(mismatches, f)

    og.shutdown()
