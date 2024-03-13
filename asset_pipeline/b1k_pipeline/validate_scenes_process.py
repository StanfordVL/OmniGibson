"""
Script to validate a scene for physics
"""
import json
import sys
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET

from omnigibson.macros import gm

# Set some macros. Is this kosher?
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False

import omnigibson as og

MAX_POS_DELTA = 0.05  # 5cm
MAX_ORN_DELTA = np.deg2rad(10)  # 10 degrees

def get_poses_in_urdf(scene):
    # First, load the URDF file as an XML
    urdf_path = os.path.join(gm.DATASET_PATH, "scenes", scene, "urdf", f"{scene}_best.urdf")
    # Get the joint axes in the object
    with open(urdf_path) as f:
        tree = ET.parse(f)
    joints = list(tree.findall('.//joint'))

    states = {}
    for joint in joints:
        name = joint.attrib['name'].replace("j_", "").replace("-", "_")
        position = np.array([float(x) for x in joint.find('origin').attrib['xyz'].split(" ")])
        orientation = np.array([float(x) for x in joint.find('origin').attrib['rpy'].split(" ")])
        states[name] = (position, R.from_euler("xyz", orientation).as_quat())

    return states

if __name__ == "__main__":
    dataset_root = sys.argv[1]
    scene = sys.argv[2]
    out_path = os.path.join(sys.argv[3], f"{scene}.json")
    gm.DATASET_PATH = str(dataset_root)

    from omnigibson.systems import import_og_systems
    import_og_systems()

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

    # Objects initial poses from the URDF (these are bbox poses)
    urdf_poses = get_poses_in_urdf(scene)

    # Filter objects down
    objs = [x for x in env.scene.objects if x.name in urdf_poses]
    if len(objs != len(urdf_poses)):
        urdf_keys = set(urdf_poses.keys())
        scene_keys = set(x.name for x in objs)
        print("Warning: some objects in the URDF were not found in the scene.")
        print("In URDF but not scene:", urdf_keys - scene_keys)
        print("In scene but not URDF:", scene_keys - urdf_keys)

    links = {obj.name + "-" + link_name: link for obj in objs for link_name, link in obj.links.items()}

    # Run the simulation
    print("Stepping simulation.")
    for _ in range(150):
        env.step([])
    print("Done stepping simulation.")

    # Get their new poses
    final_poses = {obj.name: obj.get_position_orientation() for obj in objs}

    # Compare the poses of objects
    mismatches = []
    for obj in objs:
        old_pos, old_orn = urdf_poses[obj.name]
        new_pos, new_orn = final_poses[obj.name]

        delta_pos = np.linalg.norm(np.array(new_pos) - np.array(old_pos))
        if delta_pos > MAX_POS_DELTA:
            mismatches.append(f"{obj.name} position changed by {delta_pos} meters from {old_pos} to {new_pos}.")
        delta_orn_mag = (R.from_quat(new_orn) * R.from_quat(old_orn).inv()).magnitude()
        if delta_orn_mag > MAX_ORN_DELTA:
            mismatches.append(f"{obj.name} orientation changed by {delta_orn_mag} rads from {old_orn} to {new_orn}.")

    # Also check velocities of all links
    for link_name, link in links.items():
        vel = link.get_linear_velocity()
        if np.linalg.norm(vel) > 0.01:
            mismatches.append(f"{link_name} has a non-zero velocity of {vel}.")

    # Save the results
    with open(out_path, "w") as f:
        json.dump(mismatches, f)

    og.shutdown()
