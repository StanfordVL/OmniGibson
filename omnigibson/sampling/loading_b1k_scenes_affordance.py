import argparse
import copy
import csv
import json
import logging
import os
import pkgutil
import random
import time
import traceback

import bddl
import numpy as np
import yaml
from bddl.activity import Conditions, evaluate_state
from IPython import embed
from PIL import Image
from utils import *

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm, macros
from omnigibson.object_states import Contains
from omnigibson.objects import DatasetObject
from omnigibson.systems import (
    MicroPhysicalParticleSystem,
    get_system,
    remove_callback_on_system_clear,
    remove_callback_on_system_init,
)
from omnigibson.systems.system_base import PhysicalParticleSystem, VisualParticleSystem, clear_all_systems
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY, BDDLEntity
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import clear as clear_pu
from omnigibson.utils.python_utils import create_object_from_init_info

# TODO:
# 1. Set boundingCube approximation earlier (maybe right after importing the scene objects). Otherwise after loading the robot, we will elapse one physics step
# 2. Enable transition rule and refresh all rules before online validation

parser = argparse.ArgumentParser()
parser.add_argument("--scene_model", type=str, default=None, help="Scene model to sample tasks in")
parser.add_argument(
    "--activities",
    type=str,
    default=None,
    help="Activity/ie(s) to be sampled, if specified. This should be a comma-delimited list of desired activities. Otherwise, will try to sample all tasks in this scene",
)
parser.add_argument("--n_obj_rand", type=int, default=1, help="Number of times to randomly sample object instances")
parser.add_argument("--n_pose_rand", type=int, default=1, help="Number of times to randomly sample object poses")

gm.HEADLESS = False
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

CAMERA_POSE = ([0.03938865, -0.06428928, 1.91435064], [0.54142459, 0.12331421, 0.18468735, 0.81089062])


def main(random_selection=False, headless=False, short_exec=False):
    args = parser.parse_args()

    # If we want to create a stable scene config, do that now
    default_scene_fpath = f"{gm.DATASET_PATH}/scenes/{args.scene_model}/json/{args.scene_model}_best.json"

    # Define the configuration to load -- we'll use a Fetch
    cfg = {
        # Use default frequency
        "env": {
            "action_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_file": default_scene_fpath,
            "scene_model": args.scene_model,
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb"],
                "grasping_mode": "physical",
                "default_arm_pose": "diagonal45",
                "default_reset_mode": "untuck",
                "position": np.ones(3) * -50.0,
            },
        ],
    }
    env = og.Environment(configs=copy.deepcopy(cfg))
    og.sim.stop()

    # embed()

    scene_object_names = set([obj.name for obj in og.sim.scene.objects])

    activities = args.activities.split(",")

    env.task_config["type"] = "BehaviorTask"
    env.task_config["online_object_sampling"] = False
    for activity in activities:
        env.task_config["activity_name"] = activity
        for obj_rand_idx in range(args.n_obj_rand):
            for pose_rand_idx in range(args.n_pose_rand):
                assert og.sim.is_stopped()

                # embed()
                activity_instance_id = obj_rand_idx * args.n_pose_rand + pose_rand_idx
                env.task_config["activity_instance_id"] = activity_instance_id

                fname = BehaviorTask.get_cached_activity_scene_filename(
                    scene_model=args.scene_model,
                    activity_name=activity,
                    activity_definition_id=0,
                    activity_instance_id=activity_instance_id,
                )
                task_json_path = os.path.join(
                    gm.DATASET_PATH, "scenes", og.sim.scene.scene_model, "json", f"{fname}.json"
                )
                with open(task_json_path, "r") as f:
                    task_json = json.load(f)

                objs_added = []
                for i, (key, val) in enumerate(task_json["objects_info"]["init_info"].items()):
                    if key in scene_object_names:
                        continue
                    obj = create_object_from_init_info(val)
                    og.sim.import_object(obj)
                    obj.set_position([100, 100, 100 + i])
                    objs_added.append(obj)

                og.sim.write_metadata("task", task_json["metadata"]["task"])
                env._load_task()

                og.sim.play()
                og.sim.load_state(task_json["state"])
                og.sim.step()

                # Run policy rollout or demo collection
                og.sim.viewer_camera.set_position_orientation(*CAMERA_POSE)
                og.sim.step()
                for _ in range(10):
                    og.sim.render()

                Image.fromarray(og.sim.viewer_camera.get_obs()[0]["rgb"]).save(f"{activity_instance_id}.png")

                og.sim.stop()
                for obj in objs_added:
                    og.sim.remove_object(obj)
                clear_all_systems()
                og.sim.write_metadata("task", None)

    print("Successful shutdown!")


if __name__ == "__main__":
    main()

    # Shutdown at the end
    og.shutdown()
