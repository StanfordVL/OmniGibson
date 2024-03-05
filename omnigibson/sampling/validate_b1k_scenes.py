import logging
import os
import yaml
import copy
import time
import argparse
import bddl
import pkgutil
import omnigibson as og
from omnigibson.macros import gm, macros
import json
import csv
import traceback
from omnigibson.objects import DatasetObject
from omnigibson.tasks import BehaviorTask
from omnigibson.systems import remove_callback_on_system_init, remove_callback_on_system_clear, get_system, MicroPhysicalParticleSystem
from omnigibson.systems.system_base import clear_all_systems
from omnigibson.utils.python_utils import clear as clear_pu
from omnigibson.utils.python_utils import create_object_from_init_info
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
from bddl.activity import Conditions, evaluate_state
import numpy as np
import gspread
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--scene_model", type=str, required=True,
                    help="Scene model to sample tasks in")
parser.add_argument("--activities", type=str, default=None,
                    help="Activity/ie(s) to be sampled, if specified. This should be a comma-delimited list of desired activities. Otherwise, will try to sample all tasks in this scene")
parser.add_argument("--start_at", type=str, default=None,
                    help="If specified, activity to start at, ignoring all previous")

gm.HEADLESS = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True

macros.prims.entity_prim.DEFAULT_SLEEP_THRESHOLD = 0.0

def main(random_selection=False, headless=False, short_exec=False):
    args = parser.parse_args()

    # Make sure scene can be sampled by current user
    validate_scene_can_be_sampled(scene=args.scene_model)

    # Get the default scene instance
    default_scene_fpath = f"{gm.DATASET_PATH}/scenes/{args.scene_model}/json/{args.scene_model}_stable.json"
    assert os.path.exists(default_scene_fpath), f"Default scene file {default_scene_fpath} does not exist!"
    with open(default_scene_fpath, "r") as f:
        default_scene_dict = json.load(f)

    valid_tasks = get_valid_tasks()
    n_b1k_tasks = len(valid_tasks)
    mapping = parse_task_mapping(fpath=TASK_INFO_FPATH)
    activities = args.activities

    should_start = args.start_at is None

    # Grab all the spreadsheet data to create pruned list
    data = worksheet.get(f"A2:I{2 + n_b1k_tasks - 1}")
    rows_to_validate = []
    for i, (activity, in_progress, success, validated, scene_id, user, reason, exception, misc) in enumerate(data):
        if activities is not None and activity not in activities:
            continue
        # Skip until we should start
        if not should_start:
            if args.start_at == activity:
                should_start = True
            else:
                continue
        if scene_id != args.scene_model:
            # Not the right scene, continue
            continue
        if success in {"", "DNS"} or int(success) != 1:
            # Not a success, continue
            continue
        if validated != "" and int(validated) == 1:
            # Already validated, continue
            continue
        if user != USER:
            # Not the right user, so doesn't have the stored sampled json, continue
            continue

        # Add to activities to validate (row number)
        rows_to_validate.append(i + 2)

    # Now take pruned list and iterate through to actually validate the scenes
    for row in rows_to_validate:
        # sleep to avoid gspread query limits
        time.sleep(1)

        # Grab row info
        activity, in_progress, success, validated, scene_id, user, reason, exception, misc = worksheet.get(f"A{row}:I{row}")[0]
        print(f"Validating activity: {activity}...")

        # If already validated, continue
        if validated != "" and int(validated) == 1:
            # Already validated, continue
            continue

        # If another thread is already in the process of validating, skip
        if in_progress != "" and int(in_progress) == 1:
            continue

        # Reserve this task by marking in_progress = 1
        worksheet.update_acell(f"B{row}", 1)

        validated, reason = False, ""

        # Define the configuration to load -- we'll use a Fetch
        cfg = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": args.scene_model,
            },
            "task": {
                "type": "BehaviorTask",
                "online_object_sampling": False,
                "activity_name": activity,
            },
            "robots": [
                {
                    "type": "Fetch",
                    "obs_modalities": ["rgb"],
                    "grasping_mode": "physical",
                    "default_arm_pose": "diagonal30",
                    "default_reset_mode": "tuck",
                },
            ],
        }

        # Create the environment
        # env = create_env_with_stable_objects(cfg)
        env = og.Environment(configs=copy.deepcopy(cfg))

        # Attempt to validate
        try:
            # Validate task
            with open(env.scene.scene_file, "r") as f:
                task_scene_dict = json.load(f)

            # from IPython import embed; print("validate_task"); embed()
            validated, error_msg = validate_task(env.task, task_scene_dict, default_scene_dict)

            if validated:
                og.log.info(f"\n\nValidation success: {activity}\n\n")
                reason = ""
            else:
                og.log.error(f"\n\nValidation failed: {activity}.\n\nFeedback: {error_msg}\n\n")
                reason = error_msg

            # Write to google sheets
            cell_list = worksheet.range(f"B{row}:H{row}")
            for cell, val in zip(cell_list,
                                 ("", int(success), int(validated), args.scene_model, USER, reason, "")):
                cell.value = val
            worksheet.update_cells(cell_list)

        except Exception as e:
            traceback_str = f"{traceback.format_exc()}"
            og.log.error(traceback_str)
            og.log.error(f"\n\nCaught exception validating activity {activity} in scene {args.scene_model}:\n\n{e}\n\n")

            # Clear the in_progress reservation and note the exception
            cell_list = worksheet.range(f"B{row}:H{row}")
            for cell, val in zip(cell_list,
                                 ("", int(success), 0, args.scene_model, USER, reason, traceback_str)):
                cell.value = val
            worksheet.update_cells(cell_list)

        try:
            # Stop sim, clear simulator, and re-create environment
            og.sim.stop()
            og.sim.clear()
        except AttributeError as e:
            # This is the "GetPath" error that happens sporatically. It's benign, so we ignore it
            pass

if __name__ == "__main__":
    main()
    print("Successful shutdown!")
    # Shutdown at the end
    og.shutdown()
