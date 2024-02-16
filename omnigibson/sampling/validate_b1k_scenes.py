import logging
import os
import yaml
import copy
import time
import argparse
import bddl
import pkgutil
import omnigibson as og
from omnigibson.macros import gm
import json
import csv
import traceback
from omnigibson.objects import DatasetObject
from omnigibson.tasks import BehaviorTask
from omnigibson.systems import remove_callback_on_system_init, remove_callback_on_system_clear
from omnigibson.systems.system_base import clear_all_systems
from omnigibson.utils.python_utils import clear as clear_pu
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
from bddl.activity import Conditions
import numpy as np
import gspread

"""
1. gcloud auth login
2. gcloud auth application-default login
3. gcloud config set project lucid-inquiry-205018
4. gcloud iam service-accounts create cremebrule
5. gcloud iam service-accounts keys create key.json --iam-account=cremebrule@lucid-inquiry-205018.iam.gserviceaccount.com
6. mv key.json /home/cremebrule/.config/gcloud/key.json
"""

SAMPLING_SHEET_KEY = "1Vt5s3JrFZ6_iCkfzZr0eb9SBt2Pkzx3xxzb4wtjEaDI"
CREDENTIALS = "/home/jdw/.config/gcloud/key.json"
WORKSHEET = "GTC2024"
USER = "cremebrule"

client = gspread.service_account(filename=CREDENTIALS)
worksheet = client.open_by_key(SAMPLING_SHEET_KEY).worksheet(WORKSHEET)

ACTIVITY_TO_ROW = {activity: i + 2 for i, activity in enumerate(worksheet.col_values(1)[1:])}

SCENE_MAPPING_FPATH = "/home/jdw/Downloads/BEHAVIOR-1K Tasks.csv"
SYNSET_INFO_FPATH = "/home/jdw/Downloads/BEHAVIOR-1K Synsets.csv"

UNSUPPORTED_PREDICATES = {"broken", "assembled", "attached"}

parser = argparse.ArgumentParser()
parser.add_argument("--scene_model", type=str, required=True,
                    help="Scene model to sample tasks in")
parser.add_argument("--activities", type=str, default=None,
                    help="Activity/ie(s) to be sampled, if specified. This should be a comma-delimited list of desired activities. Otherwise, will try to sample all tasks in this scene")
parser.add_argument("--start_at", type=str, default=None,
                    help="If specified, activity to start at, ignoring all previous")
parser.add_argument("--overwrite_existing", action="store_true",
                    help="If set, will overwrite any existing tasks that are found. Otherwise, will skip.")

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False


# CAREFUL!! Only run this ONCE before starting sampling!!!
def write_activities_to_spreadsheet():
    valid_tasks_sorted = sorted(get_valid_tasks())
    n_tasks = len(valid_tasks_sorted)
    cell_list = worksheet.range(f"A{2}:A{2 + n_tasks - 1}")
    for cell, task in zip(cell_list, valid_tasks_sorted):
        cell.value = task
    worksheet.update_cells(cell_list)


def get_predicates(conds):
    preds = []
    if isinstance(conds, str):
        return preds
    assert isinstance(conds, list)
    contains_list = np.any([isinstance(ele, list) for ele in conds])
    if contains_list:
        for ele in conds:
            preds += get_predicates(ele)
    else:
        preds.append(conds[0])
    return preds


def get_subjects(conds):
    subjs = []
    if isinstance(conds, str):
        return subjs
    assert isinstance(conds, list)
    contains_list = np.any([isinstance(ele, list) for ele in conds])
    if contains_list:
        for ele in conds:
            subjs += get_subjects(ele)
    else:
        subjs.append(conds[1])
    return subjs


def get_rooms(conds):
    rooms = []
    if isinstance(conds, str):
        return rooms
    assert isinstance(conds, list)
    contains_list = np.any([isinstance(ele, list) for ele in conds])
    if contains_list:
        for ele in conds:
            rooms += get_rooms(ele)
    elif conds[0] == "inroom":
        rooms.append(conds[2])
    return rooms


def get_valid_tasks():
    return set(activity for activity in os.listdir(os.path.join(bddl.__path__[0], "activity_definitions")))


def get_notready_synsets():
    notready_synsets = set()
    with open(SYNSET_INFO_FPATH) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            synset, status = row[:2]
            if status == "Not Ready":
                notready_synsets.add(synset)

    return notready_synsets


def parse_task_mapping(fpath):
    mapping = dict()
    rows = []
    with open(fpath) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            rows.append(row)

    notready_synsets = get_notready_synsets()

    for row in rows[1:]:
        activity_name = row[0].split("-")[0]

        # Skip any that is missing a synset
        required_synsets = set(list(entry.strip() for entry in row[1].split(","))[:-1])
        if len(notready_synsets.intersection(required_synsets)) > 0:
            continue

        # Write matched ready scenes
        ready_scenes = set(list(entry.strip() for entry in row[2].split(","))[:-1])

        # There's always a leading whitespace
        if len(ready_scenes) == 0:
            continue

        mapping[activity_name] = ready_scenes

    return mapping


def get_scene_compatible_activities(scene_model, mapping):
    return [activity for activity, scenes in mapping.items() if scene_model in scenes]


def main(random_selection=False, headless=False, short_exec=False):
    args = parser.parse_args()

    # Define the configuration to load -- we'll use a Fetch
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene_model,
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["scan", "rgb", "depth"],
                "grasping_mode": "physical",
                "default_arm_pose": "diagonal30",
                "default_reset_mode": "tuck",
            },
        ],
    }

    # Create the environment
    # Attempt to sample the activity
    env = og.Environment(configs=copy.deepcopy(cfg))

    # Store the initial state -- this is the safeguard to reset to!
    scene_initial_state = copy.deepcopy(env.scene._initial_state)
    og.sim.stop()

    n_scene_objects = len(env.scene.objects)

    # Set environment configuration after environment is loaded, because we will load the task
    env.task_config["type"] = "BehaviorTask"
    env.task_config["online_object_sampling"] = True

    #####################################

    valid_tasks = get_valid_tasks()
    n_b1k_tasks = len(valid_tasks)
    mapping = parse_task_mapping(fpath=SCENE_MAPPING_FPATH)
    activities = set(get_scene_compatible_activities(scene_model=args.scene_model, mapping=mapping) \
        if args.activities is None else args.activities.split(","))

    should_start = args.start_at is None

    # Grab all the spreadsheet data to create pruned list
    data = worksheet.get(f"A2:I{2 + n_b1k_tasks - 1}")
    rows_to_validate = []
    for i, (activity, in_progress, success, validated, scene_id, user, reason, exception, misc) in enumerate(data):
        # Skip until we should start
        if not should_start:
            if args.start_at == activity:
                should_start = True
            else:
                continue
        if scene_id != args.scene_model:
            # Not the right scene, continue
            continue
        if success == "" or int(success) != 1:
            # Not a success, continue
            continue
        if validated != "" and int(validated) == 1:
            # Already validated, continue
            continue
        if user != USER:
            # Not the right user, so doesn't have the stored sampled json, continue
            continue

        # Add to activities to validate (row number, activity
        rows_to_validate.append(i + 1)

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

        success, reason = False, ""

        conditions = Conditions(activity, 0, simulator_name="omnigibson")
        env.task_config["activity_name"] = activity
        scene_instance = BehaviorTask.get_cached_activity_scene_filename(
            scene_model=args.scene_model,
            activity_name=activity,
            activity_definition_id=0,
            activity_instance_id=0,
        )

        # Make sure sim is stopped
        assert og.sim.is_stopped()

        # Attempt to validate
        try:
            # 1. Validate the native USDs jsons are stable and similar -- compare all object kinematics (poses, joint
            #       states) with respect to the native scene file

            # 2. Validate loading the task is stable -- all object positions and velocities should be somewhat stable
            #       after a physics timestep occurs

            # 3. Validate object set is consistent (no faulty transition rules occurring) -- we expect the number
            #       of active systems (and number of active particles) and the number of objects to be the same after
            #       taking a physics step

            # 4. Validate longer-term stability -- take N=10 timesteps, and make sure all object positions and velocities
            #       are still stable (positions don't drift too much, and velocities are close to 0)


            relevant_rooms = set(get_rooms(conditions.parsed_initial_conditions))
            print(f"relevant rooms: {relevant_rooms}")
            for obj in og.sim.scene.objects:
                if isinstance(obj, DatasetObject):
                    obj_rooms = {"_".join(room.split("_")[:-1]) for room in obj.in_rooms}
                    active = len(relevant_rooms.intersection(obj_rooms)) > 0
                    obj.visual_only = not active
                    obj.visible = active

            og.log.info(f"Sampling task: {activity}")
            env._load_task()
            assert og.sim.is_stopped()

            success = env.task.feedback is None
            if success:
                # Sampling success
                og.sim.play()
                # This will actually reset the objects to their sample poses
                env.task.reset(env)

                # TODO: figure out whether we also should update in_room for newly imported objects
                env.task.save_task(override=args.overwrite_existing)

                og.sim.stop()
                og.log.info(f"\n\nSampling success: {activity}\n\n")
                reason = ""
            else:
                og.log.error(f"\n\nSampling failed: {activity}.\n\nFeedback: {env.task.feedback}\n\n")
                reason = env.task.feedback

            # Write to google sheets
            cell_list = worksheet.range(f"B{row}:G{row}")
            for cell, val in zip(cell_list,
                                 ("", int(success), 0, args.scene_model, USER, "" if reason is None else reason)):
                cell.value = val
            worksheet.update_cells(cell_list)

            # Clear task callbacks if sampled
            if should_sample:
                callback_name = f"{activity}_refresh"
                og.sim.remove_callback_on_import_obj(name=callback_name)
                og.sim.remove_callback_on_remove_obj(name=callback_name)
                remove_callback_on_system_init(name=callback_name)
                remove_callback_on_system_clear(name=callback_name)

                # Remove all the additionally added objects
                for obj in env.scene.objects[n_scene_objects:]:
                    og.sim.remove_object(obj)

            # Clear all systems
            clear_all_systems()
            clear_pu()

            og.sim.step()
            og.sim.play()
            # This will clear out the previous attachment group in macro particle systems
            og.sim.scene.load_state(scene_initial_state)
            og.sim.step()
            og.sim.scene.update_initial_state()
            og.sim.stop()

        except Exception as e:
            og.log.error(traceback.format_exc())
            og.log.error(f"\n\nCaught exception sampling activity {activity} in scene {args.scene_model}:\n\n{e}\n\n")

            # Clear the in_progress reservation and note the exception
            worksheet.update_acell(f"B{row}", "")
            worksheet.update_acell(f"H{row}", f"{e}")

            try:
                # Stop sim, clear simulator, and re-create environment
                e_str = f"{e}"
                og.sim.stop()
                og.sim.clear()
            except AttributeError as e:
                # This is the "GetPath" error that happens sporatically. It's benign, so we ignore it
                pass

            env = og.Environment(configs=copy.deepcopy(cfg))

            # Store the initial state -- this is the safeguard to reset to!
            scene_initial_state = copy.deepcopy(env.scene._initial_state)
            og.sim.stop()

            n_scene_objects = len(env.scene.objects)

            # Set environment configuration after environment is loaded, because we will load the task
            env.task_config["type"] = "BehaviorTask"
            env.task_config["online_object_sampling"] = True

    # Shutdown at the end
    og.shutdown()


if __name__ == "__main__":
    main()
