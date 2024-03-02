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
from omnigibson.object_states import Contains
from omnigibson.tasks import BehaviorTask
from omnigibson.systems import remove_callback_on_system_init, remove_callback_on_system_clear
from omnigibson.systems.system_base import clear_all_systems, PhysicalParticleSystem
from omnigibson.utils.python_utils import clear as clear_pu
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
from omnigibson.utils.constants import PrimType
from bddl.activity import Conditions, evaluate_state
import numpy as np
import gspread
import getpass

"""
1. gcloud auth login
2. gcloud auth application-default login
3. gcloud config set project lucid-inquiry-205018
4. gcloud iam service-accounts create cremebrule
5. gcloud iam service-accounts keys create key.json --iam-account=cremebrule@lucid-inquiry-205018.iam.gserviceaccount.com
6. mv key.json /home/cremebrule/.config/gcloud/key.json
"""

SAMPLING_SHEET_KEY = "1Vt5s3JrFZ6_iCkfzZr0eb9SBt2Pkzx3xxzb4wtjEaDI"
CREDENTIALS = "key.json"
WORKSHEET = "GTC2024 - 5a2d64"
USER = getpass.getuser()

client = gspread.service_account(filename=CREDENTIALS)
worksheet = client.open_by_key(SAMPLING_SHEET_KEY).worksheet(WORKSHEET)

ACTIVITY_TO_ROW = {activity: i + 2 for i, activity in enumerate(worksheet.col_values(1)[1:])}

SCENE_INFO_FPATH =  "BEHAVIOR-1K Scenes.csv"
TASK_INFO_FPATH = "BEHAVIOR-1K Tasks.csv"
SYNSET_INFO_FPATH = "BEHAVIOR-1K Synsets.csv"

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

gm.HEADLESS = True
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


# CAREFUL!! Only run this ONCE before starting sampling!!!
def write_scenes_to_spreadsheet():
    # Get scenes
    scenes_sorted = get_scenes()
    n_scenes = len(scenes_sorted)
    cell_list = worksheet.range(f"R{2}:R{2 + n_scenes - 1}")
    for cell, scene in zip(cell_list, scenes_sorted):
        cell.value = scene
    worksheet.update_cells(cell_list)


def validate_scene_can_be_sampled(scene):
    scenes_sorted = get_scenes()
    n_scenes = len(scenes_sorted)
    # Sanity check scene -- only scenes are allowed that whose user field is either:
    # (a) blank or (b) filled with USER
    # scene_user_list = worksheet.range(f"R{2}:S{2 + n_scenes - 1}")
    def get_user(val):
        return None if (len(val) == 1 or val[1] == "") else val[1]

    scene_user_mapping = {val[0]: get_user(val) for val in worksheet.get(f"T{2}:U{2 + n_scenes - 1}")}

    # Make sure scene is valid
    assert scene in scene_user_mapping, f"Got invalid scene name to sample: {scene}"

    # Assert user is None or is USER, else False
    scene_user = scene_user_mapping[scene]
    assert scene_user is None or scene_user == USER, \
        f"Cannot sample scene {scene} with user {USER}! Scene already has user: {scene_user}."

    # Fill in this value to reserve it
    idx = scenes_sorted.index(scene)
    worksheet.update_acell(f"U{2 + idx}", USER)


def prune_unevaluatable_predicates(init_conditions):
    pruned_conditions = []
    for condition in init_conditions:
        if condition.body[0] in {"insource", "future", "real"}:
            continue
        pruned_conditions.append(condition)

    return pruned_conditions


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


def get_scenes():
    scenes = set()
    with open(SCENE_INFO_FPATH) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for i, row in enumerate(reader):
            # Skip first row since it's the header
            if i == 0:
                continue
            scenes.add(row[0])

    return tuple(sorted(scenes))


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

    # Make sure scene can be sampled by current user
    validate_scene_can_be_sampled(scene=args.scene_model)

    # Define the configuration to load -- we'll use a Fetch
    cfg = {
        "env": {
            "action_frequency": 50,
            "physics_frequency": 100,
        },
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

    valid_tasks = get_valid_tasks()
    mapping = parse_task_mapping(fpath=TASK_INFO_FPATH)
    activities = get_scene_compatible_activities(scene_model=args.scene_model, mapping=mapping) \
        if args.activities is None else args.activities.split(",")

    # Create the environment
    # Attempt to sample the activity
    env = og.Environment(configs=copy.deepcopy(cfg))

    # Take a few steps to let objects settle, then update the scene initial state
    # This is to prevent nonzero velocities from causing objects to fall through the floor when we disable them
    # if they're not relevant for a given task
    for _ in range(10):
        og.sim.step()
    for obj in env.scene.objects:
        obj.keep_still()
    env.scene.update_initial_state()

    # Store the initial state -- this is the safeguard to reset to!
    scene_initial_state = copy.deepcopy(env.scene._initial_state)
    og.sim.stop()

    n_scene_objects = len(env.scene.objects)

    # Set environment configuration after environment is loaded, because we will load the task
    env.task_config["type"] = "BehaviorTask"
    env.task_config["online_object_sampling"] = True

    should_start = args.start_at is None
    for activity in sorted(activities):
        print(f"Checking activity: {activity}...")
        if not should_start:
            if args.start_at == activity:
                should_start = True
            else:
                continue

        # sleep to avoid gspread query limits
        time.sleep(1)

        # Don't sample any invalid activities
        if activity not in valid_tasks:
            continue

        if activity not in ACTIVITY_TO_ROW:
            continue

        # Get info from spreadsheet
        row = ACTIVITY_TO_ROW[activity]
        in_progress, success, validated, scene_id, user, reason, exception, misc = worksheet.get(f"B{row}:I{row}")[0]

        # If we manually do not want to sample the task (DO NOT SAMPLE == "DNS", skip)
        if success.lower() == "dns":
            continue

        # Only sample stuff which is fixed
        # if "fixed" not in misc.lower():
        #     continue

        # If we've already sampled successfully (success is populated with a 1) and we don't want to overwrite the
        # existing sampling result, skip
        if success != "" and int(success) == 1 and not args.overwrite_existing:
            continue

        # If another thread is already in the process of sampling, skip
        if in_progress != "" and int(in_progress) == 1:
            continue

        # Reserve this task by marking in_progress = 1
        worksheet.update_acell(f"B{row}", 1)

        should_sample, success, reason = True, False, ""

        # Skip any with unsupported predicates, but still record the reason why we can't sample
        conditions = Conditions(activity, 0, simulator_name="omnigibson")
        init_predicates = set(get_predicates(conditions.parsed_initial_conditions))
        unsupported_predicates = set.intersection(init_predicates, UNSUPPORTED_PREDICATES)
        if len(unsupported_predicates) > 0:
            should_sample = False
            reason = f"Unsupported predicate(s): {unsupported_predicates}"

        env.task_config["activity_name"] = activity
        scene_instance = BehaviorTask.get_cached_activity_scene_filename(
            scene_model=args.scene_model,
            activity_name=activity,
            activity_definition_id=0,
            activity_instance_id=0,
        )

        # Make sure sim is stopped
        assert og.sim.is_stopped()

        # Attempt to sample
        try:
            if should_sample:
                relevant_rooms = set(get_rooms(conditions.parsed_initial_conditions))
                print(f"relevant rooms: {relevant_rooms}")
                for obj in og.sim.scene.objects:
                    if isinstance(obj, DatasetObject):
                        obj_rooms = {"_".join(room.split("_")[:-1]) for room in obj.in_rooms}
                        active = len(relevant_rooms.intersection(obj_rooms)) > 0 or obj.category in {"floors", "walls"}
                        obj.visual_only = not active
                        obj.visible = active

                og.log.info(f"Sampling task: {activity}")
                env._load_task()
                assert og.sim.is_stopped()

                success, feedback = env.task.feedback is None, env.task.feedback

                if success:
                    # Set masses of all task-relevant objects to be very high
                    # This is to avoid particles from causing instabilities
                    # Don't use this on cloth since these may be unstable at high masses
                    for obj in env.scene.objects[n_scene_objects:]:
                        if obj.prim_type != PrimType.CLOTH and Contains in obj.states and any(obj.states[Contains].get_value(system) for system in PhysicalParticleSystem.get_active_systems()):
                            obj.root_link.mass = max(1.0, obj.root_link.mass)

                    # Sampling success
                    og.sim.play()
                    # This will actually reset the objects to their sample poses
                    env.task.reset(env)

                    for i in range(50):
                        og.sim.step()

                    # Make sure init conditions are still true
                    valid_init_state, results = evaluate_state(prune_unevaluatable_predicates(env.task.activity_initial_conditions))
                    if not valid_init_state:
                        success = False
                        reason = f"BDDL Task init conditions were invalid. Results: {results}"

                if success:
                    # TODO: figure out whether we also should update in_room for newly imported objects
                    env.task.save_task(override=args.overwrite_existing)

                    og.sim.stop()
                    og.log.info(f"\n\nSampling success: {activity}\n\n")
                    reason = ""
                else:
                    og.log.error(f"\n\nSampling failed: {activity}.\n\nFeedback: {reason}\n\n")
                    reason = env.task.feedback

            else:
                og.log.error(f"\n\nSampling failed: {activity}.\n\nFeedback: {reason}\n\n")

            # Write to google sheets
            cell_list = worksheet.range(f"B{row}:H{row}")
            for cell, val in zip(cell_list,
                                 ("", int(success), "", args.scene_model, USER, "" if reason is None else reason, "")):
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
            traceback_str = f"{traceback.format_exc()}"
            og.log.error(traceback_str)
            og.log.error(f"\n\nCaught exception sampling activity {activity} in scene {args.scene_model}:\n\n{e}\n\n")

            # Clear the in_progress reservation and note the exception
            cell_list = worksheet.range(f"B{row}:H{row}")
            for cell, val in zip(cell_list,
                                 ("", 0, "", args.scene_model, USER, "" if reason is None else reason, traceback_str)):
                cell.value = val
            worksheet.update_cells(cell_list)

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

    print("Successful shutdown!")
    # Shutdown at the end
    og.shutdown()


if __name__ == "__main__":
    main()
