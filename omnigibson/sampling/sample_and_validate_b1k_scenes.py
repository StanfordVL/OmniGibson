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
from omnigibson.systems import remove_callback_on_system_init, remove_callback_on_system_clear, get_system, MicroPhysicalParticleSystem
from omnigibson.systems.system_base import clear_all_systems, PhysicalParticleSystem
from omnigibson.utils.python_utils import clear as clear_pu
from omnigibson.utils.python_utils import create_object_from_init_info
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
parser.add_argument("--create_stable_scene", action="store_true",
                    help="If set, will create a stable scene json to compare against and terminate early. Will not validate tasks.")

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

def create_stable_scene_json(args):
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene_model,
        },
    }

    # Create the environment
    env = og.Environment(configs=copy.deepcopy(cfg))

    # Take a few steps to let objects settle, then update the scene initial state
    # This is to prevent nonzero velocities from causing objects to fall through the floor when we disable them
    # if they're not relevant for a given task
    for _ in range(10):
        og.sim.step()
    for obj in env.scene.objects:
        obj.keep_still()
    env.scene.update_initial_state()

    # Save this as a stable file
    path = os.path.join(gm.DATASET_PATH, "scenes", og.sim.scene.scene_model, "json", f"{args.scene_model}_stable.json")
    og.sim.save(json_path=path)


def validate_task(env, activity, default_scene_dict, task_scene_dict):
    assert og.sim.is_stopped()
    loaded_scene = False

    # Get task info
    env.task_config["activity_name"] = activity
    env.task_config["online_object_sampling"] = False

    conditions = Conditions(activity, 0, simulator_name="omnigibson")
    relevant_rooms = set(get_rooms(conditions.parsed_initial_conditions))
    active_obj_names = set()
    for obj in og.sim.scene.objects:
        if isinstance(obj, DatasetObject):
            obj_rooms = {"_".join(room.split("_")[:-1]) for room in obj.in_rooms}
            active = len(relevant_rooms.intersection(obj_rooms)) > 0 or obj.category in {"floors", "walls"}
            obj.visual_only = not active
            obj.visible = active
            if active:
                active_obj_names.add(obj.name)

    # 1. Validate the native USDs jsons are stable and similar -- compare all object kinematics (poses, joint
    #       states) with respect to the native scene file
    print(f"Step 1: Checking USDs...")
    def _validate_identical_object_kinematic_state(obj_name, default_obj_dict, obj_dict, check_vel=True):
        # Check root link state
        for key, val in default_obj_dict["root_link"].items():
            # Skip velocities if requested
            if not check_vel and "vel" in key:
                continue
            obj_val = obj_dict["root_link"][key]
            # # TODO: Update ori value to be larger tolerance
            # tol = 0.15 if "ori" in key else 0.05
            if not np.all(np.isclose(np.array(val), np.array(obj_val), atol=0.05)):
                return False, f"{obj_name} root link mismatch in {key}: default scene has: {val}, task scene has: {obj_val}"

        # Check any non-robot joint values
        # This is because the controller can cause the robot to drift over time
        if "robot" not in obj_name:
            # Check joint states
            for jnt_name, jnt_info in default_obj_dict["joints"].items():
                for key, val in jnt_info.items():
                    if "effort" in key:
                        # Don't check effort
                        continue
                    obj_val = obj_dict["joints"][jnt_name][key]
                    if not np.all(np.isclose(np.array(val), np.array(obj_val), atol=0.05)):
                        return False, f"{obj_name} joint mismatch in {jnt_name} - {key}: default scene has: {val}, task scene has: {obj_val}"

        # If all passes, return True
        return True, None

    def _validate_object_state_stability(obj_name, obj_dict):
        # Check close to zero root link velocity
        for key, atol in zip(("lin_vel", "ang_vel"), (0.01, 0.05)):
            val = obj_dict["root_link"][key]
            if not np.all(np.isclose(np.array(val), 0.0, atol=atol)):
                return False, f"{obj_name} root link {key} is not close to 0: {val}"

        # Check close to zero joint velocities
        for jnt_name, jnt_info in obj_dict["joints"].items():
            val = jnt_info["vel"]
            if not np.all(np.isclose(np.array(val), 0.0, atol=0.1)):
                return False, f"{obj_name} joint {jnt_name}'s velocity is not close to 0: {val}"

        # If all passes, return True
        return True, None

    # Sanity check all object poses
    for obj_name, default_obj_info in default_scene_dict["state"]["object_registry"].items():
        # Skip any active objects since they may have changed
        if obj_name in active_obj_names:
            continue
        obj_info = task_scene_dict["state"]["object_registry"][obj_name]
        valid_obj, err_msg = _validate_identical_object_kinematic_state(obj_name, default_obj_info, obj_info, check_vel=True)
        if not valid_obj:
            return False, loaded_scene, f"Failed validation step 1: USDs do not have similar kinematic states. Specific error: {err_msg}"

    # Sanity check for zero velocities for all objects
    for obj_name, obj_info in task_scene_dict["state"]["object_registry"].items():
        obj_info = task_scene_dict["state"]["object_registry"][obj_name]
        valid_obj, err_msg = _validate_object_state_stability(obj_name, obj_info)
        if not valid_obj:
            return False, loaded_scene, f"Failed validation step 1: task USD does not have close to zero velocities. Specific error: {err_msg}"


    # 2. Validate loading the task is stable -- all object positions and velocities should be somewhat stable
    #       after a physics timestep occurs, and should be relatively consistent with the original pre-loaded state
    print(f"Step 2: Checking loaded task environment...")
    def _validate_object_stability(obj):
        # Check close to zero root link velocities
        for key, val, atol in zip(("lin_vel", "ang_vel"), (obj.get_linear_velocity(), obj.get_angular_velocity()), (0.01, 0.05)):
            if not np.all(np.isclose(val, 0.0, atol=atol)):
                return False, f"{obj.name} root link {key} is not close to 0: {val}"

        # Check close to zero joint velocities if articulated
        if obj.n_joints > 0:
            jnt_vels = obj.get_joint_velocities()
            if not np.all(np.isclose(jnt_vels, 0.0, atol=0.1)):
                return False, f"{obj.name} joint velocities are not close to 0: {jnt_vels}"

        return True, None

    # Manually import task-relevant objects
    init_info = task_scene_dict["objects_info"]["init_info"]
    init_state = task_scene_dict["state"]["object_registry"]
    init_systems = task_scene_dict["state"]["system_registry"].keys()
    task_metadata = {}
    try:
        task_metadata = task_scene_dict["metadata"]["task"]
    except:
        pass

    # Create desired systems
    for system_name in init_systems:
        get_system(system_name)

    # Iterate over all scene info, and instantiate object classes linked to the objects found on the stage
    # accordingly
    for obj_name, obj_info in init_info.items():
        # Check whether we should load the object or not
        if not env.scene._should_load_object(obj_info=obj_info, task_metadata=task_metadata):
            continue
        if env.scene.object_registry("name", obj_name) is not None:
            # Object already exists, so skip
            continue
        # Create object class instance
        obj = create_object_from_init_info(obj_info)
        # Import into the simulator
        og.sim.import_object(obj)
        # Set the init pose accordingly
        obj.set_position_orientation(
            position=init_state[obj_name]["root_link"]["pos"],
            orientation=init_state[obj_name]["root_link"]["ori"],
        )

    for key, data in task_scene_dict.get("metadata", dict()).items():
        og.sim.write_metadata(key=key, data=data)

    # Load state
    with og.sim.playing():
        og.sim.load_state(task_scene_dict["state"], serialized=False)
        env.scene.update_initial_state(task_scene_dict["state"])

    loaded_scene = True
    env._load_task()

    assert og.sim.is_stopped()

    og.sim.play()
    env.task.reset(env)

    # Sanity check all object poses wrt their original pre-loaded poses
    state = og.sim.dump_state(serialized=False)
    for obj_name, obj_info in task_scene_dict["state"]["object_registry"].items():
        current_obj_info = state["object_registry"][obj_name]
        valid_obj, err_msg = _validate_identical_object_kinematic_state(obj_name, obj_info, current_obj_info, check_vel=False)
        if not valid_obj:
            return False, loaded_scene, f"Failed validation step 2: Task scene USD and loaded task environment do not have similar kinematic states. Specific error: {err_msg}"

    # Sanity check zero velocities for all objects
    for obj in env.scene.objects:
        valid_obj, err_msg = _validate_object_stability(obj)
        if not valid_obj:
            return False, loaded_scene, f"Failed validation step 2: Loaded task environment does not have close to zero velocities. Specific error: {err_msg}"

    # 3. Validate object set is consistent (no faulty transition rules occurring) -- we expect the number
    #       of active systems (and number of active particles) and the number of objects to be the same after
    #       taking a physics step, and also make sure init state is True
    print(f"Step 3: Checking BehaviorTask initial conditions and scene stability...")

    # Take a single physics step
    og.sim.step()

    def _validate_scene_stability(env, task_state, current_state, check_particle_positions=True):
        def _validate_particle_system_consistency(system_name, system_state, current_system_state, check_particle_positions=True):
            is_micro_physical = issubclass(get_system(system_name), MicroPhysicalParticleSystem)
            n_particles_key = "instancer_particle_counts" if is_micro_physical else "n_particles"
            if not np.all(np.isclose(system_state[n_particles_key], current_system_state[n_particles_key])):
                return False, f"Got inconsistent number of system {system_name} particles: {system_state['n_particles']} vs. {current_system_state['n_particles']}"

            # Validate that no particles went flying -- maximum ranges of positions should be roughly close
            n_particles = np.sum(system_state[n_particles_key])
            if n_particles > 0 and check_particle_positions:
                if is_micro_physical:
                    particle_positions = np.concatenate([inst_state["particle_positions"] for inst_state in system_state["particle_states"].values()], axis=0)
                    current_particle_positions = np.concatenate([inst_state["particle_positions"] for inst_state in current_system_state["particle_states"].values()], axis=0)
                else:
                    particle_positions = np.array(system_state["positions"])
                    current_particle_positions = np.array(current_system_state["positions"])
                pos_min, pos_max = np.min(particle_positions, axis=0), np.max(particle_positions, axis=0)
                curr_pos_min, curr_pos_max = np.min(current_particle_positions, axis=0), np.max(current_particle_positions, axis=0)
                for name, pos, curr_pos in zip(("min", "max"), (pos_min, pos_max), (curr_pos_min, curr_pos_max)):
                    if not np.all(np.isclose(pos, curr_pos, atol=0.05)):
                        return False, f"Got mismatch in system {system_name} particle positions range: {name} {pos} vs. {curr_pos}"

            return True, None

        # Sanity check consistent objects
        task_objects = {obj_name for obj_name in task_state["object_registry"].keys()}
        curr_objects = {obj_name for obj_name in current_state["object_registry"].keys()}
        mismatched_objs = set.union(task_objects, curr_objects) - set.intersection(task_objects, curr_objects)
        if len(mismatched_objs) > 0:
            return False, f"Got mismatch in active objects: {mismatched_objs}"

        # Sanity check consistent particle systems
        task_systems = {system_name for system_name in task_state["system_registry"].keys()}
        curr_systems = {system_name for system_name in current_state["system_registry"].keys()}
        mismatched_systems = set.union(task_systems, curr_systems) - set.intersection(task_systems, curr_systems)
        if len(mismatched_systems) > 0:
            return False, f"Got mismatch in active systems: {mismatched_systems}"

        for system_name, system_state in task_state["system_registry"].items():
            curr_system_state = current_state["system_registry"][system_name]
            valid_system, err_msg = _validate_particle_system_consistency(system_name, system_state, curr_system_state, check_particle_positions=check_particle_positions)
            if not valid_system:
                return False, f"Particle systems do not have consistent state. Specific error: {err_msg}"

        # Sanity check initial state
        valid_init_state, results = evaluate_state(prune_unevaluatable_predicates(env.task.activity_initial_conditions))
        if not valid_init_state:
            return False, f"BDDL Task init conditions were invalid. Results: {results}"

        return True, None

    # Sanity check scene
    valid_scene, err_msg = _validate_scene_stability(env=env, task_state=task_scene_dict["state"], current_state=state, check_particle_positions=True)
    if not valid_scene:
        return False, loaded_scene, f"Failed verification step 3: {err_msg}"

    # 4. Validate longer-term stability -- take N=10 timesteps, and make sure all object positions and velocities
    #       are still stable (positions don't drift too much, and velocities are close to 0), as well as verifying
    #       that all BDDL conditions are satisfied
    print(f"Step 4: Checking longer-term BehaviorTask initial conditions and scene stability...")

    # Take 10 steps
    for _ in range(10):
        og.sim.step()

    # Sanity check scene
    # Don't check particle positions since some particles may be falling
    # TODO: Tighten this constraint once we figure out a way to stably sample particles
    state = og.sim.dump_state(serialized=False)
    valid_scene, err_msg = _validate_scene_stability(env=env, task_state=task_scene_dict["state"], current_state=state, check_particle_positions=False)
    if not valid_scene:
        return False, loaded_scene, f"Failed verification step 4: {err_msg}"

    # If all steps pass, we've succeeded validation!
    og.sim.stop()

    return True, loaded_scene, None


def main(random_selection=False, headless=False, short_exec=False):
    args = parser.parse_args()

    # Make sure scene can be sampled by current user
    validate_scene_can_be_sampled(scene=args.scene_model)

    # If we want to create a stable scene config, do that now
    if args.create_stable_scene:
        return create_stable_scene_json(args=args)

    # Get the default scene instance
    default_scene_fpath = f"{gm.DATASET_PATH}/scenes/{args.scene_model}/json/{args.scene_model}_stable.json"
    assert os.path.exists(default_scene_fpath), ("Did not find default stable scene json! "
                                                 "Please run with --create_stable_scene to generate one.")
    with open(default_scene_fpath, "r") as f:
        default_scene_dict = json.load(f)

    # Define the configuration to load -- we'll use a Fetch
    cfg = {
        # Use default frequency
        "env": {
            "action_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_path": default_scene_fpath,
            "scene_model": args.scene_model,
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
    # for _ in range(10):
    #     og.sim.step()
    # for obj in env.scene.objects:
    #     obj.keep_still()
    # env.scene.update_initial_state()

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
                        if obj.prim_type != PrimType.CLOTH and Contains in obj.states and any(obj.states[Contains].get_value(system) for system in PhysicalParticleSystem.get_active_systems().values()):
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
                        env.task.feedback = reason
                if success:
                    # TODO: figure out whether we also should update in_room for newly imported objects
                    env.task.save_task(override=args.overwrite_existing)
                    og.log.info(f"\n\nSampling success: {activity}\n\n")
                    reason = ""
                else:
                    og.log.error(f"\n\nSampling failed: {activity}.\n\nFeedback: {reason}\n\n")
                    reason = env.task.feedback
                og.sim.stop()
            else:
                og.log.error(f"\n\nSampling failed: {activity}.\n\nFeedback: {reason}\n\n")

            assert og.sim.is_stopped()

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
            # og.sim.play()
            # This will clear out the previous attachment group in macro particle systems
            # og.sim.scene.load_state(scene_initial_state)
            # og.sim.step()
            # og.sim.scene.update_initial_state()
            # og.sim.stop()

            # if success:
            #     try:
            #         # Validate task
            #         with open(f"{gm.DATASET_PATH}/scenes/{args.scene_model}/json/{scene_instance}.json", "r") as f:
            #             task_scene_dict = json.load(f)
            #         validated, loaded_scene, error_msg = validate_task(
            #             env=env,
            #             activity=activity,
            #             default_scene_dict=default_scene_dict,
            #             task_scene_dict=task_scene_dict,
            #         )
            #
            #         env.task_config["online_object_sampling"] = True
            #
            #         # Make sure sim is stopped
            #         og.sim.stop()
            #
            #         # Write to google sheets
            #         cell_list = worksheet.range(f"B{row}:H{row}")
            #         for cell, val in zip(cell_list,
            #                              ("", int(success), int(validated), args.scene_model, USER,
            #                               "" if error_msg is None else error_msg, "")):
            #             cell.value = val
            #         worksheet.update_cells(cell_list)
            #
            #         # Clear task callbacks if sampled
            #         if loaded_scene:
            #             callback_name = f"{activity}_refresh"
            #             og.sim.remove_callback_on_import_obj(name=callback_name)
            #             og.sim.remove_callback_on_remove_obj(name=callback_name)
            #             remove_callback_on_system_init(name=callback_name)
            #             remove_callback_on_system_clear(name=callback_name)
            #
            #             # Remove all the additionally added objects
            #             for obj in env.scene.objects[n_scene_objects:]:
            #                 og.sim.remove_object(obj)
            #
            #         # Clear all systems
            #         clear_all_systems()
            #         clear_pu()
            #
            #         og.sim.step()
            #         # og.sim.play()
            #         # This will clear out the previous attachment group in macro particle systems
            #         # og.sim.scene.load_state(scene_initial_state)
            #         # og.sim.step()
            #         # og.sim.scene.update_initial_state()
            #         # og.sim.stop()
            #
            #         if validated:
            #             og.log.info(f"\n\nValidation success: {activity}\n\n")
            #             reason = ""
            #         else:
            #             og.log.error(f"\n\nValidation failed: {activity}.\n\nFeedback: {error_msg}\n\n")
            #             reason = error_msg
            #
            #     except Exception as e:
            #         traceback_str = f"{traceback.format_exc()}"
            #         og.log.error(traceback_str)
            #         og.log.error(f"\n\nCaught exception validating activity {activity} in scene {args.scene_model}:\n\n{e}\n\n")
            #
            #         # Clear the in_progress reservation and note the exception
            #         cell_list = worksheet.range(f"B{row}:H{row}")
            #         for cell, val in zip(cell_list,
            #                              ("", int(success), 0, args.scene_model, USER, "" if reason is None else reason, traceback_str)):
            #             cell.value = val
            #         worksheet.update_cells(cell_list)
            #
            #         env.task_config["online_object_sampling"] = True

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
