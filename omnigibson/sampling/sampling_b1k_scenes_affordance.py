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

macros.systems.micro_particle_system.MICRO_PARTICLE_SYSTEM_MAX_VELOCITY = 0.5
# Disable sleeping
macros.prims.entity_prim.DEFAULT_SLEEP_THRESHOLD = 0.0

activity_scene_data = {
    "affordance_pouring": {
        "Rs_int": {
            "agent.n.01_1": ([-0.5, 2.6, 0], T.euler2quat([0, 0, -np.pi / 2])),
            "countertop.n.01_1": "countertop_tpuwys_0",
            "floor.n.01_1": "floors_ifmioj_0",
        }
    }
}


def create_stable_scene_json(scene_model):
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
        },
    }
    env = og.Environment(configs=copy.deepcopy(cfg))

    # Take a few steps to let objects settle, then update the scene initial state
    # This is to prevent nonzero velocities from causing objects to fall through the floor when we disable them
    # if they're not relevant for a given task
    for _ in range(300):
        og.sim.step(render=False)

    # Sanity check for zero velocities for all objects
    stable_state = og.sim.dump_state()
    for obj in env.scene.objects:
        obj.keep_still()
    env.scene.update_initial_state()

    # Save this as a stable file
    path = os.path.join(gm.DATASET_PATH, "scenes", og.sim.scene.scene_model, "json", f"{scene_model}_stable.json")
    og.sim.save(json_path=path)

    og.sim.stop()
    og.sim.clear()


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


def main(random_selection=False, headless=False, short_exec=False):
    args = parser.parse_args()

    # If we want to create a stable scene config, do that now
    default_scene_fpath = f"{gm.DATASET_PATH}/scenes/{args.scene_model}/json/{args.scene_model}_stable.json"
    if not os.path.exists(default_scene_fpath):
        create_stable_scene_json(scene_model=args.scene_model)

    # Get the default scene instance
    assert os.path.exists(default_scene_fpath), "Did not find default stable scene json!"
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

    activities = args.activities.split(",")

    env = og.Environment(configs=copy.deepcopy(cfg))
    if gm.HEADLESS:
        hide_all_lights()

    # After we load the robot, we do self.scene.reset() (one physics step) and then self.scene.update_initial_state().
    # We need to set all velocities to zero after this. Otherwise, the visual only objects will drift.
    for obj in og.sim.scene.objects:
        obj.keep_still()
    og.sim.scene.update_initial_state()

    # Store the initial state -- this is the safeguard to reset to!
    scene_initial_state = copy.deepcopy(env.scene._initial_state)
    og.sim.stop()

    n_scene_objects = len(env.scene.objects)

    # Set environment configuration after environment is loaded, because we will load the task
    env.task_config["type"] = "BehaviorTask"
    env.task_config["online_object_sampling"] = True
    env.task_config["skip_sampling"] = True

    for activity in activities:
        print(f"Checking activity: {activity}...")
        env.task_config["activity_name"] = activity
        # Make sure sim is stopped
        assert og.sim.is_stopped()

        conditions = Conditions(activity, 0, simulator_name="omnigibson")
        relevant_rooms = set(get_rooms(conditions.parsed_initial_conditions))
        # print(f"relevant rooms: {relevant_rooms}")
        for obj in og.sim.scene.objects:
            if isinstance(obj, DatasetObject):
                obj_rooms = {"_".join(room.split("_")[:-1]) for room in obj.in_rooms}
                active = len(relevant_rooms.intersection(obj_rooms)) > 0 or obj.category in {"floors", "walls"}
                obj.visual_only = not active
                obj.visible = active

        og.log.info(f"Sampling task: {activity}")

        for obj_rand_idx in range(args.n_obj_rand):
            env._load_task()
            assert og.sim.is_stopped()
            success, feedback = env.task.feedback is None, env.task.feedback
            if not success:
                print("Object importing / scene compatibility failed: ", feedback)
            else:
                # Assign object scope for scene objects
                for key in env.task.object_scope:
                    val = env.task.object_scope[key]
                    if val is None:
                        entity = og.sim.scene.object_registry(
                            "name", activity_scene_data[activity][args.scene_model][key]
                        )
                        assert entity is not None
                        env.task.object_scope[key] = BDDLEntity(bddl_inst=key, entity=entity)

                og.sim.play()

                # Set robot pose to the hardcoded pose
                env.robots[0].set_position_orientation(*activity_scene_data[activity][args.scene_model]["agent.n.01_1"])
                env.robots[0].reset()

                og.sim.step()
                scene_initial_state_with_task_objects = og.sim.dump_state()

                # from IPython import embed; print("sampling success"); embed()

                for pose_rand_idx in range(args.n_pose_rand):
                    env.task.activity_instance_id = obj_rand_idx * args.n_pose_rand + pose_rand_idx

                    error_msg = env.task.sampler._sample_initial_conditions_final()
                    assert error_msg is None, f"Error in sampling initial conditions: {error_msg}"

                    for i in range(10):
                        og.sim.step(render=not gm.HEADLESS)

                    # Remove any particles that fell out of the world
                    for system_cls in (PhysicalParticleSystem, VisualParticleSystem):
                        for system in system_cls.get_active_systems().values():
                            if system.n_particles > 0:
                                particle_positions, _ = system.get_particles_position_orientation()
                                remove_idxs = np.where(particle_positions[:, -1] < -1.0)[0]
                                if len(remove_idxs) > 0:
                                    system.remove_particles(remove_idxs)

                    og.sim.step(render=not gm.HEADLESS)

                    og.sim.scene.update_initial_state()
                    env.task.save_task(override=True)
                    og.log.info(f"\n\nSampling success: {activity}\n\n")

                    og.sim.scene.load_state(scene_initial_state_with_task_objects)
                    og.sim.step(render=not gm.HEADLESS)

            # Reset the scene to the initial state so that the next set of random object instances can be sampled
            og.sim.stop()
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
            og.sim.step(not gm.HEADLESS)

            # Update the scene initial state to the original state
            og.sim.scene.update_initial_state(scene_initial_state)

    print("Successful shutdown!")


if __name__ == "__main__":
    main()

    # Shutdown at the end
    og.shutdown()
