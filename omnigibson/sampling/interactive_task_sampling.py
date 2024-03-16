import logging
import os
import yaml
import copy
import time
import argparse
import bddl
from bddl.activity import get_object_scope
import pkgutil
import omnigibson as og
from omnigibson.macros import gm, macros
import json
import csv
import traceback
from omnigibson.objects import DatasetObject
from omnigibson.object_states import Contains
from omnigibson.tasks import BehaviorTask
from omnigibson.scenes.scene_base import BOUNDING_CUBE_OBJECTS
from omnigibson.systems import remove_callback_on_system_init, remove_callback_on_system_clear, get_system, MicroPhysicalParticleSystem
from omnigibson.systems.system_base import clear_all_systems, PhysicalParticleSystem, VisualParticleSystem
from omnigibson.utils.python_utils import clear as clear_pu
from omnigibson.tasks import REGISTERED_TASKS
from omnigibson.utils.python_utils import create_object_from_init_info, create_class_from_registry_and_config
from omnigibson.utils.bddl_utils import *
from omnigibson.utils.constants import PrimType
from omnigibson.utils.asset_utils import get_all_object_category_models_with_abilities
from bddl.activity import Conditions, evaluate_state
from utils import *
import numpy as np
import random
from pathlib import Path
import json
# from omnigibson.sampling.utils import *
from omnigibson.object_states import *
from omnigibson.systems import get_system
from omnigibson.object_states import *


gm.HEADLESS = False
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

macros.systems.micro_particle_system.MICRO_PARTICLE_SYSTEM_MAX_VELOCITY = 0.5


class InteractiveSampler:
    def __init__(self, scene_model):
        self.scene_model = scene_model
        self.valid_tasks = get_valid_tasks()
        self.mapping = parse_task_mapping(fpath=TASK_INFO_FPATH)
        self.valid_activities = set(get_scene_compatible_activities(scene_model=scene_model, mapping=self.mapping))
        self.current_activity = None
        self.object_scope = None
        self.default_scene_fpath = f"{gm.DATASET_PATH}/scenes/{scene_model}/json/{scene_model}_stable.json"
        assert os.path.exists(self.default_scene_fpath), \
            "Stable scene file does not exist! Grab from /cvgl/group/Gibson/og-data-1-0-0/og_dataset/scenes/X/json/X_stable.json"

        with open(self.default_scene_fpath, "r") as f:
            self.default_scene_dict = json.load(f)

        # Define configuration to load
        cfg = {
            # Use default frequency
            "env": {
                "action_frequency": 30,
                "physics_frequency": 120,
            },
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_file": self.default_scene_fpath,
                "scene_model": scene_model,
            },
            "robots": [
                {
                    "type": "Fetch",
                    "obs_modalities": ["rgb"],
                    "grasping_mode": "physical",
                    "default_arm_pose": "diagonal30",
                    "default_reset_mode": "tuck",
                    "position": np.ones(3) * -50.0,
                },
            ],
        }
        self.env = og.Environment(cfg)
        og.sim.enable_viewer_camera_teleoperation()
        # After we load the robot, we do self.scene.reset() (one physics step) and then self.scene.update_initial_state().
        # We need to set all velocities to zero after this. Otherwise, the visual only objects will drift.
        for obj in og.sim.scene.objects:
            obj.keep_still()
        og.sim.scene.update_initial_state()

        # Store the initial state -- this is the safeguard to reset to!
        self.scene_initial_state = copy.deepcopy(self.env.scene._initial_state)
        og.sim.stop()

        self.n_scene_objects = len(self.env.scene.objects)
        self._activity_conditions = None
        self._object_instance_to_synset = None
        self._room_type_to_object_instance = None
        self._inroom_object_instances = None
        self._backend = OmniGibsonBDDLBackend()
        self._substance_instances = None

    def _build_sampling_order(self):
        """
        Sampling orders is a list of lists: [[batch_1_inst_1, ... batch_1_inst_N], [batch_2_inst_1, batch_2_inst_M], ...]
        Sampling should happen for batch 1 first, then batch 2, so on and so forth
        Example: OnTop(plate, table) should belong to batch 1, and OnTop(apple, plate) should belong to batch 2
        """
        unsampleable_conditions = []
        sampling_groups = {group: [] for group in ("kinematic", "particle", "unary")}
        self._object_sampling_conditions = {group: [] for group in ("kinematic", "particle", "unary")}
        self._object_sampling_orders = {group: [] for group in ("kinematic", "particle", "unary")}
        self._inroom_object_conditions = []

        # First, sort initial conditions into kinematic, particle and unary groups
        # bddl.condition_evaluation.HEAD, each with one child.
        # This child is either a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate or
        # a Negation of a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate
        for condition in get_initial_conditions(self._activity_conditions, self._backend, self.object_scope):
            condition, positive = process_single_condition(condition)
            if condition is None:
                continue

            # Sampled conditions must always be positive
            # Non-positive (e.g.: NOT onTop) is not restrictive enough for sampling
            if condition.STATE_NAME in KINEMATIC_STATES_BDDL and not positive:
                return "Initial condition has negative kinematic conditions: {}".format(condition.body)

            # Store any unsampleable conditions separately
            if isinstance(condition, UnsampleablePredicate):
                unsampleable_conditions.append(condition)
                continue

            # Infer the group the condition and its object instances belong to
            # (a) Kinematic (binary) conditions, where (ent0, ent1) are both objects
            # (b) Particle (binary) conditions, where (ent0, ent1) are (object, substance)
            # (d) Unary conditions, where (ent0,) is an object
            # Binary conditions have length 2: (ent0, ent1)
            if len(condition.body) == 2:
                group = "particle" if condition.body[1] in self._substance_instances else "kinematic"
            else:
                assert len(condition.body) == 1, \
                    f"Got invalid parsed initial condition; body length should either be 2 or 1. " \
                    f"Got body: {condition.body} for condition: {condition}"
                group = "unary"
            sampling_groups[group].append(condition.body)
            self._object_sampling_conditions[group].append((condition, positive))

            # If the condition involves any non-sampleable object (e.g.: furniture), it's a non-sampleable condition
            # This means that there's no ordering constraint in terms of sampling, because we know the, e.g., furniture
            # object already exists in the scene and is placed, so these specific conditions can be sampled without
            # any dependencies
            if len(self._inroom_object_instances.intersection(set(condition.body))) > 0:
                self._inroom_object_conditions.append((condition, positive))

        # Now, sort each group, ignoring the futures (since they don't get sampled)
        # First handle kinematics, then particles, then unary

        # Start with the non-sampleable objects as the first sampled set, then infer recursively
        cur_batch = self._inroom_object_instances
        while len(cur_batch) > 0:
            next_batch = set()
            for cur_batch_inst in cur_batch:
                inst_batch = set()
                for condition, _ in self._object_sampling_conditions["kinematic"]:
                    if condition.body[1] == cur_batch_inst:
                        inst_batch.add(condition.body[0])
                        next_batch.add(condition.body[0])
                if len(inst_batch) > 0:
                    self._object_sampling_orders["kinematic"].append(inst_batch)
            cur_batch = next_batch

        # Now parse particles -- simply unordered, since particle systems shouldn't impact each other
        self._object_sampling_orders["particle"].append({cond[0] for cond in sampling_groups["particle"]})
        sampled_particle_entities = {cond[1] for cond in sampling_groups["particle"]}

        # Finally, parse unaries -- this is simply unordered, since it is assumed that unary predicates do not
        # affect each other
        self._object_sampling_orders["unary"].append({cond[0] for cond in sampling_groups["unary"]})

        # Aggregate future objects and any unsampleable obj instances
        # Unsampleable obj instances are strictly a superset of future obj instances
        unsampleable_obj_instances = {cond.body[-1] for cond in unsampleable_conditions}
        self._future_obj_instances = {cond.body[0] for cond in unsampleable_conditions if isinstance(cond, ObjectStateFuturePredicate)}

        nonparticle_entities = set(self.object_scope.keys()) - self._substance_instances

        # Sanity check kinematic objects -- any non-system must be kinematically sampled
        remaining_kinematic_entities = nonparticle_entities - unsampleable_obj_instances - \
            self._inroom_object_instances - set.union(*(self._object_sampling_orders["kinematic"] + [set()]))

        # Possibly remove the agent entity if we're in an empty scene -- i.e.: no kinematic sampling needed for the
        # agent
        if self.scene_model is None:
            remaining_kinematic_entities -= {"agent.n.01_1"}

        if len(remaining_kinematic_entities) != 0:
            return f"Some objects do not have any kinematic condition defined for them in the initial conditions: " \
                   f"{', '.join(remaining_kinematic_entities)}"

        # Sanity check particle systems -- any non-future system must be sampled as part of particle groups
        remaining_particle_entities = self._substance_instances - unsampleable_obj_instances - sampled_particle_entities
        if len(remaining_particle_entities) != 0:
            return f"Some systems do not have any particle condition defined for them in the initial conditions: " \
                   f"{', '.join(remaining_particle_entities)}"


    def _parse_inroom_object_room_assignment(self):
        """
        Infers which rooms each object is assigned to
        """
        self._room_type_to_object_instance = dict()
        self._inroom_object_instances = set()
        for cond in self._activity_conditions.parsed_initial_conditions:
            if cond[0] == "inroom":
                obj_inst, room_type = cond[1], cond[2]
                obj_synset = self._object_instance_to_synset[obj_inst]
                abilities = OBJECT_TAXONOMY.get_abilities(obj_synset)
                if "sceneObject" not in abilities:
                    # Invalid room assignment
                    return f"You have assigned room type for [{obj_synset}], but [{obj_synset}] is sampleable. " \
                           f"Only non-sampleable (scene) objects can have room assignment."
                if self.scene_model is not None and room_type not in og.sim.scene.seg_map.room_sem_name_to_ins_name:
                    # Missing room type
                    return f"Room type [{room_type}] missing in scene [{self.scene_model}]."
                if room_type not in self._room_type_to_object_instance:
                    self._room_type_to_object_instance[room_type] = []
                self._room_type_to_object_instance[room_type].append(obj_inst)

                if obj_inst in self._inroom_object_instances:
                    # Duplicate room assignment
                    return f"Object [{obj_inst}] has more than one room assignment"

                self._inroom_object_instances.add(obj_inst)

    def initialize_activity_info(self, activity):
        self.current_activity = activity
        self._activity_conditions = Conditions(
            self.current_activity,
            0,
            simulator_name="omnigibson",
            predefined_problem=None,
        )

        # Get scope, making sure agent is the first entry
        self.object_scope = {"agent.n.01_1": None}
        self.object_scope.update(get_object_scope(self._activity_conditions))

        self._object_instance_to_synset = {
            obj_inst: obj_cat
            for obj_cat in self._activity_conditions.parsed_objects
            for obj_inst in self._activity_conditions.parsed_objects[obj_cat]
        }

        self._substance_instances = {obj_inst for obj_inst in self.object_scope.keys() if
                                     is_substance_synset(self._object_instance_to_synset[obj_inst])}

        self._parse_inroom_object_room_assignment()
        self._build_sampling_order()

    def get_obj(self, name):
        return self.env.scene.object_registry("name", name)

    def get_system(self, name):
        return get_system(name)

    def get_task_entity(self, synset_instance):
        return self.object_scope[synset_instance]

    def save_checkpoint(self):
        self.write_task_metadata()
        og.sim.save(json_path=os.path.join(os.path.dirname(self.default_scene_fpath), f"{self.scene_model}_{self.current_activity}_sampling_checkpoint.json"))

    def load_checkpoint(self):
        # Clear current stage
        self.clear()

        assert og.sim.is_stopped()

        fpath = os.path.join(os.path.dirname(self.default_scene_fpath), f"{self.scene_model}_{self.current_activity}_sampling_checkpoint.json")
        with open(fpath, "r") as f:
            scene_info = json.load(f)
        init_info = scene_info["objects_info"]["init_info"]
        init_state = scene_info["state"]["object_registry"]
        init_systems = scene_info["state"]["system_registry"].keys()

        # Write the metadata
        for key, data in scene_info.get("metadata", dict()).items():
            og.sim.write_metadata(key=key, data=data)

        # Create desired systems
        for system_name in init_systems:
            get_system(system_name)

        # Iterate over all scene info, and instantiate object classes linked to the objects found on the stage
        # accordingly
        for i, (obj_name, obj_info) in enumerate(init_info.items()):
            if i < self.n_scene_objects:
                continue

            # Create object class instance
            obj = create_object_from_init_info(obj_info)
            # Import into the simulator
            og.sim.import_object(obj)
            if isinstance(obj, DatasetObject) and obj.model in BOUNDING_CUBE_OBJECTS:
                link_names = BOUNDING_CUBE_OBJECTS[obj.model]
                for link_name in link_names:
                    link = obj.links[link_name]
                    for col_mesh in link.collision_meshes.values():
                        col_mesh.set_collision_approximation("boundingCube")
            # Set the init pose accordingly
            obj.set_position_orientation(
                position=init_state[obj_name]["root_link"]["pos"],
                orientation=init_state[obj_name]["root_link"]["ori"],
            )

        # Play, then update the initial state
        og.sim.play()
        self.env.scene.update_objects_info()
        self.env.scene.wake_scene_objects()
        og.sim.load_state(init_state, serialized=False)
        self.env.scene.update_initial_state(init_state)

    def write_task_metadata(self):
        # Make sure appropriate (expected) data is written to sim metadata
        # Store mapping from entity name to its corresponding BDDL instance name
        metadata = dict(
            inst_to_name={inst: entity.name for inst, entity in self.object_scope.items()},
        )
        # Write to sim
        og.sim.write_metadata(key="task", data=metadata)

    def set_task_entity(self, synset_instance, entity):
        # Sanity check to make sure category and entity are compatible
        synset = "_".join(synset_instance.split("_")[:-1])
        if isinstance(entity, DatasetObject):
            # Object
            if OBJECT_TAXONOMY.is_leaf(synset):
                categories = OBJECT_TAXONOMY.get_categories(synset)
            else:
                leafs = OBJECT_TAXONOMY.get_leaf_descendants(synset)
                categories = []
                for leaf in leafs:
                    categories += OBJECT_TAXONOMY.get_categories(leaf)
            assert entity.category in set(categories)

        else:
            # System
            assert synset == OBJECT_TAXONOMY.get_synset_from_substance(entity.name)
        self.object_scope[synset_instance] = entity

    def import_obj(self, category, model=None, synset_instance=None):
        synset = OBJECT_TAXONOMY.get_synset_from_category(category)
        abilities = OBJECT_TAXONOMY.get_abilities(synset)
        model_choices = set(get_all_object_category_models_with_abilities(category, abilities))
        model_choices = model_choices if category not in GOOD_MODELS else model_choices.intersection(GOOD_MODELS[category])
        model_choices -= BAD_MODELS.get(category, set())
        # model_choices = self._filter_model_choices_by_attached_states(model_choices, category, obj_inst)
        if model is not None:
            assert model in set(model_choices)
        else:
            model = np.random.choice(list(model_choices))

        # Potentially add additional kwargs
        obj_kwargs = dict()

        obj_kwargs["bounding_box"] = GOOD_BBOXES.get(category, dict()).get(model, None)

        name = f"{category}_{len(og.sim.scene.objects)}"
        print(f"Importing {name}...")
        obj = DatasetObject(
            name=name,
            category=category,
            model=model,
            prim_type=PrimType.CLOTH if "cloth" in OBJECT_TAXONOMY.get_abilities(synset) else PrimType.RIGID,
            **obj_kwargs,
        )
        og.sim.import_object(obj)
        if og.sim.is_playing():
            obj.set_position(np.ones(3) * len(og.sim.scene.objects) * 2)

        if synset_instance is not None:
            self.object_scope[synset_instance] = obj

        return obj

    def save(self, task_final_state):
        og.sim.load_state(task_final_state)
        og.sim.step()
        self.env.task.save_task(override=True)

    def validate(self):
        assert og.sim.is_playing()

        # Update the Behavior Task
        self.env.task_config["type"] = "BehaviorTask"
        self.env.task_config["online_object_sampling"] = False
        self.env.task_config["activity_name"] = self.current_activity

        self.write_task_metadata()

        # Load Behavior Task
        # NOTE: We abuse functionality here, and EXPECT sim to be playing
        task = create_class_from_registry_and_config(
            cls_name="BehaviorTask",
            cls_registry=REGISTERED_TASKS,
            cfg=self.env.task_config,
            cls_type_descriptor="task",
        )
        self.env._task = task
        assert og.sim.is_playing()
        task.load(env=self.env)

        # Validate current configuration
        task_final_state = og.sim.dump_state()
        task_scene_dict = {"state": task_final_state}

        validated, error_msg = validate_task(self.env.task, task_scene_dict, self.default_scene_dict)
        if not validated:
            print(error_msg)

        if validated:
            self.save(task_final_state)
            print("Success! Saving sampled task configuration...")

        return validated

    def clear(self):
        og.sim.stop()
        if self.current_activity is not None:
            callback_name = f"{self.current_activity}_refresh"
            if callback_name in og.sim._callbacks_on_import_obj:
                og.sim.remove_callback_on_import_obj(name=callback_name)
                og.sim.remove_callback_on_remove_obj(name=callback_name)
                remove_callback_on_system_init(name=callback_name)
                remove_callback_on_system_clear(name=callback_name)

        # Remove all the additionally added objects
        for obj in self.env.scene.objects[self.n_scene_objects:]:
            og.sim.remove_object(obj)

        # Clear all systems
        clear_all_systems()
        clear_pu()
        og.sim.step()

        # Update the scene initial state to the original state
        og.sim.scene.update_initial_state(self.scene_initial_state)

        # Clear task scope
        self.object_scope = {"agent.n.01_1": self.env.robots[0]}

    def set_activity(self, activity):
        self.clear()

        # Set current activity
        self.initialize_activity_info(activity)

    def play(self):
        # Synchronize all scales
        for obj in self.env.scene.objects[self.n_scene_objects:]:
            obj.scale = obj.scale

        og.sim.play()
        og.sim.scene.reset()

    def stop(self):
        og.sim.stop()

    def apply_in_rooms(self, source_obj, objs=None):
        if objs is None:
            objs = self.env.scene.objects[self.n_scene_objects:]
        elif isinstance(objs, DatasetObject):
            objs = [objs]

        for obj in objs:
            obj.in_rooms = copy.deepcopy(source_obj.in_rooms)

        og.sim.scene.object_registry.update(keys=["in_rooms"])

    def update_initial_state(self):
        self.env.scene.update_initial_state()

    def import_sampleable_objects(self):
        available_categories = set(get_all_object_categories())
        for obj_inst, obj_synset in self._object_instance_to_synset.items():

            # Don't populate agent
            if obj_synset == "agent.n.01":
                continue

            # Populate based on whether it's a substance or not
            if is_substance_synset(obj_synset):
                assert len(self._activity_conditions.parsed_objects[obj_synset]) == 1, "Systems are singletons"
                obj_inst = self._activity_conditions.parsed_objects[obj_synset][0]
                system_name = OBJECT_TAXONOMY.get_subtree_substances(obj_synset)[0]
                self.object_scope[obj_inst] = get_system(system_name)
            else:
                valid_categories = set(OBJECT_TAXONOMY.get_subtree_categories(obj_synset))
                categories = list(valid_categories.intersection(available_categories))
                if len(categories) == 0:
                    return f"None of the following categories could be found in the dataset for synset {obj_synset}: " \
                           f"{valid_categories}"

                # Don't explicitly sample if future
                if obj_inst in self._future_obj_instances:
                    continue
                # Don't sample if already in room
                if obj_inst in self._inroom_object_instances:
                    continue

                # Shuffle categories and sample to find a valid model
                np.random.shuffle(categories)
                model_choices = set()
                for category in categories:
                    # Get all available models that support all of its synset abilities
                    model_choices = set(get_all_object_category_models_with_abilities(
                        category=category,
                        abilities=OBJECT_TAXONOMY.get_abilities(OBJECT_TAXONOMY.get_synset_from_category(category)),
                    ))
                    model_choices = model_choices if category not in GOOD_MODELS else model_choices.intersection(GOOD_MODELS[category])
                    model_choices -= BAD_MODELS.get(category, set())
                    # model_choices = self._filter_model_choices_by_attached_states(model_choices, category, obj_inst)
                    if len(model_choices) > 0:
                        break

                if len(model_choices) == 0:
                    # We failed to find ANY valid model across ALL valid categories
                    return f"Missing valid object models for all categories: {categories}"

                # Randomly select an object model
                model = np.random.choice(list(model_choices))

                self.import_obj(category, model, obj_inst)

        self.play()
        self.stop()

    def pick_floor_and_move_objects_to_valid_room(self, i=0, x_offset=0, y_offset=0, z_offset=1.5):
        target_room = None
        for room_type, obj_insts in self._room_type_to_object_instance.items():
            if "floor.n.01_1" in set(obj_insts):
                target_room = room_type
                break

        assert target_room is not None

        # Find all floors with this room type
        valid_floors = []
        for floor in og.sim.scene.object_registry("category", "floors"):
            for room_inst in floor.in_rooms:
                if target_room in room_inst:
                    valid_floors.append(floor)
                    break

        assert len(valid_floors) > 0
        target_floor = valid_floors[i]

        # Move all objects to this position plus the desired offset
        floor_pos = target_floor.get_position()
        target_pos = floor_pos + np.array([x_offset, y_offset, z_offset])

        for obj in og.sim.scene.objects[self.n_scene_objects:]:
            obj.set_position(target_pos)

        # Also set viewer camera
        og.sim.viewer_camera.set_position(target_pos + np.ones(3))

        return target_floor


############################

s = InteractiveSampler(scene_model="Rs_int")


####### EXAMPLE USAGE #######

# Set an activity
s.set_activity("clean_whiskey_stones")

# Import all sampleable objects for the current activity
# TODO: Explain how to add to BAD / GOOD MODELS / GOOD BBOXES
s.import_sampleable_objects()

# Do NOT call og.sim.stop / og.sim.play(). Call the sampler's version instead
s.play()
s.stop()

# Grab object by name, or by its synset instance
stone = s.get_obj("whiskey_stone_82")
stone = s.get_task_entity("whiskey_stone.n.01")

# Grab system by name
water = s.get_system("water")

# See the current mapping from synset instance to object / system instance
print(s.object_scope)

# Some objects (such as floor.n.01 are None -- this is because it's not sampled, but is expected to pre-exist in the scene
# We need to manually select a floor and map it to the instance
# Because floor.n.01_1 is required to be in the kitchen, I click that floor in the kitchen to find its name
floor = s.get_obj("floor_ifmioj_0")
s.set_task_entity("floor.n.01_1", floor)

# You can also have the sampler automatically find a valid floor and teleport all sampled objects to that floor's
# location, offset by a desired amount
floor = s.pick_floor_and_move_objects_to_valid_room(i=0, x_offset=0, y_offset=0, z_offset=1.5)

# Set the in-room parameter for a given object, and have it infer from another object
# In this case, whiskey infers it from the floor
s.apply_in_rooms(source_obj=floor, objs=[stone])

# You can always save your current progress -- this will save the current sim state, so sim needs to be playing
s.save_checkpoint()

# Load your checkpoint at any time
s.load_checkpoint()

# Set object states as usual
whiskey = s.get_system("whiskey")
stone.states[Covered].set_value(whiskey, True)

# At any time you can validate if your current scene configuration if a valid init state
# If it's valid, it will automatically save the scene task .json and you are done!!
s.validate()

# You can clear the scene of all non-vanilla scene objects
s.clear()

# You can always import custom objects and overwrite them in the object scope
# If you don't specify model one will be randomly sampled
# If you don't specify synset instance it will not be set in the object scope
s.import_obj("stone", model=None, synset_instance="whiskey_stone.n.01_1")

# You can also update the initial state at any time so you can preserve state between stop / play cycles
s.update_initial_state()

