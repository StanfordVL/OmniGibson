import random
import os
import sys

from tasknet import get_backend
from tasknet.config import SCENE_PATH
from tasknet.sampler import Sampler
from tasknet.parsing import parse_domain, parse_problem, gen_natural_language_conditions
from tasknet.condition_evaluation import create_scope, compile_state, evaluate_state, get_ground_state_options

from gibson2.external.pybullet_tools.utils import quat_from_euler

import numpy as np
from IPython import embed
from tasknet.object_taxonomy import ObjectTaxonomy


class TaskNetTask(object):
    # TODO
    #   1. Update with new object formats
    #   2. Update initialize() to work with self.check_setup()
    #   3. Update initialize() to work with sampler code
    #   4. Various other adaptations to be seen

    def __init__(self, atus_activity=None, task_instance=None, scene_path=SCENE_PATH, predefined_problem=None):
        self.scene_path = scene_path
        self.object_taxonomy = ObjectTaxonomy()
        self.update_problem(atus_activity, task_instance,
                            predefined_problem=predefined_problem)

    def update_problem(self, atus_activity, task_instance, predefined_problem=None, kinematic_only=False):
        if predefined_problem is not None:
            self.atus_activity = "predefined"
            self.task_instance = "predefined"
        else:
            self.atus_activity = atus_activity
            self.task_instance = task_instance
        domain_name, requirements, types, actions, predicates = parse_domain(
            "igibson")
        print()
        print(predefined_problem)
        print()
        problem_name, self.objects, self.parsed_initial_conditions, self.parsed_goal_conditions = parse_problem(
            self.atus_activity,
            self.task_instance,
            domain_name,
            predefined_problem=predefined_problem)
        self.object_scope = create_scope(self.objects)
        self.obj_inst_to_obj_cat = {
            obj_inst: obj_cat
            for obj_cat in self.objects
            for obj_inst in self.objects[obj_cat]
        }

        # Generate initial and goal conditions
        self.gen_initial_conditions()
        self.gen_goal_conditions()
        self.gen_ground_goal_conditions()

        # Demo attributes
        self.instruction_order = np.arange(len(self.parsed_goal_conditions))
        np.random.shuffle(self.instruction_order)
        self.currently_viewed_index = 0
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]
        self.current_success = False
        self.current_goal_status = {"satisfied": [], "unsatisfied": []}
        self.natural_language_goal_conditions = gen_natural_language_conditions(
            self.parsed_goal_conditions)

    def initialize(self, scene_class, scene_id=None, scene_kwargs=None, online_sampling=True):
        '''
        Check self.scene to see if it works for this Task. If not, resample.
        Populate self.scene with necessary objects.
        :param scene_class: scene class from simulator
        TODO should this method take scene_path and object_path as args, instead of
            asking user to change in tasknet/config.py?
        '''
        scenes = os.listdir(self.scene_path)
        random.shuffle(scenes)
        accept_scene = True
        self.online_sampling = online_sampling

        for scene in scenes:
            if scene_id is not None and scene != scene_id:
                continue
            if '_int' not in scene:
                continue
            if scene_kwargs is None:
                self.scene = scene_class(scene)
            else:
                self.scene = scene_class(scene, **scene_kwargs)

            # Reject scenes with missing non-sampleable objects
            # Populate scope with simulator objects
            if self.online_sampling:
                accept_scene, feedback = self.check_scene()
                if not accept_scene:
                    continue

            # Import scenes and objects into simulator
            self.import_scene()
            self.import_agent()

            if self.online_sampling:
                # Sample objects to satisfy initial conditions
                accept_scene, feedback = self.sample()
                if not accept_scene:
                    continue

                # Add clutter objects into the scenes
                self.clutter_scene()

        # Generate goal condition with the fully populated self.object_scope
        self.gen_goal_conditions()
        # assert accept_scene, 'None of the available scenes satisfy these initial conditions.'

        return accept_scene

    def gen_initial_conditions(self):
        if bool(self.parsed_initial_conditions[0]):
            self.initial_conditions = compile_state(
                [cond for cond in self.parsed_initial_conditions if cond[0]
                    not in ["inroom"]],
                self,
                scope=self.object_scope,
                object_map=self.objects)

    def gen_goal_conditions(self):
        if bool(self.parsed_goal_conditions[0]):
            self.goal_conditions = compile_state(
                self.parsed_goal_conditions, self, scope=self.object_scope, object_map=self.objects)

    def gen_ground_goal_conditions(self):
        self.ground_goal_state_options = get_ground_state_options(
            self.goal_conditions,
            self,
            scope=self.object_scope,
            object_map=self.objects)
        assert len(self.ground_goal_state_options) > 0

    def show_instruction(self):
        satisfied = self.currently_viewed_instruction in self.current_goal_status["satisfied"]
        natural_language_condition = self.natural_language_goal_conditions[
            self.currently_viewed_instruction]
        objects = self.goal_conditions[self.currently_viewed_instruction].get_relevant_objects(
        )
        # text_color = "green" if satisfied else "red"
        text_color = [83. / 255., 176. / 255., 72. / 255.] if satisfied \
            else [255. / 255., 51. / 255., 51. / 255.]

        return natural_language_condition, text_color, objects

    def iterate_instruction(self):
        self.currently_viewed_index = (
            self.currently_viewed_index + 1) % len(self.parsed_goal_conditions)
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]

    def check_scene(self):
        raise NotImplementedError

    def import_agent(self):
        raise NotImplementedError

    def import_scene(self):
        raise NotImplementedError

    def clutter_scene(self):
        raise NotImplementedError

    def sample(self, kinematic_only=False):
        raise NotImplementedError

    def check_setup(self):
        '''
        Check if scene will be viable for task
        :return: binary success + unsatisfied predicates
        '''
        return evaluate_state(self.initial_conditions)

    def check_success(self):
        '''
        Check if scene satisfies goal conditions and report binary success + unsatisfied predicates
        '''
        # print('Passing trivially. Later, check scene against final conditions and report success score.')
        self.current_success, self.current_goal_status = evaluate_state(
            self.goal_conditions)
        return self.current_success, self.current_goal_status

    #### CHECKERS ####
    def exist(self, objA):
        raise NotImplementedError

    def onTop(self, objA, objB):
        raise NotImplementedError

    def inside(self, objA, objB):
        raise NotImplementedError

    def nextTo(self, objA, objB):
        raise NotImplementedError

    def under(self, objA, objB):
        raise NotImplementedError

    def touching(self, objA, objB):
        raise NotImplementedError

    #### SAMPLERS ####
    def sampleOnTop(self, objA, objB):
        raise NotImplementedError

    def sampleInside(self, objA, objB):
        raise NotImplementedError

    def sampleNextTo(self, objA, objB):
        raise NotImplementedError

    def sampleUnder(self, objA, objB):
        raise NotImplementedError

    def sampleTouching(self, objA, objB):
        raise NotImplementedError


#### Util functions ####
def organize_objects(sim_objects, dsl_objects):
    objects = {}
    for sim_obj, dsl_obj in zip(sim_objects, dsl_objects):
        if dsl_obj.category in objects:
            objects[dsl_obj.category].append(sim_obj)
        else:
            objects[dsl_obj.category] = [sim_obj]
    # print(objects)
    return objects


class TaskNetScene(object):
    def __init__(self, scene_file):
        self.scene_file = scene_file
        self.objects = []

    def add_objects(self, objects):
        self.objects = objects
