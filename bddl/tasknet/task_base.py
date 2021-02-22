import random
import os
import sys
import pdb

from tasknet.config import SCENE_PATH
from tasknet.sampler import Sampler
from tasknet.parsing import parse_domain, parse_problem
from tasknet.condition_evaluation import create_scope, compile_state, evaluate_state

import numpy as np
from IPython import embed


class TaskNetTask(object):
    # TODO
    #   1. Update with new object formats
    #   2. Update initialize() to work with self.check_setup()
    #   3. Update initialize() to work with sampler code
    #   4. Various other adaptations to be seen

    def __init__(self, atus_activity, task_instance=0, scene_path=SCENE_PATH):
        self.atus_activity = atus_activity
        self.scene_path = scene_path
        # TODO create option to randomly generate
        self.task_instance = task_instance
        domain_name, requirements, types, actions, predicates = parse_domain(
            self.atus_activity, self.task_instance)
        problem_name, self.objects, self.parsed_initial_conditions, self.parsed_goal_conditions = parse_problem(
            self.atus_activity, self.task_instance, domain_name)
        self.object_scope = create_scope(self.objects)
        self.obj_inst_to_obj_cat = {
            obj_inst: obj_cat
            for obj_cat in self.objects
            for obj_inst in self.objects[obj_cat]
        }

    def initialize(self, scene_class, scene_id=None):
        '''
        Check self.scene to see if it works for this Task. If not, resample.
        Populate self.scene with necessary objects.
        :param scene_class: scene class from simulator
        TODO should this method take scene_path and object_path as args, instead of
            asking user to change in tasknet/config.py?
        '''

        scenes = os.listdir(self.scene_path)
        random.shuffle(scenes)
        accept_scene = False
        for scene in scenes:
            print('SCENE:', scene)
            if scene == "background": 
                continue
            if scene_id is not None and scene != scene_id:
                continue
            if scene == 'Beechwood_0_int':
                pdb.set_trace()
            else:       # TODO remove after debugging
                continue
            self.scene_id = scene
            self.scene = scene_class(scene)
            # self.scene = scene_class(scene, load_object_categories=[
            #                          'table', 'counter', 'fridge'])

            # Reject scenes with missing non-sampleable objects
            # Populate scope with simulator objects
            accept_scene = self.check_scene()
            if not accept_scene:
                continue

            # Import scenes and objects into simulator
            self.import_scene()

            # Check for initial conditions, return failed conditions
            self.gen_initial_conditions()
            all_passed, results = self.check_setup()
            failed_conditions = [self.initial_conditions[cond_idx]
                                 for cond_idx in results['unsatisfied']]
            accept_scene = self.sample(failed_conditions)
            if not accept_scene:
                continue

        assert accept_scene, 'None of the available scenes satisfy these initial conditions.'

        self.gen_goal_conditions()

        return self.scene_id, self.scene

    def gen_initial_conditions(self):
        if bool(self.parsed_initial_conditions[0]):
            self.initial_conditions = compile_state(
                [cond for cond in self.parsed_initial_conditions if cond[0] != 'inroom'], 
                self, 
                scope=self.object_scope, 
                object_map=self.objects)

    def gen_goal_conditions(self):
        if bool(self.parsed_goal_conditions[0]):
            self.goal_conditions = compile_state(
                self.parsed_goal_conditions, self, scope=self.object_scope, object_map=self.objects)

    def check_scene(self):
        raise NotImplementedError

    def import_scene(self):
        raise NotImplementedError

    def sample(self):
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
        return evaluate_state(self.goal_conditions)

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
