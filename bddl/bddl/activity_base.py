import random
import os

from bddl.parsing import parse_domain, parse_problem, gen_natural_language_conditions
from bddl.condition_evaluation import create_scope, compile_state, evaluate_state, get_ground_state_options

import numpy as np
from bddl.object_taxonomy import ObjectTaxonomy


class BEHAVIORActivityInstance(object):

    def __init__(self, behavior_activity=None, activity_definition=None, scene_path=None, predefined_problem=None):
        self.scene_path = scene_path
        self.object_taxonomy = ObjectTaxonomy()
        self.update_problem(behavior_activity, activity_definition,
                            predefined_problem=predefined_problem)

    def update_problem(self, behavior_activity, activity_definition, predefined_problem=None):
        if predefined_problem is not None:
            self.behavior_activity = behavior_activity
            self.activity_definition = "predefined"
        else:
            self.behavior_activity = behavior_activity
            self.activity_definition = activity_definition
        domain_name, requirements, types, actions, predicates = parse_domain(
            "igibson")
        problem_name, self.objects, self.parsed_initial_conditions, self.parsed_goal_conditions = parse_problem(
            self.behavior_activity,
            self.activity_definition,
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

            self.move_agent()

        # Generate goal condition with the fully populated self.object_scope
        self.gen_goal_conditions()

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
        text_color = [83. / 255., 176. / 255., 72. / 255.] if satisfied \
            else [255. / 255., 51. / 255., 51. / 255.]

        return natural_language_condition, text_color, objects

    def iterate_instruction(self):
        self.currently_viewed_index = (
            self.currently_viewed_index + 1) % len(self.parsed_goal_conditions)
        self.currently_viewed_instruction = \
            self.instruction_order[self.currently_viewed_index]

    def check_scene(self):
        raise NotImplementedError

    def import_agent(self):
        raise NotImplementedError

    def move_agent(self):
        pass

    def import_scene(self):
        raise NotImplementedError

    def clutter_scene(self):
        raise NotImplementedError

    def sample(self, kinematic_only=False):
        raise NotImplementedError

    def check_success(self):
        '''
        Check if scene satisfies goal conditions and report binary success + unsatisfied predicates
        '''
        # print('Passing trivially. Later, check scene against final conditions and report success score.')
        self.current_success, self.current_goal_status = evaluate_state(
            self.goal_conditions)
        return self.current_success, self.current_goal_status
