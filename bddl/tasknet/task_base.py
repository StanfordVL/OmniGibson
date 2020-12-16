import random 
import os
import sys 

from tasknet.config import get_definition_filename
from tasknet.sampler import Sampler
from tasknet.parsing import parse_domain, parse_problem
from tasknet.condition_evaluation import compile_state, evaluate_state


class TaskNetTask(object):
    
    def __init__(self, atus_activity, task_instance=0):
        self.atus_activity = atus_activity
        self.task_instance = task_instance      # TODO create option to randomly generate 
        self.gen_conditions()

    def initialize(self, scene_class, object_class):
        '''
        Check self.scene to see if it works for this Task. If not, resample. 
        Populate self.scene with necessary objects. 
        :param scene_class: scene class from simulator 
        :param object_class: object class from simulator 
        :param handmade_scene: None (if sampling) or simulator scene self.task_instance, meant to 
                               replace TaskNet-generated scene 
        TODO should this method take scene_path and object_path as args, instead of
            asking user to change in tasknet/config.py? 
        '''
        self.initial_conditions = self.gen_initial_conditions() 

        scenes = os.listdir(scene_path)
        random.shuffle(scenes)
        accept_scene = False 
        while not accept_scene:       
            try:
                self.scene_name = scenes.pop()
                if self.scene_name == 'background':     # TODO what is 'background'?
                    continue
            except:
                raise ValueError('None of the available scenes satisfy these initial conditions.')
            self.scene = scene_class(self.scene_name)                   
            accept_scene, to_sample = self.checker.check_conditions(self.initial_conditions, self.scene, 'initial')
        
        self.sampled_simulator_objects, self.sampled_dsl_objects = self.sampler.sample_objects(to_sample, object_class)          
        # TODO Right now, self.sampled_dsl_objects is just a list of tasknet.object.BaseObjects (they need
        #           to be associated with their sim counterparts). Is this list good enough, or do we need
        #           a TaskNetScene (or something)? 
        # adding objects to the simulator happens in the simulator-specific task self.task_instance. 
        # TODO we need self.simulator_objects and self.dsl_objects that have not only the sim and dsl reprs
        #           of the sampled objects, but even the scene objects, since they will also be used in the 
        #           final conditions. Those lists should be these sampled lists, extended to include scene objs. 
        
        return self.scene_name, self.scene 

    def gen_conditions(self):
        domain_name, requirements, types, actions, predicates = parse_domain(self.atus_activity, self.task_instance)
        problem_name, objects, parsed_initial_conditions, parsed_goal_conditions = parse_problem(self.atus_activity, self.task_instance, domain_name)
        self.initial_conditions = compile_state(parsed_initial_state, self)
        self.goal_conditions = compile_state(parsed_goal_state, self)

    def check_success(self):
        '''
        Check if scene satisfies final conditions and report binary success + success score 
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
    