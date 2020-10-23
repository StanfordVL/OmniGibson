import random 
import os

from tasknet.config import TASK_CONFIGS_PATH, SCENE_PATH
from tasknet.sampler import Sampler
from tasknet.checker import Checker 
from tasknet.parser import Parser 

task_configs_path, scene_path = TASK_CONFIGS_PATH, SCENE_PATH


class TaskNetTask(object):
    
    def __init__(self, atus_activity):
        self.atus_activity = atus_activity
        self.initial_conditions = Parser.parse_conditions(self.atus_activity, 'initial')
        self.final_conditions = Parser.parse_conditions(self.atus_activity, 'final')
        self.sampler = Sampler()
        self.checker = Checker()
    
    def initialize(self, scene_class, object_class):
        '''
        Check self.scene to see if it works for this Task. If not, resample. 
        Populate self.scene with necessary objects. 
        :param: scene_class: scene class from simulator 
        :param: object_class: object class from simulator 
        TODO should this method take scene_path and object_path as args, instead of
            asking user to change in tasknet/config.py? 
        '''
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
        # adding objects to the simulator happens in the simulator-specific task instance. 
        
        return self.scene_name, self.scene


    def check_success(self):
        '''
        Check if scene satisfies final conditions and report binary success + success score 
        '''
        print('Passing trivially. Later, check scene against final conditions and report success score.')
        return all(self.final_conditions) 


class TaskNetScene(object): 
    def __init__(self, scene_file):
        self.scene_file = scene_file
        self.objects = []

    def add_objects(self, objects):
        self.objects = objects 
