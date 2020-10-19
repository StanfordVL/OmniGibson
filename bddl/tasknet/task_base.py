import random 
import os

from tasknet.config import TASK_CONFIGS_PATH, SCENE_PATH
from tasknet.sampler import Sampler

task_configs_path, scene_path = TASK_CONFIGS_PATH, SCENE_PATH


class TaskNetTask(object):
    
    def __init__(self, atus_activity):
        self.atus_activity = atus_activity
        self.initial_conditions = load_conditions(os.path.join(TASK_CONFIGS_PATH, self.atus_activity, 'initial'))
        self.final_conditions = load_conditions(os.path.join(TASK_CONFIGS_PATH, self.atus_activity, 'final'))
        self.sampler = Sampler()
    
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
        initial_satisfied = False 
        while not initial_satisfied:       
            try:
                self.scene_name = scenes.pop()
            except:
                raise ValueError('None of the available scenes satisfy these initial conditions.')
            self.scene = scene_class(self.scene_name)                   
            initial_satisfied, to_sample = self.check_scene()      # TODO get whether the scene is viable and the objects to be sampled 
            
        sampled_objects = self.sampler.sample_objects(to_sample, object_class)           # TODO get list of sampled objects/locations/conditions
        self.add_objects(sampled_objects)                          # TODO add objects to scene 
        return self.scene_name, self.scene

    def check_scene(self):
        print('Passing trivially. Later, will check scene against initial conditions and generate to_sample list.')
        return True, []

    def add_objects(self, sampled_objects):
        print('Passing trivially. Later, will add input objects to scene.')
        pass 

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
            

def load_conditions(fpath):
    '''
    Parse initial or final conditions into code 
    '''
    with open(fpath, 'r') as conditions_file:
        conditions = []
        for raw_condition in conditions_file:       # NOTE each line better be exactly one condition (change if not)
            conditions.append(dsl_parse(raw_condition)) 
    
    return conditions
        
        
def dsl_parse(raw_condition):
    '''
    Parses one raw condition (string) into DSL condition
    :params: raw_condition (string)
    :returns: DSL predicate 
    '''
    # TODO implement, possibly in separate file  
    pass     


