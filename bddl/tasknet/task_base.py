import random 
import os

from config import TASK_CONFIGS_PATH


class TaskNetTask(object):
    
    def __init__(self, atus_activity):
        self.atus_activity = atus_activity
        self.initial_conditions = load_conditions(os.path.join(TASK_CONFIGS_FOLDER, self.atus_activity, 'initial'))
        self.final_conditions = load_conditions(os.path.join(TASK_CONFIGS_FOLDER, self.atus_activity, 'final'))
    
    def initialize(self, scene_path, scene_type):
        '''
        Check self.scene to see if it works for this Task. If not, resample.
        '''
        scenes = random.shuffle(os.listdir(scene_path))
        initial_satisfied = False 
        while not initial_satisfied:       
            self.scene = scene_type(scenes.pop())             # NOTE is this supposed to be the arena?
            initial_satisfied, to_sample = True, None         # TODO  with Checker
            
        sampled_objects = []                                  # TODO with Sampler 
        self.scene.add_objects(sampled_objects) 
        return self.scene

    def check_success(self):
        '''
        Check if scene satisfies final conditions and report binary success + success score 
        '''
        # TODO implement 
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


