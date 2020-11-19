import random 
import os

from tasknet.config import TASK_CONFIGS_PATH, SCENE_PATH
from tasknet.sampler import Sampler
from tasknet.checker import TNChecker 
from tasknet.parser import Parser 

task_configs_path, scene_path = TASK_CONFIGS_PATH, SCENE_PATH


class TaskNetTask(object):
    
    def __init__(self, atus_activity, task_instance=0):
        self.atus_activity = atus_activity
        self.task_instance = task_instance      # TODO create option to randomly generate 
        self.sampler = Sampler()
        self.checker = TNChecker()
    
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

    def check_success(self):
        '''
        Check if scene satisfies final conditions and report binary success + success score 
        '''
        # print('Passing trivially. Later, check scene against final conditions and report success score.')
        self.final_conditions = self.gen_final_conditions()
        failed_conditions = []
        success = True 
        for cond_idx, condition in enumerate(self.final_conditions):
            if not condition(self.sampled_simulator_objects, self.sampled_dsl_objects):
                failed_conditions.append(cond_idx)   # TODO maybe make this more sensitive. List the condition itself? 
                                            #       List which objects specifically failed it? 
                success = False 
        return success, failed_conditions 

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

    #### TASK ####
    def gen_initial_conditions(self):
        # TODO change to parse/compile from current strat(?) Will have to figure out what to do for sampling 
        return Parser.parse_conditions(self.atus_activity, 'initial' + str(self.task_instance))

    def gen_final_conditions(self):
        '''
        Parses and compiles final conditions into a series of lambda functions, one per line in 
            final conditions file. Each lambda takes two inputs: a list of simulator objects, and 
            a list of DSL objects. They should have a one-to-one match. 
        TODO change to parse/compile from hard-code 
        '''
        conditions = []
        if self.task_instance == 0:
            # NOTE incomplete
            def cond0(sim_objects, dsl_objects):
                eggs = [sim_obj for sim_obj, dsl_obj in zip(sim_objects, dsl_objects) if dsl_obj.category == 'eggs']
                boxes = [sim_obj for sim_obj, dsl_obj in zip(sim_objects, dsl_objects) if dsl_obj.category == 'casserole_dish']
                for egg in eggs:
                    for box in boxes:
                        if inside(egg, box):
                            break
                for box in boxes:
                    for egg in eggs:
                        if not inside(egg, box):
                            return False 
                return True 
        
        elif self.task_instance == 2:
            def cond0(sim_objects, dsl_objects):
                objects = organize_objects(sim_objects, dsl_objects)
                all_containers_have_chips = all([any([self.inside(chips, container) for chips in objects['chips']]) for container in objects['container']])               
                return all_containers_have_chips
            
            def cond1(sim_objects, dsl_objects):
                objects = organize_objects(sim_objects, dsl_objects)
                all_containers_have_fruit = all([any([self.inside(fruit, container) for fruit in objects['fruit']]) for container in objects['container']])
                return all_containers_have_fruit

            def cond2(sim_objects, dsl_objects):
                objects = organize_objects(sim_objects, dsl_objects)
                all_containers_have_soda = all([any([self.inside(soda, container) for soda in objects['soda']]) for container in objects['container']])
                return all_containers_have_soda

            def cond3(sim_objects, dsl_objects):
                objects = organize_objects(sim_objects, dsl_objects)
                all_containers_have_eggs = all([any([self.inside(eggs, container) for eggs in objects['eggs']]) for container in objects['container']])
                return all_containers_have_eggs
            
            def cond4(sim_objects, dsl_objects):
                objects = organize_objects(sim_objects, dsl_objects)
                containers = [sim_obj for sim_obj, dsl_obj in zip(sim_objects, dsl_objects) if dsl_obj.category == 'container']
                all_containers_nextto_some_container = []
                for containerA in containers:
                    nextto_container = False
                    for containerB in containers:
                        if containerA.body_id == containerB.body_id:
                            continue
                        elif self.nextTo(containerA, containerB):
                            nextto_container = True 
                    all_containers_nextto_some_container.append(nextto_container)
                return all(all_containers_nextto_some_container)
            
            conditions = [
                            lambda sim_objects, dsl_objects: cond0(sim_objects, dsl_objects),
                            lambda sim_objects, dsl_objects: cond1(sim_objects, dsl_objects),  
                            lambda sim_objects, dsl_objects: cond2(sim_objects, dsl_objects),
                            # lambda sim_objects, dsl_objects: cond3(sim_objects, dsl_objects),
                            lambda sim_objects, dsl_objects: cond4(sim_objects, dsl_objects)
                        ]
            
        return conditions      


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
