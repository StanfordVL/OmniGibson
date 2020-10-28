import numpy as np 
import os 
import random 

from tasknet.config import OBJECT_MODEL_PATH, get_object_filepath
from tasknet.object import BaseObject


class Sampler(object):
    def __init__(self):
        pass

    def sample_objects(self, to_sample, object_class):
        '''
        :param: to_sample: list of (object_category, num instance, object, obj_conditions) tuples, 
                           specifying what needs to be sampled. obj_category 
                           must be a designated object model category label.  
        :returns: Object instances with appropriate specifications  
        NOTE: currently only handles instantiating objects. Demo 2.2 will handle 
              instantiating objects with specific locations. Demos 3-7 will 
              add instantiation of objects with different object states. 
        '''
        flat_tosample = []
        for obj_category, num_instances, location, obj_conditions in to_sample: # turn objs with num_instances>1 into num_instances single objects
            flat_tosample.extend([[obj_category, location, obj_conditions] for __ in range(num_instances)])
        
        sampled_simulator_objects = []
        sampled_dsl_objects = []
        for obj_category, location, obj_conditions in flat_tosample:     

            # Pick leaf category 
            leaf_category = obj_category                # TODO handle non-leaf nodes 

            # Generate specific coordinates in `location`       # TODO handle location 
            obj_pos, obj_orn = (np.random.uniform(low=0, high=2, size=3), [0, 0, 0, 1])
            obj_pos, obj_orn = ([0, 0, np.random.random()], [0, 0, 0, 1])

            # SIMULATOR OBJECT 
            # Get random object model sample from `leaf category`
            leaf_category_instance = random.choice(os.listdir(os.path.join(OBJECT_MODEL_PATH, leaf_category)))
            obj_file = get_object_filepath(leaf_category, leaf_category_instance)
            sim_obj = object_class(filename=obj_file)       # NOTE currently iGibson specific 

            # Set `obj_conditions`. Throw error if object can't have that state. This should never happen for ATUS.
            # TODO 
            sampled_simulator_objects.append([sim_obj, obj_pos, obj_orn])

            # TASKNET OBJECT 
            dsl_obj = BaseObject(obj_category, sim_obj.body_id)     # TODO is this the right way to get ID? I could 
                                                                   # do sim_obj.load. Is that safer/
                                                                   # preferred? Can I assume that the objects were 
                                                                   # already loaded? Can I assume that regardless of 
                                                                   # object type? I'm leaving it as self.body_id
                                                                   # because that seems less simulator-specific, but 
                                                                   # it is TODO simulator-specific 
            # dsl_obj.set_position(obj_pos)
            dsl_obj.position = obj_pos
            dsl_obj.orientation = obj_orn
            # TODO set dsl_obj object_conditions and possibly states. Again, throw error if invalid state for object. 
            sampled_dsl_objects.append(dsl_obj)

        return sampled_simulator_objects, sampled_dsl_objects       # TODO once sim_obj_id is obtained, assuming it is unique, these will be associated 

            
            

