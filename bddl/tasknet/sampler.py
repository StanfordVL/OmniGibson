import numpy as np 
import os 
import random 

from tasknet.config import OBJECT_MODEL_PATH, get_object_filepath


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
        for obj_category, location, obj_conditions in to_sample:     

            # Pick leaf category 
            leaf_category = obj_category                # TODO handle non-leaf nodes 

            # Generate specific coordinates in `location`
            obj_pos, obj_orn = np.random.uniform(low=0, high=2, size=3), [0, 0, 0, 1])

            # SIMULATOR OBJECTS 
            # Get random object model sample from `leaf category`
            leaf_category_instance = random.choice(os.listdir(os.path.join(OBJECT_MODEL_PATH, leaf_category)))
            obj_file = get_object_filepath(obj_category_instance)
            sim_obj = object_class(filename=obj_file)       # NOTE currently iGibson specific 

            # Set `obj_conditions`. Throw error if object can't have that state. This should never happen for ATUS.
            # TODO 
            sampled_simulator_objects.append([obj, obj_pos, obj_orn])

            # DSL OBJECTS 
            dsl_obj = BaseObject(obj_category)                   
            dsl_obj.set_position(obj_pos)
            dsl_obj.set_orientation(obj_orn)
            # TODO set dsl_obj object_conditions and possibly states. Again, throw error if invalid state for object. 
            sampled_dsl_objects.append(dsl_obj)

        return sampled_simulator_objects, sampled_dsl_objects       # TODO sim and dsl objs need to be associated 

            
            

