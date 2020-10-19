import os 
import random 

from tasknet.config import OBJECT_MODEL_PATH, get_object_filepath


class Sampler(object):
    def __init__(self):
        pass

    def sample_objects(self, to_sample, object_class):
        '''
        :param: to_sample: list of (object_category, num instance, object, obj_states) tuples, 
                           specifying what needs to be sampled. obj_category 
                           must be a designated object model category label.  
        :returns: Object instances with appropriate specifications  
        NOTE: currently only handles instantiating objects. Demo 2.2 will handle 
              instantiating objects with specific locations. Demos 3-7 will 
              add instantiation of objects with different object states. 
        '''
        flat_tosample = []
        for obj_category, num_instances, location, obj_states in to_sample: # turn objs with num_instances>1 into num_instances single objects
            flat_tosample.extend([[obj_category, location, obj_states] for __ in range(num_instances)])
        
        sampled_objects = []
        for obj_category, location, obj_states in to_sample:     
            # Get random sample from `object category`
            # TODO handle non-leaf nodes 
            obj_category_instance = random.choice(os.listdir(os.path.join(OBJECT_MODEL_PATH, obj_category)))
            obj_file = get_object_filepath(obj_category_instance)
            obj = object_class(filename=obj_file)       # NOTE currently iGibson specific 
             
            # Generate specific coordinates in `location`
            obj_pos = [0, 0, 0]
            # TODO also NOTE not putting them into the scene itself here 

            # Set `obj_states`. Throw error if object can't have that state. This should never happen for ATUS.
            # TODO 
            sampled_objects.append((obj, obj_pos))

        return sampled_objects 

            
            

