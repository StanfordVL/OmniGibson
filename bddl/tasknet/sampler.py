from config import OBJECT_MODEL_PATH


class Sampler(object):
    def __init__(self):
        pass

    def sample_objects(self, to_sample, object_class):
        '''
        :param: to_sample: list of (object_category, num instance, object, state) tuples, 
                           specifying what needs to be sampled. obj_category 
                           must be a designated object model category label.  
        :returns: Object instances with appropriate specifications  
        NOTE: currently only handles instantiating objects. Demo 2.2 will handle 
              instantiating objects with specific locations. Demos 3-7 will 
              add instantiation of objects with different object states. 
        '''
        for obj_category, num_instances, location, obj_states in to_sample:
        
        flat_tosample = []
        for obj in to_sample:
            flat_tosample.extend([obj for __ in range(obj[1])])     # turn objs with num_instances>1 into num_instances single objects
        
        for obj_category, num_instances, location, obj_states in to_sample:     # num_instances is now inaccurate (it'll be whatever it used to be but it should actually be 1). Should I make the above code messier to change it?
            obj_file = random.choice(os.listdir(os.path.join(OBJECT_MODEL_PATH, obj_category)))


