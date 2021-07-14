class BaseObject(object):
    def __init__(self, category, body_id=0):
        self._body_id = body_id
        self._position = None
        self._orientation = None
        self.category = category 
        self.properties = {'Categorizeable': category}

        # Set obj_conditions - TODO do we want all objects to have all object states? 
                             # TODO how do I get these from the simulator? That applies to pos and orn too. How much do we need to keep track of them? 
    
    @property
    def body_id(self):
        return self._body_id

    @body_id.setter                          # TODO this has to be allowed because sim obj body_ids don't exist
                                            # until they're imported into simulator, but this worries me 
    def obj_id(self, new_body_id):
        self._body_id = new_body_id

    @property 
    def position(self):
        return self._position

    @position.setter                        
    def position(self, new_position):
        self._position = new_position

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, new_orientation):
        self._orientation = new_orientation

    @property           
    def size(self):     # TODO do I need this 
        '''
        get object size
        '''
        pass

    def aabb(self):     # TODO do I need this 
        pass 
        
    def populate_object_states(self, object_taxonomy):
        object_states = {}     # TODO obtain by traversing object_taxonomy, hopefully with builtin functions
        self.object_states.update(object_states)
        
    
















