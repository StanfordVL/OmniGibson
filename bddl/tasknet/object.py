class BaseObject(object):
    def __init__(self, category):
        self._position = None
        self._orientation = None
        self.category = category 
        self.properties = {'Categorizeable': category}

        # Set obj_conditions - TODO do we want all objects to have all object states? 
                             # TODO how do I get these from the simulator? That applies to pos and orn too. How much do we need to keep track of them? 
    
    @property 
    def position(self):
        return self._position

    @position.setter                        
    def position(self, new_position):
        print('position setter called')
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
        
    def populate_object_properties(self, object_taxonomy):
        object_properties = {}     # TODO obtain by traversing object_taxonomy, hopefully with builtin functions
        self.object_properties.update(object_properties)
        
    
















