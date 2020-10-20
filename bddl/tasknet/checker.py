class Checker(object):
    def __init__(self):
        pass 

    def check_conditions(self, conditions, scene, mode):
        '''
        For a set of parsed conditions and a scene, checks if the scene satisfies the conditions,
            doesn't satisfy them but can through sampling, or cannot satisfy them. 
        :param conditions: list of lists of condition elements 
        :param scene: scene object that can be queried for already-present object categories, 
                          locations, numbers, and object_conditions
        :param mode: string, "initial" or "final" indicating which type of conditions is being checked
        :returns: (bool, list of lists), boolean indicator of scene acceptability. With mode=final, this 
                                         means the scene passes all the conditions. With mode=initial, 
                                         this means the scene would pass all the conditions if 
                                         everything in to_sample was added; list of lists of condition
                                         elements.
        '''
        scene_characteristics = []      # TODO ideally, a list of all objects w/ specs, and rooms, in scene 

        accept_scene = True
        to_sample = []
        for condition in conditions:
            result = check_single_condition(condition, scene_characteristics, mode)
            if result == 'satisfied':
                continue                
            elif result == 'sampleable':
                to_sample.append(condition)
            elif result == 'unsatisfiable' or result == 'unsatisfied':
                accept_scene = False
                break

        return accept_scene, to_sample
    
    def check_single_condition(condition, scene_characteristics, mode):
        '''
        For a single parsed condition, checks if the condition is satisfied by the 
            existing scene, if the condition is unsatisfiable in the current scene,
            or if the condition is unsatisfied but satisfiable via sampling in the current scene. 
        :param condition: list of condition elements
        :param mode: string, "initial" or "final" indicating which type of conditions is being parsed.
        '''
        print('Trivially returning "sampleable". Later, will check if each condition is satisfied, sampleable, or unsatisfiable.')

        if False:   # TODO implement checking for already satisfied condition 
            return 'satisfied'
        if mode == 'initial':
            # TODO implement checking for unsatisfied condition that can be satisfied through sampling. 
            # This means checking if it's a sampleable object, but also if the object can be 
            #   1) put at that location 
            #   2) set with those object_properties
            # if either is false, it's unsatisfiable 
            if True: 
                return 'sampleable'
            elif False:
                return 'unsatisfiable'
        else:
            return 'unsatisfied'





