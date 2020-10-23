# NOTE does a custom class that has only static methods deserve to be a class? Probably not. 
# should I just make this into a file called 'parsing.py'? Possibly. 
# Leaving as a class in case something comes up. I don't foresee needing instance methods, but 
# there could be a need for class methods if I put logical primitives into Parser. TBD. 

from tasknet.config import get_conditions_filename


class Parser(object):
    def __init__(self):
        pass

    @staticmethod
    def parse_conditions(atus_activity, mode):
        '''
        Parses contents of condition_filename into TaskNet checkable conditions. 
        :param atus_activity: string, name of atus activity 
        :param mode: string, "initial" or "final" indicating which type of 
                             conditions is being parsed 
        NOTE: requires that task_configs/atus_activity/initial have all conditions 
              formatted as "obj_category num_instances location obj_conditions"     TODO refine
              TODO final condition formatting needs even more thought, see personal notes 
              requires that all task_configs/atus_activity/<initial,final> have
              one condition per line.               
        '''
        conditions_filename = get_conditions_filename(atus_activity, mode)
        conditions = []
        if mode == 'initial':
            with open(conditions_filename, 'r') as conditions_file:
                for line in conditions_file:
                    conditions.append(Parser.parse_single_condition(line, mode))

        elif mode == 'final':
            pass 

        else:
            raise ValueError('Invalid checker mode "{}". Please use "initial" or "final".'.format(mode))
 
        return conditions

    @staticmethod
    def parse_single_condition(condition_string, mode):
        '''
        Parses a single condition string into TaskNet checkable condition.
        :param condition_string: string, one condition line from ATUS conditions files 
        :param mode: string, "initial" or "final" indicating which type of condition
                             is being parsed 
        '''
        # TODO: a lot! 
        #   Have to make the conditions. This will be added to over time. 
        #   Have to specify a condition format. For now, splitting initial and final 
        #       under the assumption that initial conds will follow my format.  
        if mode == 'initial':
            units = [unit.strip(' \n') for unit in condition_string.split(' ')]
            obj_category = units[0]
            num_instances = int(units[1])
            location = None         # TODO location
            obj_conditions = []     # TODO object states 
            single_condition = [obj_category, num_instances, location, obj_conditions]

        elif mode == 'final':
            pass

        return single_condition    


     
        
