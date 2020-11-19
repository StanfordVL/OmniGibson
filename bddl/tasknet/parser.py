# NOTE these are basically static methods, but it's hard to turn it into just
# a set of parsing functions because of the underlying PDDLParser object. 
# Maybe try later.  

from tasknet.config import get_conditions_filename
from pddl_parser import PDDLParser
 
 
class TNParser(PDDLParser):

    def parse_predicates(self, atus_activity, instance):
        domain_filename = get_definition_filename(atus_activity, instance, domain=True)
        self.parse_domain(domain_filename)      # sets self.predicates
        return self.predicates 
    
    def parse_initial_state(self, atus_activity, instance):
        problem_filename = get_definition_filename(atus_activity, instance)
        self.parse_problem(problem_filename)    # sets the three things returned here 
        return self.objects, self.state
    
    def parse_goal(self, atus_activity, instance):
        problem_filename = get_definition_filename(atus_activity, instance)
        self.parse_problem(problem_filename)
        return self.positive_goals, self.negative_goals


"""
# "Deprecated", as if it was ever deployed 
class _Parser(object):
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
"""
     
        
