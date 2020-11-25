# NOTE these are basically static methods, but it's hard to turn it into just
# a set of parsing functions because of the underlying PDDLParser object. 
# Maybe try later.  

from tasknet.config import get_definition_filename
from tasknet.pddl_parser import PDDLParser
 

### PARSING ###

class TNParser(PDDLParser):
    def __init__(self, atus_activity, instance):
        super().__init__()
        self.atus_activity = atus_activity
        self.instance = instance
        domain_filename = get_definition_filename(atus_activity, instance, domain=True)
        self.parse_domain(domain_filename)      # sets self.predicates
        problem_filename = get_definition_filename(atus_activity, instance)
        self.parse_problem(problem_filename)

    def get_predicates(self):
        return self.predicates 
    
    def get_initial_state(self):
        return self.objects, self.state
    
    def get_goal(self):
        return self.positive_goals, self.negative_goals


### COMPILING ###
# def compile_sampleable_objects(task_objects):
#     for obj_cat in task_objects:
#         num_objs = len(task_objects[obj_cat])
        

# def compile_initialstate_conditions(task_objects, initial_state):
#     # Create to-sample list 
#     objects_to_sample = {}
#     for obj_cat in task_objects:
#         if sampleable(obj_cat):
#             for obj_inst in task_objects[obj_cat]:
#                 objects_to_sample[obj_inst] = [obj_cat]
    
#     for condition in initial_state:
#         predicate = condition[0]
#         sampleable_object_involved = False 
        
#         # If there are sampleable objects involved, add this condition to their records
#         for name in objects_to_sample:
#             if name in condition[1:]:        # if the object instance name is an arg to the (unary or binary) predicate...
#                 objects_to_sample[name].append(condition)           # ... put the whole condition into that object instance's record.
#                                                                     # NOTE this means that if there's a binary predicate with two sampleable arguments, it will appear twice in the to_sample list. But it's just redundant, I don't think it's problematic. 
#                 sampleable_object_involved = True 
        
#         # If there are nonsampleable objects involved, TODO 
#         # NOTE see notes for what decision is made 
#         conditions = []
    
#     return objects_to_sample, conditions 


# def compile_goal_conditions(goal):
#     # TODO run condition compiler on positive and negative goals 
#     pass


# def compile_condition(condition):
#     # TODO turn a given condition into... a lambda function? 
#     pass 


# Utils 
def sampleable(obj_cat):
    # TODO replace with query from database that says (not)sampleable
    not_sampleables = ['shelf', 'counter']
    if obj_cat in not_sampleables:
        return False
    else:
        return True 
    # TODO change the above to elif obj_cat in sampleables and have a 
    # third case to throw an error with unsupported object categories 
    # NOTE or should this happen earlier? 


if __name__ == '__main__':
    import pprint
    tnp = TNParser('kinematic_checker_testing', 2)
    print('Predicates:', tnp.get_predicates())
    objects, init_state = tnp.get_initial_state()
    print('Objects:', objects)
    print('Initial state:', init_state)
    positive_goals, negative_goals = tnp.get_goal()
    print('Positive goals:', positive_goals)
    print('Negative goals:', negative_goals)



# # "Deprecated", as if it was ever deployed 
# class _Parser(object): 
#     def __init__(self):
#         pass

#     @staticmethod
#     def parse_conditions(atus_activity, mode):
#         '''
#         Parses contents of condition_filename into TaskNet checkable conditions. 
#         :param atus_activity: string, name of atus activity 
#         :param mode: string, "initial" or "final" indicating which type of 
#                              conditions is being parsed 
#         NOTE: requires that task_configs/atus_activity/initial have all conditions 
#               formatted as "obj_category num_instances location obj_conditions"     TODO refine
#               TODO final condition formatting needs even more thought, see personal notes 
#               requires that all task_configs/atus_activity/<initial,final> have
#               one condition per line.               
#         '''
#         conditions_filename = get_conditions_filename(atus_activity, mode)
#         conditions = []
#         if mode == 'initial':
#             with open(conditions_filename, 'r') as conditions_file:
#                 for line in conditions_file:
#                     conditions.append(Parser.parse_single_condition(line, mode))

#         elif mode == 'final':
#             pass 

#         else:
#             raise ValueError('Invalid checker mode "{}". Please use "initial" or "final".'.format(mode))
 
#         return conditions

#     @staticmethod
#     def parse_single_condition(condition_string, mode):
#         '''
#         Parses a single condition string into TaskNet checkable condition.
#         :param condition_string: string, one condition line from ATUS conditions files 
#         :param mode: string, "initial" or "final" indicating which type of condition
#                              is being parsed 
#         '''
#         # TODO: a lot! 
#         #   Have to make the conditions. This will be added to over time. 
#         #   Have to specify a condition format. For now, splitting initial and final 
#         #       under the assumption that initial conds will follow my format.  
#         if mode == 'initial':
#             units = [unit.strip(' \n') for unit in condition_string.split(' ')]
#             obj_category = units[0]
#             num_instances = int(units[1])
#             location = None         # TODO location
#             obj_conditions = []     # TODO object states 
#             single_condition = [obj_category, num_instances, location, obj_conditions]

#         elif mode == 'final':
#             pass

#         return single_condition    

     
        
