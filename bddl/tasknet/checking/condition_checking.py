# NOTE the parsing is basically static methods, but it's hard to turn it into just
# a set of parsing functions because of the underlying PDDLParser object. 
# Maybe try later.  

from tasknet.config import get_definition_filename
from tasknet.checking.pddl_parser import PDDLParser
from tasknet.checking.logic import HEAD
 

#################### PARSING ####################

# TODO turn parsing into pure functionality and no classes, 
# use functions to parse specific ATUS activity + instance 
class Parser(PDDLParser):
    def __init__(self, atus_activity, instance):
        super().__init__()
        self.atus_activity = atus_activity
        self.instance = instance
        domain_filename = get_definition_filename(atus_activity, instance, domain=True)
        self.parse_domain(domain_filename)      # sets self.predicates
        problem_filename = get_definition_filename(atus_activity, instance)
        self.parse_problem(problem_filename)    # sets self.goals

    def get_predicates(self):
        return self.predicates 
    
    def get_objects(self):
        return self.objects
    
    def get_initial_state(self):
        return self.initial_state
    
    def get_goal(self):
        return self.goal_state


#################### COMPILATION ####################

def create_scope():
    pass 
    # TODO 

def compile_state(parsed_state, task, scope=None):
    compiled_state = []
    for parsed_condition in parsed_state:
        scope = scope if scope is not None else {}
        compiled_state.append(HEAD(scope, task, body))
    return compiled_state 


#################### EVALUATION ####################

def evaluate_state(compiled_state):
    results = {'satisfied': [], 'unsatisfied': []}
    for i, compiled_condition in enumerate(compiled_state):
        if compiled_condition.evaluate():
            results['satisfied'].append(i)
    return not bool(results['unsatisfied']), results 


if __name__ == '__main__':
    import pprint
    tnp = Parser('kinematic_checker_testing', 2)
    print('Predicates:', tnp.get_predicates())
    objects = tnp.get_objects()
    init_state = tnp.get_initial_state()
    print('Objects:', objects)
    print('Initial state:', init_state)
    goal = tnp.get_goal()
    print('Goals:', goal)
