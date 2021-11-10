import bddl 
from bddl.parsing import parse_domain, parse_problem, gen_natural_language_conditions
from bddl.condition_evaluation import create_scope, compile_state, evaluate_state, get_ground_state_options
from bddl.object_taxonomy import ObjectTaxonomy


class Conditions(object):
    def __init__(self, 
                 behavior_activity, 
                 activity_definition, 
                 simulator_name):
        """Object to store behavior activity content and compile conditions for checking and 
            simulator use 

        Args:
            behavior_activity (str): behavior activity being used 
            activity_definition (int): specific definition of behavior_activity
            simulator_name (str): simulator that BEHAVIOR is being used with
        """
        self.behavior_activity = behavior_activity 
        self.activity_definition = activity_definition 
        domain_name, *__ = parse_domain(simulator_name)
        __, self.parsed_objects, self.parsed_initial_conditions, self.parsed_goal_conditions = parse_problem(
                        self.behavior_activity,
                        self.activity_definition, 
                        domain_name
        )

######## API ########    

def get_object_scope(conds):
    """Create unpopulated object scope to populate for generating goal and
        ground goal conditions. 

    Args:
        conds (Conditions): conditions for the particular activity and definition

    Returns:
        dict<str: None>: unpopulated scope with string keys to be mapped to 
                            simulator object values 
    """
    return create_scope(conds.parsed_objects)

def get_initial_conditions(conds, backend, scope):
    """Create compiled initial conditions that can be checked and sampled

    Args:
        conds (Conditions): conditions for the particular activity and definition

    Returns:
        list<bddl.condition_evaluation.HEAD>: compiled conditions if initial 
                                                condition definition is not 
                                                empty else None 
    """
    if bool(conds.parsed_initial_conditions[0]):
        initial_conditions = compile_state(
            [cond for cond in conds.parsed_initial_conditions if cond[0] not in ["inroom"]],
            backend,
            scope=scope,
            object_map=conds.parsed_objects
        )
        return initial_conditions

def get_goal_conditions(conds, backend, scope):
    """Create compiled goal conditions with a populated object scope for checking

    Args:
        conds (Conditions): conditions for the particular activity and definition
        populated_object_scope (dict<str: simulator object>): scope mapping object 
                                                                terms in BDDL to 
                                                                simulator objects

    Returns:
        list<bddl.condition_evaluation.HEAD>: compiled conditions if goal condition
                                                definition is not empty else None 
    """
    if bool(conds.parsed_goal_conditions[0]):
        goal_conditions = compile_state(
            conds.parsed_goal_conditions,
            backend,
            scope=scope,
            object_map=conds.parsed_objects
        )
        return goal_conditions 

def get_ground_goal_state_options(conds, backend, scope):
    """Create compiled ground solutions to goal state with a populated object scope
        for checking progress on specific solutions 

    Args:
        conds (Conditions): conditions for the particular activity and definition
        populated_object_scope (dict<str: simulator object>): scope mapping object 
                                                                terms in BDDL to 
                                                                simulator objects

    Returns:
        list<bddl.condition_evaluation.HEAD>: compiled goal solutions
    
    Raises:
        AssertionError if there are no ground solutions
    """
    ground_goal_state_options = get_ground_state_options(
        conds.goal_conditions,
        backend,
        scope=scope,
        object_map=conds.parsed_objects
    )
    assert len(ground_goal_state_options) > 0
    return ground_goal_state_options

def evaluate_goal_conditions(goal_conditions):
    """Evaluate compiled goal state to see if current simulator state has been met 

    Args:
        goal_conditions (list<bddl.condition_evaluation.HEAD>): list of compiled
                                                                goal conditions with
                                                                populated scope 

    Returns:
        bool, dict<str: list<int>>: [description]
    """
    return evaluate_state(goal_conditions)


class Instructions(object):
    def __init__(self, behavior_activity, activity_definition, simulator_name):
        self.conditions = Conditions(behavior_activity, activity_definition, simulator_name)
    
    # TODO implement class to serve VR instructions