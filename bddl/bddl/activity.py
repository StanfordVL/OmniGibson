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
        bddl.set_backend("iGibson")
    
    def get_object_scope(self):
        """Create unpopulated object scope to populate for generating goal and
            ground goal conditions. 

        Returns:
            dict<str: None>: unpopulated scope with string keys to be mapped to 
                             simulator object values 
        """
        return create_scope(self.parsed_objects)

    def get_initial_conditions(self):
        """Create compiled initial conditions that can be checked and sampled

        Returns:
            list<bddl.condition_evaluation.HEAD>: compiled conditions if initial 
                                                  condition definition is not 
                                                  empty else None 
        """
        if bool(self.parsed_initial_conditions[0]):
            initial_conditions = compile_state(
                [cond for cond in self.parsed_initial_conditions if cond[0] not in ["inroom"]],
                scope=self.object_scope,
                object_map=self.parsed_objects
            )
            return initial_conditions

    def get_goal_conditions(self, populated_object_scope):
        """Create compiled goal conditions with a populated object scope for checking

        Args:
            populated_object_scope (dict<str: simulator object>): scope mapping object 
                                                                  terms in BDDL to 
                                                                  simulator objects

        Returns:
            list<bddl.condition_evaluation.HEAD>: compiled conditions if goal condition
                                                  definition is not empty else None 
        """
        if bool(self.parsed_goal_conditions[0]):
            goal_conditions = compile_state(
                self.parsed_goal_conditions,
                scope=populated_object_scope,
                object_map=self.parsed_objects
            )
            return goal_conditions 
    
    def get_ground_goal_state_options(self, populated_object_scope):
        """Create compiled ground solutions to goal state with a populated object scope
           for checking progress on specific solutions 

        Args:
            populated_object_scope (dict<str: simulator object>): scope mapping object 
                                                                  terms in BDDL to 
                                                                  simulator objects

        Returns:
            list<bddl.condition_evaluation.HEAD>: compiled goal solutions
        
        Raises:
            AssertionError if there are no ground solutions
        """
        ground_goal_state_options = get_ground_state_options(
            self.goal_conditions,
            scope=populated_object_scope,
            object_map=self.parsed_objects
        )
        assert len(ground_goal_state_options) > 0
        return ground_goal_state_options


class Instructions(object):
    def __init__(self, behavior_activity, activity_definition, simulator_name):
        self.conditions = Conditions(behavior_activity, activity_definition, simulator_name)
    
    # TODO implement class to serve VR instructions