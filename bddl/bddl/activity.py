from ast import parse
import numpy as np
import os 
import random 

from bddl.parsing import parse_domain, parse_problem, gen_natural_language_conditions
from bddl.condition_evaluation import create_scope, compile_state, evaluate_state, get_ground_state_options
from bddl.object_taxonomy import ObjectTaxonomy


def get_object_taxonomy():
    return ObjectTaxonomy()

def get_conditions(behavior_activity, 
                   activity_definition,
                   simulator_name, 
                   predefined_problem=None):
    domain_name, *__ = parse_domain(simulator_name)
    __, parsed_objects, parsed_initial_conditions, parsed_goal_conditions = parse_problem(
        behavior_activity,
        activity_definition,
        domain_name,
    )

    object_scope = create_scope(parsed_objects)

    if bool(parsed_initial_conditions[0]):
        initial_conditions = compile_state(
            [cond for cond in parsed_initial_conditions if cond[0] not in ["inroom"]],
            # task??,
            scope=object_scope,
            object_map=parsed_objects
        )
    
    if bool(parsed_goal_conditions[0]):
        goal_conditions = compile_state(
            parsed_goal_conditions, 
            # task??
            scope=object_scope, 
            object_map=parsed_objects
        )
    
    ground_goal_state_options = get_ground_state_options(
        goal_conditions,
        #task??
        scope=object_scope, 
        object_map=parsed_objects
    )
    assert len(ground_goal_state_options) > 0