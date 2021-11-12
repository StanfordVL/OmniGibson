import bddl
from bddl.activity import *


behavior_activity = "packing_lunches"
activity_definition = 0
simulator_name = "igibson"          

conds = Conditions(behavior_activity, activity_definition, simulator_name)
scope = get_object_scope(conds)
backend = None                      # TODO pass in backend from iGibson 
init = get_initial_conditions(conds, backend, scope)
goal = get_goal_conditions(conds, backend, scope)
populated_scope = None              # TODO populate scope in iGibson, e.g. through sampling 
goal = get_goal_conditions(conds, backend, populated_scope)
ground = get_ground_goal_state_options(conds, backend, populated_scope)
natural_init = get_natural_initial_conditions(conds)
natural_goal = get_natural_goal_conditions(conds)


print("####### Initial #######")
print(init)
print()
print("####### Goal #######")
print(goal)
print()
print("####### Ground #######")
print(ground)
print()
print("####### Natural language conditions #######")
print(natural_init)
print(natural_goal)

print("####### Goal evaluation #######")
for __ in range(100):
    print(evaluate_goal_conditions(goal))