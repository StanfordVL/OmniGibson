import bddl
from bddl.activity import *


behavior_activity = "packing_lunches"
activity_definition = 0
simulator_name = "igibson"          

conds = Conditions(behavior_activity, activity_definition, simulator_name)
scope = get_object_scope(conds)
backend = None                      # TODO pass in backend from iGibson 
init = get_initial_conditions(conds, backend)
populated_scope = None              # TODO populate scope in iGibson, e.g. through sampling 
goal = get_goal_conditions(conds, backend, populated_scope)
ground = get_ground_goal_state_options(conds, backend, populated_scope)

print("####### Initial #######")
print(init)
print()
print("####### Goal #######")
print(goal)
print()
print("####### Ground #######")
print(ground)
print()

print("####### Goal evaluation #######")
for __ in range(100):
    print(evaluate_goal_conditions(goal))