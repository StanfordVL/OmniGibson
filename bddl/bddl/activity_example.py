import bddl
from bddl.activity import Conditions, ObjectTaxonomy


bddl.set_backend("iGibson")
behavior_activity = "packing_lunches"
activity_definition = 0
simulator_name = "igibson"

conds = Conditions(behavior_activity, activity_definition, simulator_name)
scope = conds.get_object_scope()
populated_scope = None      # TODO populate scope in iGibson 
init = conds.get_initial_conditions()
goal = conds.get_goal_conditions(populated_scope)
ground = conds.get_ground_goal_state_options(populated_scope)


print("####### Initial #######")
print(init)
print()
print("####### Goal #######")
print(goal)
print()
print("####### Ground #######")
print(ground)
