from activity import Conditions, ObjectTaxonomy

behavior_activity = "packing_lunches"
activity_definition = 0
simulator_name = "igibson"

conds = Conditions(behavior_activity, activity_definition, simulator_name)

print("####### Initial #######")
print(conds.initial_conditions)
print()
print("####### Goal #######")
print(conds.goal_conditions)
print()
print("####### Ground #######")
print(conds.ground_goal_state_options)
