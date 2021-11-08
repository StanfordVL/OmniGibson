from activity import get_conditions

behavior_activity = "packing_lunches"
activity_definition = 0
simulator_name = "igibson"

init, goal, ground = get_conditions(behavior_activity, 
                                    activity_definition,
                                    simulator_name)

print("####### Initial #######")
print(init)
print()
print("####### Goal #######")
print(goal)
print()
print("####### Ground #######")
print(ground)
