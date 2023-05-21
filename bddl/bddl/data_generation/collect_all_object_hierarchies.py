import os 
import json 
import pandas 

HIERARCHY_DIR = "hierarchy_per_activity"
CURRENTLY_ASSIGNED_ACTIVITIES = "../house_room_info/currently_assigned_activities.json"

def create_save_all_activity_hierarchy_dict():
    all_hierarchies = {}
    with open(CURRENTLY_ASSIGNED_ACTIVITIES, "r") as f:
        curr_acts = set(json.load(f))
    activity_fns = [activity_fn for activity_fn in os.listdir(HIERARCHY_DIR) if 'json' in activity_fn]
    for activity_fn in activity_fns:
        activity = activity_fn.split(".")[0]
        if (activity in curr_acts) or (activity == "arranging-home"):   # Either a currently assigned activity, or the default
            with open(os.path.join(HIERARCHY_DIR, activity_fn), 'r') as f:
                activity_hierarchy = json.load(f)
                all_hierarchies[activity_fn.split('.')[0]] = activity_hierarchy

    with open('all_activity_hierarchies.json', 'w') as f:
        json.dump(all_hierarchies, f, indent=4)