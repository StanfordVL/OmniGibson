from bddl.activity import Conditions
import bddl.bddl_verification as ver
import re
from bddl.config import get_definition_filename
from bddl.utils import flatten_list
import sys
from pathlib import Path
import os


def get_goal_inroom_insts_cats(activity):
    conds = Conditions(activity, 0, "omnigibson")

    # Check if inroom insts appear in the goal
    inroom_insts = set([init_cond[1] for init_cond in conds.parsed_initial_conditions if init_cond[0] == "inroom"])
    print(inroom_insts)
    # Inroom insts that appear in the goal
    # goal_inroom_insts = [term for term in list(flatten_list(conds.parsed_goal_conditions)) if term in inroom_insts]
    goal_terms = set([term.strip("?") for term in flatten_list(conds.parsed_goal_conditions)])
    print(goal_terms)

    goal_inroom_insts = inroom_insts.intersection(goal_terms)
    
    # Inroom insts' cats that appear in the goal
    inroom_inst_cats = set([re.match(ver.OBJECT_CAT_AND_INST_RE, inroom_inst).group(0) for inroom_inst in inroom_insts])
    # goal_inroom_cats = [term for term in list(flatten_list(conds.parsed_goal_conditions)) if term in inroom_cats]
    goal_inroom_inst_cats = inroom_inst_cats.intersection(goal_terms)

    return goal_inroom_insts, goal_inroom_inst_cats 


def add_starred_instance(activity):
    # Given an activity, add a <cat>_* for EVERY cat belonging to an instance that both 1) appears in an inroom and 2) appears in the goal in its instance form (not cat form)
    conds = Conditions(activity, 0, "omnigibson")
    goal_inroom_insts, __ = get_goal_inroom_insts_cats(activity)
    print(goal_inroom_insts)
    goal_inroom_inst_cats = set([re.match(ver.OBJECT_CAT_AND_INST_RE, inroom_inst).group(0) for inroom_inst in goal_inroom_insts])
    print(goal_inroom_inst_cats)
    for cat in goal_inroom_inst_cats:
        conds.parsed_objects[cat].append(cat + "_*")

    # Construct the new (:objects ...) section
    objects_section = "(:objects\n"
    for category, objs in conds.parsed_objects.items():
        objects_section += "        " + " ".join(objs) + " - " + category + "\n"
    objects_section += "    )"
    print(objects_section)

    with open(get_definition_filename(activity, 0), "r") as f: 
        problem_text = f.read() 

    problem_text = re.sub(r'\(:objects[\s\S]*?\)', objects_section, problem_text, flags=re.MULTILINE)

    with open(get_definition_filename(activity, 0), "w") as f:
        f.write(problem_text)


if __name__ == "__main__":
    # bddl_fn = sys.argv[1]
    # activity = Path(bddl_fn).parts[-2]
    # add_starred_instance(activity)

    for activity in sorted(os.listdir(ver.PROBLEM_FILE_DIR)):
        if "-" in activity: continue 
        if not os.path.isdir(os.path.join(ver.PROBLEM_FILE_DIR, activity)): continue
        goal_inroom_insts, goal_inroom_cats = get_goal_inroom_insts_cats(activity)
        if len(goal_inroom_cats) > 0 or len(goal_inroom_insts) > 0:
            add_starred_instance(activity)