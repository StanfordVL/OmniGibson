import re
import os
import copy
import json
import pandas as pd
import sys
import glob
import csv
from collections import Counter

from bddl.parsing import parse_problem, construct_bddl_from_parsed
import bddl.activity
from bddl_debug_backend import DebugBackend, DebugGenericObject

PROBLEM_FILE_DIR = "../bddl/activity_definitions"
PROPS_TO_SYNS_JSON = "../bddl/generated_data/properties_to_synsets.json"
SYNS_TO_PROPS_JSON = "../bddl/generated_data/propagated_annots_canonical.json"
CSVS_DIR = "tm_csvs"


# PREDICATES

def is_water_sourced(objects, init, goal):
    obj_list = []
    for obj_cat in objects:
        obj_list.extend(objects[obj_cat])
    objs = set(obj_list)
    # NOTE Assumes that if water is in goal, it'll be in objects
    if "water.n.06_1" in objs:
        # Find a placement for the water in the init
        for atom in init: 
            # match = re.match(r'^\((filled|covered) water\.n\.06_1', atom) is not None
            placed = (atom[0] in set(["filled", "covered"])) and (atom[1] == "water.n.06_1")
            # match_insource = re.match(r'^\(insource water\.n\.06_1 sink\.n\.01_[0-9]+\)$', atom) is not None
            insourced = (atom[0] == "insource") and (atom[1] == "water.n.06_1") and (re.match(r"sink\.n\.01_[0-9]+", atom[2]) is not None)
            if placed:
                # Make sure the container is in the object list 
                container = atom[-1]
                if container in objs: 
                    print("water was in container")
                    break 
                else:
                    # print(container)
                    # print(objs)
                    raise ValueError("Water is filling/covering something in the init, but that thing is not in the objects section.")
            if insourced:
                # Make sure the sink is in the object list
                sink = atom[-1]
                if sink not in objs:
                    raise ValueError("Water is insource a sink, but that sink is not in the objects section.")
                # Make sure the sink is in a room
                for atom in init: 
                    if (atom[0] == "inroom" and atom[1] == sink):
                        print("water was in sink and sink was in room")
                        return True
                else:
                    raise ValueError("Water is insource a sink and sink is in objects section, but not in an inroom")
    else:
        print("no water")
    return True


# OBJECTS

# Constants

OBJECT_INSTANCE_RE = r"[A-Za-z-_]+\.n\.[0-9]+_[0-9]+"
OBJECT_CAT_RE = r"[A-Za-z-_]+\.n\.[0-9]+$"
OBJECT_CAT_AND_INST_RE = r"[A-Za-z-_]+\.n\.[0-9]+"
SINGLE_CAT_QUANTS = ["forall", "exists", "forn"]
DOUBLE_CAT_QUANTS = ["forpairs", "fornpairs"]
ROOMS = set([
    "kitchen", 
    "dining_room",
    "living_room", 
    "utility_room", 
    "empty_room", 
    "bedroom", 
    "bathroom", 
    "garden", 
    "shared_office", 
    "corridor", 
    "classroom", 
    "grocery_store",
    "computer_lab",
    "playroom",
    "sauna",
    "childs_room",
    "garage",
    "closet",
    "storage_room",
    "entryway",
    "private_office",
    "meeting_room",
    "bar",
    "staircase",
    "spa",
    "television_room",
    "lobby"
])
PLACEMENTS = set([
    # "connected",
    "ontop", 
    "inside", 
    "under", 
    "filled", 
    "covered", 
    "overlaid", 
    "saturated", 
    "inroom", 
    "insource", 
    # "hung", 
    "future",
    "attached",
    "draped",
    "contains"
])
SUBSTANCE_PLACEMENTS = set(["saturated", "filled", "covered", "insource", "contains"])
FUTURE_SYNSET = "future"

# Helpers

def _traverse_goal_for_objects(expr, objects=None):
    objects = objects if objects is not None else set()
    # Check that category declarations in quantifiers are really categories, and equal
    if expr[0] in ["forall", "exists", "forpairs"]:
        term, __, cat = expr[1]
        assert term.strip("?") == cat, f"mismatched term and cat declaration: {term}, {cat}"
        assert re.match(OBJECT_CAT_RE, term.strip("?")) is not None, f"non-category term in quantifier declaration: {term}"
        if expr[0] in ["forpairs"]: 
            term, __, cat = expr[2]
            assert term.strip("?") == cat, f"mismatched term and cat declaration: {term}, {cat}"
            assert re.match(OBJECT_CAT_RE, term.strip("?")) is not None, f"non-category term in quantifier declaration: {term}"
        _traverse_goal_for_objects(expr[-1], objects=objects)
    if expr[0] in ["forn", "fornpairs"]:
        term, __, cat = expr[2]
        assert term.strip("?") == cat, f"mismatched term and cat declaration: {term}, {cat}"
        assert re.match(OBJECT_CAT_RE, term.strip("?")) is not None, f"non-category term in quantifier declaration: {term}"
        if expr[0] == "fornpairs": 
            term, __, cat = expr[3]
            assert term.strip("?") == cat, f"mismatched term and cat declaration: {term}, {cat}"
            assert re.match(OBJECT_CAT_RE, term.strip("?")) is not None, f"non-category term in quantifier declaration: {term}"
        _traverse_goal_for_objects(expr[-1], objects=objects)
    
    # Check the subexpr for atomic formulae in base case, else recurse 
    if type(expr[-1]) is not list: 
        for obj in expr[1:]:
            assert re.match(OBJECT_CAT_AND_INST_RE, obj.strip("?")) is not None, f"malformed object term in goal: {obj}"
            objects.add(obj.strip("?"))
    else: 
        if expr[0] in ["and", "or"]:
            for subexpr in expr[1:]:
                _traverse_goal_for_objects(subexpr, objects=objects)
        else:
            _traverse_goal_for_objects(expr[-1], objects=objects)


def _get_defn_elements_from_file(activity):
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, 'problem0.bddl')
    with open(defn_fn, "r") as f:
        __, objects, init, goal = parse_problem(activity, 0, "omnigibson", predefined_problem=f.read())
    return activity, objects, init, goal


def _get_objects_from_object_list(objects):
    instances, categories = set(), set()
    for cat, insts in objects.items():
        categories.add(cat)
        for inst in insts: 
            instances.add(inst)
    return instances, categories


def _get_instances_in_init(init):
    '''
    Take a parsed :init condition and return a set of all instances in it.
    '''
    init_insts = set()
    for literal in init: 
        formula = literal[1] if literal[0] == "not" else literal
        for inst in formula[1:]: 
            assert (re.match(OBJECT_INSTANCE_RE, inst) is not None) or (inst in ROOMS), f":init has category: {inst}" 
            if inst not in ROOMS:
                init_insts.add(inst)
    return init_insts


def _get_objects_in_goal(goal):
    goal_objects = set()
    goal = ["and"] + goal
    _traverse_goal_for_objects(goal, goal_objects)
    return goal_objects

def _get_unique_items_from_transition_map():
    obj_set = set()
    for fname in glob.glob(CSVS_DIR):
        with open(fname) as f:
            for row in f:
                first = row.split(',')[0]
                if '.n.' in first:
                    obj_set.add(first.rpartition('_')[0])

    obj_set.remove('')
    
    for obj in obj_set:
        print(obj)


# Checkers

def object_list_correctly_formatted(activity):
    '''
    Verify that object list has the following properties for each line: 
    - Each ending term matches category regex
    - Penultimate term is `-`
    - First through third-to-last terms match instance regex
    - First through third-to-last terms are of the same category as the ending term
    '''
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read()
    objects_section = defn.split("(:objects\n")[-1].split("\n    )")[0].split("\n")
    for line in objects_section:
        elements = line.strip().split(" ")
        category = elements[-1]
        assert re.match(OBJECT_CAT_RE, category) is not None, f"Malformed category at end of object section line: {category}"
        assert elements[-2] == "-", f"There should be a hyphen but instead there is {elements[-2]}"
        for inst in elements[:-2]:
            assert re.match(OBJECT_INSTANCE_RE, inst) is not None, f"Malformed instance {inst}"
            assert category == re.match(OBJECT_CAT_AND_INST_RE, inst).group(), f"Mismatched category and object: {category} and {inst}"


def all_objects_appropriate(activity):
    '''
    Checks the following: 
    1. All objects in :objects are in :init
    2. All objects in :init are in :objects (so basically, the set of objects in :objects and 
        the set of objects in :init are equivalent)
    3. The set of object instances and categories in :goal is a subset of the set of object 
        instances and categories in :objects
    4. There are no categories in :init
    '''
    __, objects, init, goal = _get_defn_elements_from_file(activity)
    instances, categories = _get_objects_from_object_list(objects)
    init_insts = _get_instances_in_init(init)
    
    assert init_insts.issubset(instances), f":init has object instances not in :objects: {init_insts.difference(instances)}"
    assert instances.issubset(init_insts), f":objects has object instances not in :init: {instances.difference(init_insts)}"
    
    goal_objects = _get_objects_in_goal(goal)
    assert goal_objects.issubset(categories.union(instances)), f":goal has objects not in :objects: {goal_objects.difference(categories.union(instances))}"


def all_objects_placed(activity): 
    '''
    Make sure everything is placed relative to a ROOM, even transitively.
    This is because when I hand-edit, I may accidentally not put every new scene object in an inroom.
    Meanwhile, it's unlikely that TODO fill in whatever else was I thinking of that this function won't check
        but is unlikely lol.
        - Appropriate placements for the object type - I'm unlikely to mess that up if I do say so myself

    NOTE should only be executed AFTER all_objects_appropraite
    '''
    __, objects, init, goal = _get_defn_elements_from_file(activity)
    insts = _get_instances_in_init(init)
    insts = set([inst for inst in insts if ["future", inst] not in init])

    # Make sure everything not set to `future` is placed relative to a ROOM
    placed_insts = set()
    old_placed_insts = set()
    saturated = False 
    while not saturated:
        for inst in insts:                 
            if inst in placed_insts:
                continue
            for literal in init: 
                formula = literal[1] if literal[0] == "not" else literal 
                # NOTE only uncomment below line suffix when dealing with situations where substance and object have been flipped
                if (formula[0] == FUTURE_SYNSET and formula[1] == inst) or ((formula[0] in PLACEMENTS) and (formula[1] == inst) and ((formula[2] in ROOMS) or (formula[2] in placed_insts))) or ((formula[0] in SUBSTANCE_PLACEMENTS) and (formula[1] in placed_insts) and (formula[2] == inst)):
                    placed_insts.add(inst)
        saturated = old_placed_insts == placed_insts 
        old_placed_insts = copy.deepcopy(placed_insts)
    assert not placed_insts.difference(insts), "There are somehow placed insts that are not in the overall set of insts."
    assert placed_insts == insts, f"Unplaced object instances: {insts.difference(placed_insts)}"


def all_synsets_valid(activity):
    with open(SYNS_TO_PROPS_JSON, "r") as f:
        syns_to_props = json.load(f)
    __, objects, init, goal = _get_defn_elements_from_file(activity)
    instances, categories = _get_objects_from_object_list(objects)
    init_insts = _get_instances_in_init(init)
    goal_objects = _get_objects_in_goal(goal)
    object_insts = set([re.match(OBJECT_CAT_AND_INST_RE, inst).group() for inst in instances.union(init_insts).union(goal_objects)])
    object_terms = object_insts.union(categories)
    for proposed_syn in object_terms: 
        assert (proposed_syn in syns_to_props) or (proposed_syn == "agent.n.01"), f"Invalid synset: {proposed_syn}"


def no_unused_scene_objects(activity):
    __, __, init, __ = _get_defn_elements_from_file(activity)
    inroomed_objects = [atom[1] for atom in init if "inroom" in atom]
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, 'problem0.bddl')
    with open(defn_fn, "r") as f: 
        defn = f.read() 
    for inroomed_object in inroomed_objects:
        inroom_cat = re.match(OBJECT_CAT_AND_INST_RE, inroomed_object).group()
        defn_no_objects = defn.split("(:init")[-1]
        if len(re.findall(inroom_cat, defn_no_objects)) + 0 <= 1:
            raise AssertionError(f"Potential unused scene object {inroomed_object}")
        

# Check uncontrolled categories

def future_and_real_present(activity): 
    __, objects, init, goal = _get_defn_elements_from_file(activity)
    init_objects = _get_instances_in_init(init)
    future_objects = set([literal[1] for literal in init if literal[0] == "future"])
    real_objects = set([expression[1].strip("?") for expression in goal if expression[0] == "real"])
    for expression in goal:
        if expression[0] == "or":
            for disjunct in expression[1:]:
                if disjunct[0] == "real":
                    real_objects.add(disjunct[1].strip("?"))
    assert future_objects.issubset(real_objects) or (("washer.n.03" in objects) and future_objects.difference(real_objects) == set(["water.n.06_1"])), f"{future_objects.difference(real_objects)} in future clauses but not real clauses (and doesn't satisfy washer/water exception)"
    assert real_objects.issubset(future_objects.union(init_objects)), f"{real_objects.difference(future_objects)} in real clauses but not future clauses or init"


def no_repeat_object_lines(activity): 
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read()
    # print(defn.split("(:objects")[1].split(":(init")[0].split("\n")[1:-4])
    from pprint import pprint
    objects_lines = [line.strip() for line in defn.split("(:objects")[1].split("(:init")[0].split("\n")[1:-3]]
    # pprint(objects_lines)
    seen_cats = set() 
    for objects_line in objects_lines:
        if objects_line == ")":
            continue
        *__, __, cat = objects_line.split(" ")
        assert cat not in seen_cats, f"Category repeated: {cat}"
        seen_cats.add(cat)


def no_qmarks_in_init(activity): 
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read() 
    init = defn.split("(:init")[1].split("(:goal")[0]
    assert "?" not in init, "Inappropriate ? in :init."


def no_contradictory_init_atoms(activity):
    __, __, init, __ = _get_defn_elements_from_file(activity)
    for literal in init: 
        if literal[0] == "not":
            assert literal[1] not in init, f"Contradictory init statements: {literal[1]}"


def no_uncontrolled_category(activity):
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, 'problem0.bddl')
    with open(defn_fn, "r") as f:
        defn=f.read()
    conds = bddl.activity.Conditions(activity, 0, "omnigibson", predefined_problem=defn)
    scope = bddl.activity.get_object_scope(conds)
    init = bddl.activity.get_initial_conditions(conds, DebugBackend(), scope, generate_ground_options=False)
    # Pretend scope has been filled 
    for name in scope: 
        scope[name] = DebugGenericObject(name)
    goal = bddl.activity.get_goal_conditions(conds, DebugBackend(), scope, generate_ground_options=False)


def agent_present(activity):
    __, __, init, __ = _get_defn_elements_from_file(activity)
    for literal in init: 
        if (literal[0] == "ontop") and (literal[1] == "agent.n.01_1"):
            break
    else:
        raise AssertionError("Agent not present.")


# MAIN 

def verify_definition(activity, csv=False):
    object_list_correctly_formatted(activity)
    all_objects_appropriate(activity)
    all_objects_placed(activity)
    future_and_real_present(activity)
    no_repeat_object_lines(activity)
    no_qmarks_in_init(activity)
    no_contradictory_init_atoms(activity)
    no_uncontrolled_category(activity)
    all_synsets_valid(activity)
    agent_present(activity)
    if csv:
        no_filled_in_tm_recipe_goal(activity)
        sync_csv(activity)


# Master planning sheet
def batch_verify_all(csv=False): 
    for activity in sorted(os.listdir(PROBLEM_FILE_DIR)):
        if "-" in activity: continue        # TODO deal with these directories
        if not os.path.isdir(os.path.join(PROBLEM_FILE_DIR, activity)): continue
        print()
        print(activity)
        if os.path.exists(os.path.join(CSVS_DIR, activity + ".csv")):
            try:
                verify_definition(activity, csv=csv)
            except FileNotFoundError:
                print()
                print("file not found for", activity)
                continue
            except AssertionError as e:
                print()
                print(activity)
                print(e)
                to_continue = input("continue? y/n: ")
                while to_continue != "y":
                    to_continue = input("continue? y/n: ")
                continue
        else:
            verify_definition(activity, csv=False)


def unpack_nested_lines(sec):
    '''
    takes in a list of lines such as init or goal
    returns non-nested sublines (i.e. unpacks forall statements, or statements, etc.) that describe object(s)
    does not preserve order, because order doesn't matter when scanning through all lines
    '''
    lines = sec.copy()
    res = []
    while lines:
        line = lines.pop(0)

        if type(line[1]) is list:
            for subline in line[1:]:
                if len(subline) >= 2 and '-' not in subline:
                    lines.append(subline)
        else:
            res.append(line)

    return res


# Transition maps

def no_filled_in_tm_recipe_goal(activity):
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read()
    goal_section = defn.split(":goal")[-1]
    assert "filled" not in goal_section, "filled in TM BDDL :goal"

    csv = os.path.join(CSVS_DIR, activity + ".csv")
    with open(csv, "r") as f:
        lines = list(f.readlines())
    container_lines = [lines[i + 1] for i in range(len(lines) - 1) if "container," in lines[i]]
    for container_line in container_lines:
        assert "filled" not in container_line, f"filled in TM CSV container line: {container_line}"


def sync_csv(activity):
    csv = os.path.join(CSVS_DIR, activity + ".csv")

    csv_objs = set()
    bddl_objs = set()
    bddl_ignore = set()

    with open(csv) as f:
        output_objs = []
        output_flag = False
        for row in f:
            first = row.split(',')[0]
            # collect output objects
            # remove if they weren't the last step
            if first == 'output-objects':
                output_flag = True
                # removing objects here allows us to exclude input objects which are outputs of the previous step
                csv_objs.difference_update(output_objs)
                output_objs = []
            if '.n.' in first:
                csv_objs.add(first)
                if output_flag == True:
                    output_objs.append(first)
            if first == "tool":
                output_flag = False 

    csv_objs.discard('')

    __, objects, init, _ = _get_defn_elements_from_file(activity)
    bddl_objs, _ = _get_objects_from_object_list(objects)
    # print(bddl_objs)
    for literal in init: 
        formula = literal[1] if literal[0] == "not" else literal
        #things to ignore
        if formula[0] in ["insource", "filled"]:
            bddl_ignore.add(formula[1])
    things_to_ignore = [
        # can't put jar here because sometimes output container is a mason_jar
        "countertop",
        "bottle",
        "sack",
        "agent.n.01",
        "floor.n.01",
        "electric_refrigerator.n.01",
        "cabinet.n.01",
        "tupperware.n.01"
    ]
    for obj in list(bddl_objs):
        for thing in things_to_ignore:
            if thing in obj:
                bddl_objs.remove(obj)

    bddl_objs = bddl_objs - bddl_ignore 
    
    assert len(csv_objs - bddl_objs) == 0 and len(bddl_objs - csv_objs) == 0, f"Items in csv but not bddl: {csv_objs - bddl_objs} \nItems in bddl but not csv: {bddl_objs - csv_objs}"


def batch_sync_csv():
    for fname in os.listdir(CSVS_DIR):
        activity = fname[:-len(".csv")]
        print()
        print(activity)
        try:
            sync_csv(activity)
        except FileNotFoundError:
            print()
            print("file not found for", activity)
            continue
        except AssertionError as e:
            print()
            print(activity)
            print(e)
            to_continue = input("continue? y/n: ")
            while to_continue != "y":
                to_continue = input("continue? y/n: ")
            continue
        


def main():
    if sys.argv[1] == "verify": 
        verify_definition(sys.argv[2])
    
    elif sys.argv[1] == "verify_csv":
        verify_definition(sys.argv[2], csv=True)
        
    elif sys.argv[1] == "batch_verify": 
        batch_verify_all()

    elif sys.argv[1] == "batch_verify_csv": 
        batch_verify_all(csv=True)

    elif sys.argv[1] == "sync_csv":
        sync_csv(sys.argv[2])

    elif sys.argv[1] == "batch_sync_csv":
        batch_sync_csv()


if __name__ == "__main__":
    main()
