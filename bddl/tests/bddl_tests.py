import re
import os
import copy
import json
import pandas as pd
import sys
import glob
import csv
from collections import Counter

from bddl.parsing import parse_problem, parse_domain
import bddl.activity
from bddl_debug_backend import *
import test_utils


# Predicates

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


# Checkers

def object_list_correctly_formatted(defn):
    '''
    Verify that object list has the following properties for each line: 
    - Each ending term matches category regex
    - Penultimate term is `-`
    - First through third-to-last terms match instance regex
    - First through third-to-last terms are of the same category as the ending term
    '''
    objects_section = defn.split("(:objects\n")[-1].split("\n    )")[0].split("\n")
    for line in objects_section:
        elements = line.strip().split(" ")
        category = elements[-1]
        assert re.match(test_utils.OBJECT_CAT_RE, category) is not None, f"Malformed category at end of object section line: {category}"
        assert elements[-2] == "-", f"There should be a hyphen but instead there is {elements[-2]}"
        for inst in elements[:-2]:
            assert re.match(test_utils.OBJECT_INSTANCE_RE, inst) is not None, f"Malformed instance {inst}"
            assert category == re.match(test_utils.OBJECT_CAT_AND_INST_RE, inst).group(), f"Mismatched category and object: {category} and {inst}"


def all_objects_appropriate(objects, init, goal):
    '''
    Checks the following: 
    1. All objects in :objects are in :init
    2. All objects in :init are in :objects (so basically, the set of objects in :objects and 
        the set of objects in :init are equivalent)
    3. The set of object instances and categories in :goal is a subset of the set of object 
        instances and categories in :objects
    4. There are no categories in :init
    '''
    instances, categories = test_utils._get_objects_from_object_list(objects)
    init_insts = test_utils._get_instances_in_init(init)
    
    assert init_insts.issubset(instances), f":init has object instances not in :objects: {init_insts.difference(instances)}"
    assert instances.issubset(init_insts), f":objects has object instances not in :init: {instances.difference(init_insts)}"
    
    goal_objects = test_utils._get_objects_in_goal(goal)
    assert goal_objects.issubset(categories.union(instances)), f":goal has objects not in :objects: {goal_objects.difference(categories.union(instances))}"


def all_objects_placed(init): 
    '''
    Make sure everything is placed relative to a ROOM, even transitively.
    This is because when I hand-edit, I may accidentally not put every new scene object in an inroom.
    Meanwhile, it's unlikely that TODO fill in whatever else was I thinking of that this function won't check
        but is unlikely lol.
        - Appropriate placements for the object type - I'm unlikely to mess that up if I do say so myself

    NOTE should only be executed AFTER all_objects_appropraite
    '''
    insts = test_utils._get_instances_in_init(init)
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
                if (formula[0] == test_utils.FUTURE_PREDICATE and formula[1] == inst) or ((formula[0] in test_utils.PLACEMENTS) and (formula[1] == inst) and ((formula[2] in test_utils.ROOMS) or (formula[2] in placed_insts))) or ((formula[0] in test_utils.SUBSTANCE_PLACEMENTS) and (formula[1] in placed_insts) and (formula[2] == inst)):
                    placed_insts.add(inst)
        saturated = old_placed_insts == placed_insts 
        old_placed_insts = copy.deepcopy(placed_insts)
    assert not placed_insts.difference(insts), "There are somehow placed insts that are not in the overall set of insts."
    assert placed_insts == insts, f"Unplaced object instances: {insts.difference(placed_insts)}"


def no_invalid_synsets(objects, init, goal, syns_to_props):
    instances, categories = test_utils._get_objects_from_object_list(objects)
    init_insts = test_utils._get_instances_in_init(init)
    goal_objects = test_utils._get_objects_in_goal(goal)
    object_insts = set([re.match(test_utils.OBJECT_CAT_AND_INST_RE, inst).group() for inst in instances.union(init_insts).union(goal_objects)])
    object_terms = object_insts.union(categories)
    for proposed_syn in object_terms: 
        assert (proposed_syn in syns_to_props) or (proposed_syn == "agent.n.01"), f"Invalid synset: {proposed_syn}"


def no_invalid_predicates(init, goal, domain_predicates):
    atoms = []
    for literal in init: 
        atoms.append(literal[1] if literal[0] == "not" else literal)
    atoms.extend(test_utils._get_atoms_in_goal(goal))
    for atom in atoms: 
        assert atom[0] in domain_predicates, f"Invalid predicate: {atom[0]}" 
        

# Check uncontrolled categories

def future_and_real_present(objects, init, goal): 
    init_objects = test_utils._get_instances_in_init(init)
    future_objects = set([literal[1] for literal in init if literal[0] == "future"])
    real_objects = set([expression[1].strip("?") for expression in goal if expression[0] == "real"])
    for expression in goal:
        if expression[0] == "or":
            for disjunct in expression[1:]:
                if disjunct[0] == "real":
                    real_objects.add(disjunct[1].strip("?"))
    assert future_objects.issubset(real_objects) or (("washer.n.03" in objects) and future_objects.difference(real_objects) == set(["water.n.06_1"])), f"{future_objects.difference(real_objects)} in future clauses but not real clauses (and doesn't satisfy washer/water exception)"
    assert real_objects.issubset(future_objects.union(init_objects)), f"{real_objects.difference(future_objects)} in real clauses but not future clauses or init"


def no_repeat_object_lines(defn): 
    from pprint import pprint
    objects_lines = [line.strip() for line in defn.split("(:objects")[1].split("(:init")[0].split("\n")[1:-3]]
    seen_cats = set() 
    for objects_line in objects_lines:
        if objects_line == ")":
            continue
        *__, __, cat = objects_line.split(" ")
        assert cat not in seen_cats, f"Category repeated: {cat}"
        seen_cats.add(cat)


def no_qmarks_in_init(defn): 
    init = defn.split("(:init")[1].split("(:goal")[0]
    assert "?" not in init, "Inappropriate ? in :init."


def no_contradictory_init_atoms(init):
    for literal in init: 
        if literal[0] == "not":
            assert literal[1] not in init, f"Contradictory init statements: {literal[1]}"


def no_uncontrolled_category(activity, defn):
    conds = bddl.activity.Conditions(activity, 0, "omnigibson", predefined_problem=defn)
    scope = bddl.activity.get_object_scope(conds)
    bddl.activity.get_initial_conditions(conds, DebugBackend(), scope, generate_ground_options=False)
    # Pretend scope has been filled 
    for name in scope: 
        scope[name] = DebugGenericObject(name, DebugSimulator())
    bddl.activity.get_goal_conditions(conds, DebugBackend(), scope, generate_ground_options=False)


def agent_present(init):
    for literal in init: 
        if (literal[0] == "ontop") and (literal[1] == "agent.n.01_1"):
            break
    else:
        raise AssertionError("Agent not present.")


def problem_name_correct(activity, definition_id=0):
    defn_fn = os.path.join(test_utils.PROBLEM_FILE_DIR, activity, f'problem{definition_id}.bddl')
    with open(defn_fn, "r") as f:
        problem_name, *__ = parse_problem(activity, 0, "omnigibson", predefined_problem=f.read())
    assert (problem_name == f"{activity}-{definition_id}") or (problem_name == f"{activity.lower()}-{definition_id}"), f"Wrong problem name '{problem_name}' for activity '{activity}'"


def no_misaligned_synsets_predicates(init, goal, syns_to_props):
    for literal in init: 
        init_atom = literal[1] if literal[0] == "not" else literal
        test_utils.check_synset_predicate_alignment(init_atom, syns_to_props)
    goal_atoms = test_utils._get_atoms_in_goal(goal)
    for goal_atom in goal_atoms:
        test_utils.check_synset_predicate_alignment(goal_atom, syns_to_props)


def no_unnecessary_specific_containers(objects, init, goal, syns_to_props):
    specific_fillable_containers = [obj_cat for obj_cat in objects.keys() if obj_cat != "agent.n.01" and "fillable" in syns_to_props[obj_cat] and test_utils.is_specific_container_synset(obj_cat)]
    
    atoms = []
    for literal in init: 
        atoms.append(literal[1] if literal[0] == "not" else literal)
    goal_atoms = [[term.strip("?") for term in goal_atom] for goal_atom in test_utils._get_atoms_in_goal(goal)]
    atoms.extend(goal_atoms)
    fill_atoms = [atom for atom in atoms if (atom[0] in ["filled", "contains", "insource", "inside"])]

    for specific_fillable_container in specific_fillable_containers:
        for atom in fill_atoms: 
            # print(atom)
            if (atom[0] in ["filled", "contains", "insource"]) and (re.match(test_utils.OBJECT_CAT_AND_INST_RE, atom[1]).group() == specific_fillable_container):
                break 
            if (atom[0] == "inside") and (re.match(test_utils.OBJECT_CAT_AND_INST_RE, atom[2]).group() == specific_fillable_container):
                break
        else:
            raise AssertionError(f"Substance-specific fillable container {specific_fillable_container} that does not fill/contain anything/have anything inside. Switch to container__of version.")


def no_substances_with_multiple_instances(objects, syns_to_props):
    for cat, insts in objects.items(): 
        if "substance" in syns_to_props[cat]:
            assert len(insts) == 1, f"Substance {cat} with {len(insts)} instances instead of 1"


# MAIN 

def verify_definition(activity, syns_to_props, domain_predicates, csv=False):
    defn_fn = os.path.join(test_utils.PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read() 
    __, objects, init, goal = test_utils._get_defn_elements_from_file(activity)
    object_list_correctly_formatted(defn)
    all_objects_appropriate(objects, init, goal)
    all_objects_placed(init)
    future_and_real_present(objects, init, goal)
    no_repeat_object_lines(defn)
    no_qmarks_in_init(defn)
    no_contradictory_init_atoms(init)
    no_invalid_synsets(objects, init, goal, syns_to_props)
    no_invalid_predicates(init, goal, domain_predicates)
    no_uncontrolled_category(activity, defn)
    no_misaligned_synsets_predicates(init, goal, syns_to_props)
    no_unnecessary_specific_containers(objects, init, goal, syns_to_props)
    no_substances_with_multiple_instances(objects, syns_to_props)
    agent_present(init)
    problem_name_correct(activity)
    if csv:
        no_filled_in_tm_recipe_goal(activity)
        sync_csv(activity)


# Master planning sheet
def batch_verify_all(csv=False): 
    with open(test_utils.SYNS_TO_PROPS_JSON, "r") as f:
        syns_to_props = json.load(f) 
    *__, domain_predicates = parse_domain("omnigibson")
    for activity in sorted(os.listdir(test_utils.PROBLEM_FILE_DIR)):
        if "-" in activity: continue
        if not os.path.isdir(os.path.join(test_utils.PROBLEM_FILE_DIR, activity)): continue
        print()
        print(activity)
        if os.path.exists(os.path.join(test_utils.CSVS_DIR, activity + ".csv")):
            try:
                verify_definition(activity, syns_to_props, domain_predicates, csv=csv)
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
            verify_definition(activity, syns_to_props, domain_predicates, csv=False)


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
    defn_fn = os.path.join(test_utils.PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read()
    goal_section = defn.split(":goal")[-1]
    assert "filled" not in goal_section, "filled in TM BDDL :goal"

    csv = os.path.join(test_utils.CSVS_DIR, activity + ".csv")
    with open(csv, "r") as f:
        lines = list(f.readlines())
    container_lines = [lines[i + 1] for i in range(len(lines) - 1) if "container," in lines[i]]
    for container_line in container_lines:
        assert "filled" not in container_line, f"filled in TM CSV container line: {container_line}"


def sync_csv(activity):
    csv = os.path.join(test_utils.CSVS_DIR, activity + ".csv")

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

    __, objects, init, _ = test_utils._get_defn_elements_from_file(activity)
    bddl_objs, _ = test_utils._get_objects_from_object_list(objects)
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
    for fname in os.listdir(test_utils.CSVS_DIR):
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
