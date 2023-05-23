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

PROBLEM_FILE_DIR = "../activity_definitions"
PROPS_TO_SYNS_JSON = "../generated_data/properties_to_synsets.json"
SYNS_TO_PROPS_JSON = "../generated_data/propagated_annots_canonical.json"

# PREDICATES

def is_water_sourced(objects, init, goal):
    obj_list = []
    for obj_cat in objects:
        obj_list.extend(objects[obj_cat])
    objs = set(obj_list)
    # print("OBJS:", objs)
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


def detect_double_and(init, goal):
    pass
    
# Double open parentheses - NOTE doing this by hand because I would need the compiled version to get all of them
#   and I don't want to deal with it
# def detect_double_open_parens_after_and(goal):
#     for subexpr in goal: 
#         if subexpr[0] == "and":
#             # Check if there's only one child and it's a list
#             if len(subexpr) == 2:

#     return False

# )( - NOTE didn't actually find any


def switch_subst_non_subst(act, init, goal):
    substance_placements = ["saturated", "filled", "contains", "covered"]
    for sec in [init, goal]:
        for line in sec:
            pred, *objs = line 
            if pred in substance_placements:
                obj1, obj2 = objs 
                line[1] = obj2
                line[2] = obj1
    return init, goal


# TODO assert no hungs in :init - well actually, deal with this once 
#   simulator team has gotten back with a design


def melted_subst_state_to_synset(activity):
    '''
    Converts objects in melted state in init to melted__* substances. 
        Since if it was in melted state in init it was a substance, fingers 
        crossed that there was only one instance
        NOTE: this does NOT change the uses to substance-appropriate ones. 
            Too complicated.
    Converts objects in melted state in goal to melted__* substances

    TODO this isn't done. It's easier to change by hand because of containers.
    '''
    __, objects, init, goal = get_defn_elements_from_file(activity)
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, 'problem0.bddl')
    with open(defn_fn, "r") as f: 
        defn = f.read()
    for literal in init: 
        if literal[0] == "melted": 
            melting_subst = literal[1]
            # Replace the object with melted__object
            melting_cat = instance_to_cat(melting_subst)
            melted_syn_defn_raw = defn.replace(melting_cat, f"melted__{melting_cat}")
            melted_syn_defn = melted_syn_defn_raw.replace(f"(melted {melting_subst})", "")
    with open(defn_fn, "w") as f:
        f.write(melted_syn_defn)
    

def sliced_state_to_synset(activity, synset, replacement, first_appearance): 
    '''
    For all objects in a sliced state: 
        If replacement is sliced: 
            If sliced in init: change to sliced synset
            If sliced in goal: add sliced synset to objects, add future statement, 
                add real statement, replace synset in goal with sliced synset
        If replacement is diced: 
            Same thing as sliced, but diced. 
            Will also need to change states to substance-appropriate states, but that
                needs to be manual.

    NOTE: various decisions need to be made manually, so this will be called as a tool 
        during manual review
    
    Doesn't removed `sliced` lines, it's too complicated. Will do manually
    '''
    __, objects, init, goal = get_defn_elements_from_file(activity)
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, 'problem0.bddl')
    with open(defn_fn, "r") as f:
        defn = f.read() 
    if first_appearance == "init": 
        # Replace synset completely (it'll never go back to being whole)
        defn = defn.replace(synset, f"{replacement}__{synset}")
        # Remove (sliced synset) line
        defn = defn.replace(f"(sliced {replacement}__{synset})", "")
    elif first_appearance == "goal":
        # Add object line
        if replacement == "diced":
            instances_string = f"{replacement}__{synset}_1"
        elif replacement == "half": # TODO need to double them!
            num_instances = len(objects[synset]) * 2
            instances = [f"{replacement}__{synset}_{index + 1}" for index in range(num_instances)]
            instances_string = " ".join(instances)
        else:
            raise ValueError("Invalid replacement parameter.")
        defn = defn.replace(
            f"- {synset}\n",
            f"- {synset}\n        {instances_string} - {replacement}__{synset}\n"
        )

        # Add future lines
        if replacement == "diced": 
            future_section = f"        (future {replacement}__{synset}_1)"
        elif replacement == "half": 
            num_instances = len(objects[synset]) * 2
            future_section = ""
            for index in range(num_instances): 
                future_section += f"        (future {replacement}__{synset}_{index + 1})\n"
        else:
            raise ValueError("Invalid replacement parameter.")
        defn = defn.replace("        (ontop agent.n.01_1", f"{future_section}\n        (ontop agent.n.01_1")

        # Replace synset in goal section
        defn_parts = defn.split("(:goal")
        defn_parts[1] = defn_parts[1].replace(synset, f"{replacement}__{synset}")
        defn = "(:goal".join(defn_parts)

        # Add real lines
        if replacement == "diced": 
            real_section = f"            (real ?{replacement}__{synset}_1)\n"
        elif replacement == "half": 
            num_instances = len(objects[synset]) * 2
            real_section = ""
            for index in range(num_instances): 
                real_section += f"            (real ?{replacement}__{synset}_{index + 1})\n"
        else:
            raise ValueError("Invalid replacement parameter.")
        defn = defn.replace("(:goal \n        (and \n", f"(:goal \n        (and \n{real_section}")

    else:
        raise ValueError("Invalid first_appearance parameter,") 
    
    with open(defn_fn, "w") as f:
        f.write(defn)


def add_goal_question_marks(activity): 
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, 'problem0.bddl')
    with open(defn_fn, "r") as f:
        defn = f.read() 
    defn_parts = defn.split("(:goal")
    defn_parts[1] = re.sub(r"([A-Za-z-_]+\.n\.[0-9]+)", r"?\1", defn_parts[1])
    defn_parts[1] = defn_parts[1].replace("- ?", "- ")
    defn_parts[1] = defn_parts[1].replace("??", "?")
    defn = "(:goal".join(defn_parts)
    with open(defn_fn, "w") as f:
        f.write(defn)


def hot_substance_state_to_synset(activity, substance_synset):
    '''
    Replaces (hot substance) with just a hot__substance synset
    Does per substance per activity, because sometimes it's more appropriate 
        to just remove. Even doing in a batch and then going through each one
        to remove would require looking at each one.
    If (hot substance) in init: 
        Replace synset everywhere with hot synset 
    If (hot substance) in goal and not in init: 
        Future/real hot synset
        Goal state still needs to be dealt with by hand, in case we want some of the 
            non-hot substance to still exist
    '''
    __, objects, init, goal = get_defn_elements_from_file(activity)
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, 'problem0.bddl')
    with open(defn_fn, "r") as f:
        defn = f.read()
    # Hot substance in in it
    if ["hot", substance_synset + "_1"] in init: 
        defn = defn.replace(substance_synset, f"hot__{substance_synset}")
        defn = defn.replace(f"\n        (hot hot__{substance_synset}_1)", "")
    elif ["hot", f"?{substance_synset}_1"] in goal: 
        with open(defn_fn, "r") as f: 
            defn = f.read()
        # Add hot synset as an object under non-hot synset
        defn = defn.replace(f"{substance_synset}_1 - {substance_synset}", f"{substance_synset}_1 - {substance_synset}\n        hot__{substance_synset}_1 - hot__{substance_synset}")
        # Add future atom to init
        defn = defn.replace("(ontop agent.n.01_1", f"(future hot__{substance_synset}_1)\n        (ontop agent.n.01_1")
        # Add real atom to goal
        defn_parts = defn.split("(and")
        defn_parts[1] = f"\n            (real ?hot__{substance_synset}_1)" + defn_parts[1]
        defn = "(and".join(defn_parts)
        # NOTE not replacing synset in goal because it's possible we don't want to do that everywhere, in case we want to keep some 
        #   non-hot substance
    with open(defn_fn, "w") as f:
        f.write(defn)


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

def traverse_goal_for_objects(expr, objects=None):
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
        traverse_goal_for_objects(expr[-1], objects=objects)
    if expr[0] in ["forn", "fornpairs"]:
        term, __, cat = expr[2]
        assert term.strip("?") == cat, f"mismatched term and cat declaration: {term}, {cat}"
        assert re.match(OBJECT_CAT_RE, term.strip("?")) is not None, f"non-category term in quantifier declaration: {term}"
        if expr[0] == "fornpairs": 
            term, __, cat = expr[3]
            assert term.strip("?") == cat, f"mismatched term and cat declaration: {term}, {cat}"
            assert re.match(OBJECT_CAT_RE, term.strip("?")) is not None, f"non-category term in quantifier declaration: {term}"
        traverse_goal_for_objects(expr[-1], objects=objects)
    
    # Check the subexpr for atomic formulae in base case, else recurse 
    if type(expr[-1]) is not list: 
        for obj in expr[1:]:
            assert re.match(OBJECT_CAT_AND_INST_RE, obj.strip("?")) is not None, f"malformed object term in goal: {obj}"
            objects.add(obj.strip("?"))
    else: 
        if expr[0] in ["and", "or"]:
            for subexpr in expr[1:]:
                traverse_goal_for_objects(subexpr, objects=objects)
        else:
            traverse_goal_for_objects(expr[-1], objects=objects)


def get_defn_elements_from_file(activity):
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, 'problem0.bddl')
    with open(defn_fn, "r") as f:
        __, objects, init, goal = parse_problem(activity, 0, "omnigibson", predefined_problem=f.read())
    return activity, objects, init, goal

def get_objects_from_object_list(objects):
    instances, categories = set(), set()
    for cat, insts in objects.items():
        categories.add(cat)
        for inst in insts: 
            instances.add(inst)
    return instances, categories

def instance_to_cat(instance):
    return "_".join(instance.split("_")[:-1])

def add_agent(activity, floor_instance_id=1):
    __, objects, init, goal = get_defn_elements_from_file(activity)
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read()
    assert "(ontop agent.n.01_1 floor.n.01_" not in defn, "Agent line already exists."
    if "floor.n.01" in objects: 
        max_floor_id = len(objects["floor.n.01"])
        if floor_instance_id <= max_floor_id:
            objects["agent.n.01"] = ["agent.n.01_1"]
            init.append(["ontop", "agent.n.01_1", f"floor.n.01_{floor_instance_id}"])
        elif floor_instance_id == max_floor_id + 1:
            confirm = ""
            while confirm not in ["y", "n"]:
                confirm = input(f"There are {max_floor_id} floors already, but you are asking for the next one. Do you want this? [y/n]: ")
            if confirm == "y": 
                room = input("Room: ")
                room_confirm = input(f"Confirm {room}? [y/n]: ")
                while room_confirm != "y": 
                    room = input("Room: ")
                    room_confirm = input(f"Confirm {room}? [y/n]: ")
                for literal in init: 
                    if (literal[0] == "inroom") and (re.match(r"floor\.n\.01_[0-9]+", literal[1]) is not None) and (literal[2] == room):
                        print("There is already a floor in that room. Pick a different room or add the agent to that floor.")
                        return 
                objects["floor.n.01"].append(f"floor.n.01_{max_floor_id + 1}")
                objects["agent.n.01"] = ["agent.n.01_1"]
                init.append(["inroom", f"floor.n.01_{max_floor_id + 1}", room])
                init.append(["ontop", "agent.n.01_1", f"floor.n.01_{max_floor_id + 1}"])
            else:
                print("Then please try this function again.")
                return 
        else:
            raise ValueError("Given floor_instance_id is higher than max floor id + 1")
    else:
        room = input("Room: ")
        room_confirm = input(f"Confirm {room}? [y/n]: ")
        while room_confirm != "y": 
            room = input("Room: ")
            room_confirm = input(f"Confirm {room}? [y/n]: ")
        objects["agent.n.01"] = ["agent.n.01_1"]
        objects["floor.n.01"] = ["floor.n.01_1"]
        init.append(["inroom", f"floor.n.01_1", room])
        init.append(["ontop", "agent.n.01_1", f"floor.n.01_1"])

    agent_defn = construct_bddl_from_parsed(activity, 0, objects, init, goal)
    with open(defn_fn, "w") as f:
        f.write(agent_defn)


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


def get_instances_in_init(init):
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


def get_objects_in_goal(goal):
    goal_objects = set()
    goal = ["and"] + goal
    traverse_goal_for_objects(goal, goal_objects)
    return goal_objects


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
    __, objects, init, goal = get_defn_elements_from_file(activity)
    instances, categories = get_objects_from_object_list(objects)
    init_insts = get_instances_in_init(init)
    
    assert init_insts.issubset(instances), f":init has object instances not in :objects: {init_insts.difference(instances)}"
    assert instances.issubset(init_insts), f":objects has object instances not in :init: {instances.difference(init_insts)}"
    
    goal_objects = get_objects_in_goal(goal)
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
    __, objects, init, goal = get_defn_elements_from_file(activity)
    insts = get_instances_in_init(init)
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
    __, objects, init, goal = get_defn_elements_from_file(activity)
    instances, categories = get_objects_from_object_list(objects)
    init_insts = get_instances_in_init(init)
    goal_objects = get_objects_in_goal(goal)
    object_terms = instances.union(categories).union(init_insts).union(goal_objects)
    for term in object_terms: 
        proposed_syn = "_".join(term.split("_")[:-1])
        assert proposed_syn in syns_to_props, f"Invalid synset: {proposed_syn}"


def no_substance_container_synsets(activity):
    __, objects, init, goal = get_defn_elements_from_file(activity)
    instances, categories = get_objects_from_object_list(objects)
    init_insts = get_instances_in_init(init)
    goal_objects = get_objects_in_goal(goal)
    object_terms = instances.union(categories).union(init_insts).union(goal_objects)
    substance_container_synsets = set([
        object_term for object_term in object_terms if "__container" in object_term
    ])
    assert not substance_container_synsets, f"Substance container synsets present: {substance_container_synsets}"


def no_unused_scene_objects(activity):
    __, objects, init, goal = get_defn_elements_from_file(activity)
    instances, __ = get_objects_from_object_list(objects)
    inroomed_objects = [atom[1] for atom in init if "inroom" in atom]
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, 'problem0.bddl')
    with open(defn_fn, "r") as f: 
        defn = f.read() 
    for inroomed_object in inroomed_objects:
        inroomed_cat = "_".join(inroomed_object.split("_")[:-1]) + " "
        if len(re.findall(inroomed_object, defn)) + \
                    len(re.findall(inroomed_cat, defn)) == 3:
            raise AssertionError(f"Potential unused scene object {inroomed_object}")


# Check uncontrolled categories (get this from bddl repo)

def future_and_real_present(activity): 
    __, objects, init, goal = get_defn_elements_from_file(activity)
    init_objects = get_instances_in_init(init)
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
    __, __, init, __ = get_defn_elements_from_file(activity)
    for literal in init: 
        if literal[0] == "not":
            assert literal[1] not in init, f"Contradictory init statements: {literal[1]}"


def no_broken_predicate(activity): 
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read() 
    assert "(broken " not in defn, "`broken` predicate used" 


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
    __, __, init, __ = get_defn_elements_from_file(activity)
    for literal in init: 
        if (literal[0] == "ontop") and (literal[1] == "agent.n.01_1"):
            break
    else:
        raise AssertionError("Agent not present.")


def get_unique_items_from_transition_map():
    path = "comp_decomp_results/csvs/*.csv"
    obj_set = set()
    for fname in glob.glob(path):
        with open(fname) as f:
            for row in f:
                first = row.split(',')[0]
                if '.n.' in first:
                    obj_set.add(first.rpartition('_')[0])
                    # print(first.rpartition('_')[0])

    obj_set.remove('')
    
    for obj in obj_set:
        print(obj)


def check_tmrecipe_reduced_proportions():
    activities_to_check = []
    for activity in os.listdir("comp_decomp_results/csvs"):
        print()
        print(activity)
        with open(os.path.join("comp_decomp_results", "csvs", activity), "r") as f:
            reader = csv.reader(f)
            cats = []
            checking_inputs = False
            for line in reader: 
                if not len(line): continue
                if line[0] == "input-objects":
                    cats = []
                    checking_inputs = True
                elif line[0] == "output-objects":
                    checking_inputs = False
                    cats_counts = tuple(v for v in Counter(cats).values() if v > 1)
                    if len(cats_counts) > 1:
                        reduced = input(f"Is {cats_counts} reduced? [y/n]: ")
                        if reduced != "y":
                            activities_to_check.append([activity, reduced])
                elif checking_inputs and re.match(OBJECT_CAT_AND_INST_RE, line[0]) is not None: 
                    cats.append(re.match(OBJECT_CAT_AND_INST_RE, line[0]).group())
    print(activities_to_check)
    with open("tmp_activities_to_check.json", "w") as f:
        json.dump(activities_to_check, f)
    return activities_to_check


        # cats = [match.group() for match in [re.match(OBJECT_CAT_AND_INST_RE, line[0]) for line in reader if len(line)] if match is not None]
    
        # for line in reader: 
            # if re.match(OBJECT_INSTANCE_RE, line[0]) is not None: 
            #     cat = re.match(OBJECT_CAT_AND_INST_RE, line[0]).group()
            #     if cat in objects: 
            #         objects[cat].append(line[0])
            #     else: 
            #         objects[cat] = [line[0]]

    


# MAIN 

def verify_definition(activity, csv=False):
    object_list_correctly_formatted(activity)
    all_objects_appropriate(activity)
    all_objects_placed(activity)
    future_and_real_present(activity)
    no_repeat_object_lines(activity)
    no_qmarks_in_init(activity)
    no_contradictory_init_atoms(activity)
    # TODO uncomment after assemblies have been handled - may no longer be needed
    # no_broken_predicate(activity)
    no_uncontrolled_category(activity)
    # TODO uncomment the line below once all synsets have been added to final_propagated.json
    # all_synsets_valid(activity)
    # TODO uncomment the line below when ready to check for containers
    # no_substance_container_synsets(activity)
    # TODO uncomment after this function is fixed
    # no_unused_scene_objects(activity)
    agent_present(activity)
    if csv:
        no_filled_in_tm_recipe_goal(activity)
        sync_csv(activity)


# Master planning sheet
BATCH_DEFINITIONS_FILE = "b1k_master_planning.csv"

def batch_verify():
    plan_df = pd.read_csv(BATCH_DEFINITIONS_FILE)
    batch = plan_df["task_name"]
    for activity in batch: 
        print()
        print(activity)
        try:
            verify_definition(activity.replace(" ", "-"))
        except FileNotFoundError:
            continue
        except AssertionError as e:
            print()
            print(activity)
            print(e)
            continue

def batch_verify_all(): 

    plan_df = pd.read_csv(BATCH_DEFINITIONS_FILE)
    batch = plan_df["task_name"]
    for activity in batch: 
        print() 
        print(activity)
        try:
            verify_definition(activity.replace(" ", "-"))
        except FileNotFoundError:
            continue


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

def find_touching_deformables_or_substances(init, goal, sd):
    '''
    for a given activity, return all lines where 
    a deformable/substance is touching another deformable/substance
    a line in this case is a list containing 3 strings
    '''
    lines = []
    for sec in [init, goal]:
        unpacked_sec = unpack_nested_lines(sec)
        for line in unpacked_sec: 
            pred, *objs = line

            if pred in ['overlaid', 'ontop', 'touching']:
                obj1, obj2 = objs
                if 'sink' in obj2:
                    # hardcoded because a lot of things are overlaid on sink
                    # and ink is a substance which is a substring of sink
                    continue
                obj1_found = False
                obj2_found = False
                for item in sd:
                    if item in obj1:
                        obj1_found = True
                    if item in obj2:
                        obj2_found = True
                    if obj1_found and obj2_found:
                        lines.append(line)
                        break
    return lines


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
    # print(csv_objs)

    __, objects, init, _ = get_defn_elements_from_file(activity)
    bddl_objs, _ = get_objects_from_object_list(objects)
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


CSVS_DIR = "comp_decomp_results/csvs"
def batch_sync_csv():
    for fname in os.listdir(CSVS_DIR):
        activity = fname[:-len(".csv")]
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
            continue
        


def main():
    if sys.argv[1] == "main": 
        activities = os.listdir(PROBLEM_FILE_DIR)
        parsed_problems = []
        for activity in activities:
            # print()
            # print(activity)
            defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, 'problem0.bddl')
            with open(defn_fn, "r") as f:
                parsed_problem = parse_problem(activity, 0, 'omnigibson', predefined_problem=f.read())
                parsed_problems.append(parsed_problem)
        with open(PROPS_TO_SYNS_JSON, "r") as f:
            props_to_syns = json.load(f)
        substances_and_deformables = set(props_to_syns["substance"] + props_to_syns["deformable"])
        touching_sd_acts = []
        for act, objects, init, goal in parsed_problems:
            # print()
            # print(act)
            is_water_sourced(objects, init, goal)
            # touching_sd = find_touching_deformables_or_substances(init, goal, substances_and_deformables)
            # if touching_sd:
            #     touching_sd_acts.append(act)
            # if detect_double_open_parens_after_and(goal):
            #     print()
            #     print(goal)
        if touching_sd_acts:
            raise ValueError(f'The following activities contain lines where a substance/deformable is touching another: {touching_sd_acts}')

    elif sys.argv[1] == "verify": 
        verify_definition(sys.argv[2])
    
    elif sys.argv[1] == "verify_csv":
        verify_definition(sys.argv[2], csv=True)
    
    elif sys.argv[1] == "handle_sliced":
        sliced_state_to_synset(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    
    elif sys.argv[1] == "add_qmarks":
        add_goal_question_marks(sys.argv[2])
    
    elif sys.argv[1] == "hot_substance": 
        hot_substance_state_to_synset(sys.argv[2], sys.argv[3])

    elif sys.argv[1] == "transition_map":
        get_unique_items_from_transition_map()

    elif sys.argv[1] == "switch_subst":
        switch_subst_non_subst(sys.argv[2])
        
    elif sys.argv[1] == "batch_verify": 
        batch_verify_all()

    elif sys.argv[1] == "sync_csv":
        sync_csv(sys.argv[2])

    elif sys.argv[1] == "batch_sync_csv":
        batch_sync_csv()
    
    elif sys.argv[1] == "add_agent": 
        add_agent(sys.argv[2])
    
    elif sys.argv[1] == "check_proportion": 
        check_tmrecipe_reduced_proportions()


if __name__ == "__main__":
    main()
