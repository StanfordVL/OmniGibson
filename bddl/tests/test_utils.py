import json
import os
import re 

from bddl.generated_data.transition_map.tm_submap_params import TM_SUBMAPS_TO_PARAMS
from bddl.parsing import parse_problem, parse_domain

# Files

PROBLEM_FILE_DIR = "../bddl/activity_definitions"
PROPS_TO_SYNS_JSON = "../bddl/generated_data/properties_to_synsets.json"
SYNS_TO_PROPS_JSON = "../bddl/generated_data/propagated_annots_canonical.json"
CSVS_DIR = "tm_csvs"

# Constants

*__, domain_predicates = parse_domain("omnigibson")
UNARIES = [predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 1]
BINARIES = [predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 2]

VALID_ATTACHMENTS = set([
    ("mixing_bowl.n.01", "electric_mixer.n.01"),
    ("cork.n.04", "wine_bottle.n.01"),
    ("menu.n.01", "wall.n.01"),
    ("broken__light_bulb.n.01", "table_lamp.n.01"),
    ("light_bulb.n.01", "table_lamp.n.01"),
    ("lens.n.01", "digital_camera.n.01"),
    ("screen.n.01", "wall.n.01"),
    ("antler.n.01", "wall.n.01"),
    ("skateboard_wheel.n.01", "skateboard.n.01"),
    ("blackberry.n.01", "scrub.n.01"),
    ("raspberry.n.02", "scrub.n.01"),
    ("dip.n.07", "candlestick.n.01"),
    ("sign.n.02", "wall.n.01"),
    ("wreath.n.01", "wall.n.01"),
    ("bow.n.08", "wall.n.01"),
    ("holly.n.03", "wall.n.01"),
    ("curtain_rod.n.01", "wall.n.01"),
    ("bicycle.n.01", "bicycle_rack.n.01"),
    ("bicycle_rack.n.01", "wall.n.01"),
    ("dartboard.n.01", "wall.n.01"),
    ("rug.n.01", "wall.n.01"),
    ("fairy_light.n.01", "wall.n.01"),
    ("lantern.n.01", "wall.n.01"),
    ("address.n.05", "wall.n.01"),
    ("hanger.n.02", "wardrobe.n.01"),
    ("flagpole.n.02", "wall.n.01"),
    ("picture_frame.n.01", "wall.n.01"),
    ("wind_chime.n.01", "pole.n.01"),
    ("pole.n.01", "wall.n.01"),
    ("hook.n.05", "trailer_truck.n.01"),
    ("fire_alarm.n.02", "wall.n.01"),
    ("poster.n.01", "wall.n.01"),
    ("painting.n.01", "wall.n.01"),
    ("hanger.n.02", "coatrack.n.01"),
    ("license_plate.n.01", "car.n.01"),
    ("gummed_label.n.01", "license_plate.n.01"),
    ("wallpaper.n.01", "wall.n.01"),
    ("mirror.n.01", "wall.n.01"),
    ("webcam.n.02", "desktop_computer.n.01"),
    ("kayak.n.01", "kayak_rack.n.01"),
    ("kayak_rack.n.01", "wall.n.01"),
    ("fish.n.02", "fishing_rod.n.01"),
    ("bicycle_rack.n.01", "recreational_vehicle.n.01"),
])

VALID_ROOMS = set()

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
FUTURE_PREDICATE = "future"


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


def _traverse_goal_for_atoms(expr, goal_atoms):
    if all(type(subexpr) == str for subexpr in expr):
        goal_atoms.append(expr)
    elif expr[0] in ["and", "or"]:
        for subexpr in expr[1:]:
            _traverse_goal_for_atoms(subexpr, goal_atoms)
    elif expr[0] in ["forall", "exists", "forn", "forpairs", "fornpairs"]:
        _traverse_goal_for_atoms(expr[-1], goal_atoms)
    elif expr[0] == "imply":
        _traverse_goal_for_atoms(expr[1], goal_atoms)
        _traverse_goal_for_atoms(expr[2], goal_atoms)
    elif expr[0] == "not":
        _traverse_goal_for_atoms(expr[1], goal_atoms)
    else:
        raise ValueError(f"Unhandled logic operator {expr[0]}")


def _get_atoms_in_goal(goal):
    goal_atoms = []
    for goal_expr in goal:
        _traverse_goal_for_atoms(goal_expr, goal_atoms)
    return goal_atoms


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


def is_specific_container_synset(synset): 
    return "__" in synset and "__of__" not in synset and "diced__" not in synset and "cooked__" not in synset and "half__" not in synset


def check_synset_predicate_alignment(atom, syns_to_props):
    if atom[0] == "ontop" and atom[1] == "agent.n.01_1":
        return 

    pred, *object_insts = atom 
    objects = []
    for object_inst in object_insts: 
        syn_match = re.match(OBJECT_CAT_AND_INST_RE, object_inst.strip("?"))
        if syn_match is not None:
            objects.append(syn_match.group())
        elif True: # object_inst in VALID_ROOMS:    # TODO uncomment when VALID_ROOMS is populated
            if pred == "inroom":
                objects.append(object_inst)
            else:
                raise AssertionError(f"Nonsynset {object_inst} outside inroom")
        else:
            raise AssertionError(f"Invalid room {object_inst}")
    assert (pred in UNARIES) or (pred in BINARIES), f"Invalid predicate: {pred}"
    assert ((pred in UNARIES) and (len(objects) == 1)) or ((pred in BINARIES) and (len(objects) == 2)), f"Atom has wrong arity: {atom}"

    # Unaries
    if pred == "cooked": 
        assert "cookable" in syns_to_props[objects[0]], f"Inapplicable cooked: {atom}"
    if pred == "frozen": 
        assert "freezable" in syns_to_props[objects[0]], f"Inapplicable frozen: {atom}"
    if pred == "closed" or pred == "open":
        assert "openable" in syns_to_props[objects[0]], f"Inapplicable closed/open: {atom}"
    if pred == "folded" or pred == "unfolded":
        assert "drapeable" in syns_to_props[objects[0]], f"Inapplicable folded/unfolded: {atom}"
    if pred == "toggled_on":
        assert "toggleable" in syns_to_props[objects[0]], f"Inapplicable toggled_on: {atom}"
    if pred == "hot": 
        assert "heatable" in syns_to_props[objects[0]], f"Inapplicable hot: {atom}"
    if pred == "on_fire": 
        assert "flammable" in syns_to_props[objects[0]], f"Inapplicable on_fire: {atom}"
    if pred == "assembled": 
        assert "assembleable" in syns_to_props[objects[0]], f"Inapplicable assembled: {atom}"
    if pred == "broken": 
        assert "breakable" in syns_to_props[objects[0]], f"Inapplicable broken: {atom}"
    
    # Binaries
    if pred == "saturated":
        assert ("particleRemover" in syns_to_props[objects[0]]) and ("substance" in syns_to_props[objects[1]]), f"Inapplicable saturated: {atom}"
    if pred == "covered":
        assert ("nonSubstance" in syns_to_props[objects[0]]) and ("substance" in syns_to_props[objects[1]]), f"Inapplicable covered: {atom}"
    if pred == "filled":
        assert ("fillable" in syns_to_props[objects[0]]) and ("physicalSubstance" in syns_to_props[objects[1]]), f"Inapplicable filled/empty: {atom}"
    if pred == "contains" or pred == "empty":
        assert ("fillable" in syns_to_props[objects[0]]) and ("substance" in syns_to_props[objects[1]]), f"Inapplicable contains: {atom}"
    if pred == "ontop":
        assert ("nonSubstance" in syns_to_props[objects[0]]) and ("nonSubstance" in syns_to_props[objects[1]]), f"Inapplicable ontop: {atom}"
    if pred == "nextto":
        assert ("nonSubstance" in syns_to_props[objects[0]]) and ("nonSubstance" in syns_to_props[objects[1]]), f"Inapplicable nextto: {atom}"
    if pred == "under":
        assert ("nonSubstance" in syns_to_props[objects[0]]) and ("rigidBody" in syns_to_props[objects[1]]), f"Inapplicable under: {atom}"
    if pred == "touching": 
        assert ("rigidBody" in syns_to_props[objects[0]]) and ("rigidBody" in syns_to_props[objects[1]]), f"Inapplicable touching: {atom}"
    if pred == "inside": 
        assert ("nonSubstance" in syns_to_props[objects[0]]) and ("nonSubstance" in syns_to_props[objects[1]]), f"Inapplicable inside: {atom}"
    if pred == "overlaid": 
        assert ("drapeable" in syns_to_props[objects[0]]) and ("rigidBody" in syns_to_props[objects[1]]), f"Inapplicable overlaid: {atom}"
    if pred == "attached":
        assert tuple(objects) in VALID_ATTACHMENTS, f"Inapplicable attached: {atom}"
    if pred == "draped": 
        assert ("drapeable" in syns_to_props[objects[0]]) and ("rigidBody" in syns_to_props[objects[1]]), f"Inapplicable overlaid: {atom}"
    if pred == "insource": 
        assert (("particleSource" in syns_to_props[objects[0]]) or ("particleApplier" in syns_to_props[objects[0]])) and ("substance" in syns_to_props[objects[1]]), f"Inapplicable insource: {atom}"


def check_clashing_transition_rules():
    # Check within each submap
    for submap_name in TM_SUBMAPS_TO_PARAMS:
        with open(os.path.join("..", "bddl", "generated_data", "transition_map", "tm_jsons", submap_name + ".json"), "r") as f:
            submap = json.load(f)
        seen_object_sets = []
        for rule in submap: 
            # Get relevant parameters
            rule_name = rule.get("rule_name", "No name")
            input_objects = rule.get("input_objects", {})
            input_states = rule.get("input_states", {})
            input_states = input_states if input_states is not None else {}
            if submap_name == "heat_cook": 
                equipment = set([list(rule["heat_source"].keys())[0], list(rule["container"].keys())[0]])
            elif submap_name == "single_toggleable_machine":
                equipment = set([list(rule["machine"].keys())[0]])
            else:
                equipment = set()       # Equivalence will pass trivially when checked, because this rule already clashes
            output_objects = rule.get("output_objects", {})
            output_states = rule.get("output_states", {})
            output_states = output_states if output_states is not None else {}

            # NOTE doing input_objects.keys, not input_objects.items, because simulator is not actually sensitive to amount. It only checks for category, 
            #   so different amounts but same categories still need to result in the same output.
            
            # Collect all sets of input objects to check subsetting
            input_objects = set(sorted(input_objects.keys(), key=lambda x: x[0]))
            output_objects = set(sorted(output_objects.keys(), key=lambda x: x[0]))
            input_states_strs = set([syns + "@" + ";".join([f"{pred}:{val}" for pred, val in sorted(states, key=lambda x: x[0])]) for syns, states in sorted(input_states.items(), key=lambda x: x[0])])
            output_states_strs = set([syns + "@" + ";".join([f"{pred}:{val}" for pred, val in sorted(states, key=lambda x: x[0])]) for syns, states in sorted(output_states.items(), key=lambda x: x[0])])
            for seen_rule_name, seen_object_set, seen_states_set, seen_equipment, seen_output_objects, seen_output_states in seen_object_sets:
                # If we see that our next input objects set is a subset or superset...
                if input_objects.issuperset(seen_object_set) or input_objects.issubset(seen_object_set):
                    # Construct a set of atomic formulae in string format
                    if input_states_strs.issuperset(seen_states_set) or input_states_strs.issubset(seen_states_set):
                        if equipment == seen_equipment:
                            # At this point, the output needs to be identical
                            if not output_objects == seen_output_objects or not output_states_strs == seen_output_states:
                                raise AssertionError(f"Clashing rules with input objects {rule_name} and {seen_rule_name} in submap {submap_name}.")
                
            seen_object_sets.append((rule_name, input_objects, input_states_strs, equipment, output_objects, output_states_strs))

    # Check heat_cook rules vs cooking individual items, since those are a potential clash that we know of
    # for activity in os.listdir(PROBLEM_FILE_DIR):
    #     if not os.path.isdir(os.path.join(PROBLEM_FILE_DIR, activity)): continue
    #     __, objects, init, goal = _get_defn_elements_from_file(activity)
        
    #     # Check for a rigid body ending cooked and starting not cooked/cook state unspecified, or a future clause for a cooked substance
    #     for cooking_rigids: 


    # NOTE other heuristics as we discover them


if __name__ == "__main__":
    check_clashing_transition_rules()