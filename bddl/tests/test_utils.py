import json
import os
import re 
from bddl.generated_data.transition_map.tm_submap_params import TM_SUBMAPS_TO_PARAMS

# Constants

UNARIES = ["cooked", "real", "future", "frozen", "closed", "open", "folded", "unfolded", "toggled_on", "hot", "on_fire", "assembled", "broken"]
BINARIES = ["saturated", "covered", "filled", "contains", "ontop", "nextto", "empty", "under", "touching", "inside", "overlaid", "attached", "draped", "insource", "inroom"]

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
FUTURE_SYNSET = "future"


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
        seen = set() 
        for rule in submap: 
            input_objects = rule.get("input_objects", {})
            input_states = rule.get("input_states", {})
            input_states = input_states if input_states is not None else {}
            # NOTE doing input_objects.keys, not input_objects.items, because simulator is not actually sensitive to amount. It only checks for category, 
            #   so different amounts but same categories still need to result in the same output.
            # Alphabetize everything.
            input_objects = sorted(input_objects.keys(), key=lambda x: x[0])
            if "bagel_dough.n.01" in input_objects:
                print(input_objects)
            input_states_str = [syns + "@" + ";".join([f"{pred}:{val}" for pred, val in sorted(states, key=lambda x: x[0])]) for syns, states in sorted(input_states.items(), key=lambda x: x[0])]
            input_states_str = "-".join(input_states_str)
            if "bagel_dough.n.01" in input_objects:
                print(input_states_str)


if __name__ == "__main__":
    check_clashing_transition_rules()