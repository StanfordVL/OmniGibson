import json
import os 
import pandas as pd
import pathlib


TRANSITION_MAP_DIR = pathlib.Path(__file__).parents[1] / "generated_data" / "transition_map" / "tm_jsons"


def generate_slicing_rules(syns_to_param_props, props_to_syns):
    """
    Generates transition rules for slicing of sliceable objects.
    Form: 
    {
        "rule_name": <non-unique string>,
        "input_objects": {
            <sliceable>: 1 
        },
        "output_objects": {
            <sliceable_derivative_synset>: 2
        }
    }
    Assumptions not listed in rule: 
    - Slicer is present and slicing region of slicer is touching sliceable - this is required for the rule to fire
    """

    rules = []
    sliceables = props_to_syns["sliceable"]
    for sliceable in sliceables: 
        assert "sliceable_derivative_synset" in syns_to_param_props[sliceable]["sliceable"], f"Synset {sliceable} has no sliceable derivative synset. Add one in the parameter annotations or handle otherwise."
        assert syns_to_param_props[sliceable]["sliceable"]["sliceable_derivative_synset"] in syns_to_param_props, f"Synset {sliceable} has sliceable derivative synset {syns_to_param_props[sliceable]['sliceable_derivative_synset']} that is not a valid synset. Please check."
        sliceable_derivative_synset = syns_to_param_props[sliceable]["sliceable"]["sliceable_derivative_synset"]

        rule = {
            "rule_name": f"{sliceable}-slicing",
            "input_objects": {
                sliceable: 1
            },
            "output_objects": {
                sliceable_derivative_synset: 2
            }
        }

        rules.append(rule)
    
    rules.sort(key=lambda x: x["rule_name"])
    return rules


def generate_dicing_rules(syns_to_param_props, props_to_syns):
    """
    Generates transition rules for dicing of diceable objects.
    Form: 
    {
        "rule_name": <unique string name>,
        "input_objects": {
            <diceable>: 1 
        },
        "input_states": {
            <diceable>: [
                [
                    "cooked",
                    <bool>
                ]
            ]
        }
        "output_objects": {
            <cooked_diceable_derivative_synset> if cooked else <uncooked_diceable_derivative_synset>: 1
        }
    }
    Assumptions not listed in rule: 
    - Slicer is present and slicing region of slicer is touching diceable - this is required for the rule to fire
    """

    rules = []
    diceables = props_to_syns["diceable"]
    cookables = set(props_to_syns["cookable"])
    for diceable in diceables: 
        assert "uncooked_diceable_derivative_synset" in syns_to_param_props[diceable]["diceable"], f"Synset {diceable} has no uncooked diceable derivative synset. Add one in the parameter annotations."
        assert syns_to_param_props[diceable]["diceable"]["uncooked_diceable_derivative_synset"] in syns_to_param_props, f"Synset {diceable} has uncooked diceable derivative synset {syns_to_param_props[diceable]['uncooked_diceable_derivative_synset']} that is not a valid synset. Please check."
        uncooked_diceable_derivative_synset = syns_to_param_props[diceable]["diceable"]["uncooked_diceable_derivative_synset"]

        uncooked_rule = {
            "rule_name": f"uncooked-{diceable}-dicing",
            "input_objects": {
                diceable: 1
            },
            "input_states": {
                diceable: [
                    [
                        "cooked",
                        False
                    ]
                ]
            },
            "output_objects": {
                uncooked_diceable_derivative_synset: 1
            }
        }
        rules.append(uncooked_rule)

        if diceable in cookables:
            assert "cooked_diceable_derivative_synset" in syns_to_param_props[diceable]["diceable"], f"Synset {diceable} has no cooked diceable derivative synset. Add one in the parameter annotations."
            assert syns_to_param_props[diceable]["diceable"]["cooked_diceable_derivative_synset"] in syns_to_param_props, f"Synset {diceable} has cooked diceable derivative synset {syns_to_param_props[diceable]['cooked_diceable_derivative_synset']} that is not a valid synset. Please check."
            cooked_diceable_derivative_synset = syns_to_param_props[diceable]["diceable"]["cooked_diceable_derivative_synset"]

            cooked_rule = {
                "rule_name": f"cooked-{diceable}-dicing",
                "input_objects": {
                    diceable: 1
                },
                "input_states": {
                    diceable: [
                        [
                            "cooked",
                            True
                        ]
                    ]
                },
                "output_objects": {
                    cooked_diceable_derivative_synset: 1
                }
            }
            rules.append(cooked_rule)
    
    rules.sort(key=lambda x: x["rule_name"])
    return rules 


def generate_substance_cooking_rules(syns_to_param_props, props_to_syns):
    """
    Generates transition rules for dicing of diceable objects.
    Form: 
    {
        "rule_name": <non-unique string>,
        "input_objects": {
            <cookable,substance>: 1
        },
        "output_objects": {
            <substance_cooking_derivative_synset>: 1
        }
    }
    Assumptions not listed in rule: 
    TODO is the below accurate? 
    - Cookable substance must be filling a fillable and the fillable must be hot for the rule to fire
    """

    rules = []
    cookable_substances = set(props_to_syns["cookable"]).intersection(set(props_to_syns["substance"]))
    for cookable_substance in cookable_substances: 
        assert "substance_cooking_derivative_synset" in syns_to_param_props[cookable_substance]["cookable"], f"Synset {cookable_substance} has no substance cooking derivative synset. Add one in the parameter annotations or handle otherwise."
        assert syns_to_param_props[cookable_substance]["cookable"]["substance_cooking_derivative_synset"] in syns_to_param_props, f"Synset {cookable_substance} has substance cooking derivative synset {syns_to_param_props[cookable_substance]['substance_cooking_derivative_synset']} that is not a valid synset. Please check."
        substance_cooking_derivative_synset = syns_to_param_props[cookable_substance]["cookable"]["substance_cooking_derivative_synset"]

        rule = {
            "rule_name": f"{cookable_substance}-cooking",
            "input_objects": {
                cookable_substance: 1
            },
            "output_objects": {
                substance_cooking_derivative_synset: 1
            }
        }
        rules.append(rule)
    
    rules.sort(key=lambda x: x["rule_name"])
    return rules 


def generate_melting_rules(syns_to_param_props, props_to_syns):
    """
    Generates transition rules for melting of meltable objects.
    Form: 
    {
        "rule_name": <unique string name>,
        "input_objects": {
            <meltable>: 1
        },
        "output_objects": {
            <meltable_derivative_synset>: 1
        }
    }
    """
    rules = []
    meltables = props_to_syns["meltable"]
    for meltable in meltables: 
        assert "meltable_derivative_synset" in syns_to_param_props[meltable]["meltable"], f"Synset {meltable} has no meltable derivative synset. Add one in the parameter annotations or handle otherwise."
        assert syns_to_param_props[meltable]["meltable"]["meltable_derivative_synset"] in syns_to_param_props, f"Synset {meltable} has meltable derivative synset {syns_to_param_props[meltable]['meltable_derivative_synset']} that is not a valid synset. Please check."
        meltable_derivative_synset = syns_to_param_props[meltable]["meltable"]["meltable_derivative_synset"]

        rule = {
            "rule_name": f"{meltable}-melting",
            "input_objects": {
                meltable: 1
            },
            "output_objects": {
                meltable_derivative_synset: 1
            }
        }

        rules.append(rule)
    
    rules.sort(key=lambda x: x["rule_name"])
    return rules


def create_get_save_implicit_transition_rules(syns_to_param_props, props_to_syns):
    print("Creating implicit transition rule jsons...")
    slicing_rules = generate_slicing_rules(syns_to_param_props, props_to_syns)
    dicing_rules = generate_dicing_rules(syns_to_param_props, props_to_syns)
    substance_cooking_rules = generate_substance_cooking_rules(syns_to_param_props, props_to_syns)
    melting_rules = generate_melting_rules(syns_to_param_props, props_to_syns)

    with open(os.path.join(TRANSITION_MAP_DIR, "slicing.json"), "w") as f:
        json.dump(slicing_rules, f, indent=4)
    with open(os.path.join(TRANSITION_MAP_DIR, "dicing.json"), "w") as f:
        json.dump(dicing_rules, f, indent=4)
    with open(os.path.join(TRANSITION_MAP_DIR, "substance_cooking.json"), "w") as f:
        json.dump(substance_cooking_rules, f, indent=4)
    with open(os.path.join(TRANSITION_MAP_DIR, "melting.json"), "w") as f:
        json.dump(melting_rules, f, indent=4)
    print("Created and saved implicit transition rule jsons.")