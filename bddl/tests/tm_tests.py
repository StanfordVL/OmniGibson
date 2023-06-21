import json
import os
import re 

from bddl.generated_data.transition_map.tm_submap_params import TM_SUBMAPS_TO_PARAMS
from bddl.parsing import parse_domain
from test_utils import check_synset_predicate_alignment


TRANSITION_MAP_DIR = "../bddl/generated_data/transition_map/tm_jsons"
SYNS_TO_PROPS_JSON = "../bddl/generated_data/propagated_annots_canonical.json"

OBJECT_CAT_RE = r"[A-Za-z-_]+\.n\.[0-9]+$"


# Tests


def no_missing_required_params(rule, submap):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param in param_metadata: 
        assert (not param_metadata[param]["required"]) or (rule[param] is not None), f"Required param {param} of rule TODO is required but not present."
    for param, value in rule.items(): 
        if param_metadata[param]["required"]:
            assert value is not None, f"Required param {param} of rule TODO is required has no value"


def no_incorrectly_formatted_params(rule, submap):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param, value in rule.items(): 
        if param_metadata[param]["type"] == "synset" and value is not None:
            assert type(value) == dict, f"Malformed synset-type value for param {param} in rule TODO"
            for proposed_synset, proposed_integer in value.items():
                assert re.match(OBJECT_CAT_RE, proposed_synset) is not None, f"Malformed synset {proposed_synset} in param {param} of rule TODO in submap {submap}"
                assert type(proposed_integer) == int, f"Malformed integer {proposed_integer} in param {param} of rule TODO"
        elif param_metadata[param]["type"] == "atom" and value is not None:
            assert type(value) == dict, f"Malformed atom-type value for param {param} in rule TODO in submap {submap}"
            for proposed_synsets, proposed_predicates_values in value.items():
                for proposed_synset in proposed_synsets.split(","):
                    assert re.match(OBJECT_CAT_RE, proposed_synset) is not None, f"Malformed synset {proposed_synset} in param {param} of rule TODO in submap {submap}"
                for proposed_predicate_value in proposed_predicates_values:
                    assert len(proposed_predicate_value) == 2, f"Malformed predicate-value pair {proposed_predicate_value} for param {param} in rule TODO"
                    predicate, val = proposed_predicate_value 
                    assert type(predicate) == str, f"Malformed predicate {predicate} for param {param} in rule TODO in submap {submap}"
                    assert type(val) == bool, f"Malformed predicate value {val} for param {param} in rule TODO in submap {submap}"
        elif param_metadata[param]["type"] == "integer": continue
        elif param_metadata[param]["type"] == "string": continue
        elif value is None: 
            continue
        else:
            raise ValueError(f"Unhandled parameter type {param_metadata[param]['type']}")
        

def no_invalid_synsets(rule, submap, syns_to_props):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param, value in rule.items(): 
        if param_metadata[param]["type"] == "synset" and value is not None:   # we know the format since we enforced it above
            for proposed_synset in value.keys():
                assert proposed_synset in syns_to_props, f"Invalid synset: {proposed_synset} in rule TODO in submap {submap}"
        elif param_metadata[param]["type"] == "atom" and value is not None:
            for proposed_synsets in value.keys():
                for proposed_synset in proposed_synsets.split(","):
                    assert proposed_synset in syns_to_props, f"Invalid synset {proposed_synset} in rule TODO in submap {submap}"
        elif param_metadata[param]["type"] == "integer": continue
        elif param_metadata[param]["type"] == "string": continue
        elif value is None:
            continue
        else:
            raise ValueError(f"Unhandled parameter type {param_metadata[param]['type']}")
        

def no_invalid_predicates(rule, submap, domain_predicates):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param, value in rule.items(): 
        if param_metadata[param]["type"] == "synset": continue 
        elif param_metadata[param]["type"] == "integer": continue
        elif param_metadata[param]["type"] == "string": continue
        elif value is None: continue
        elif param_metadata[param]["type"] == "atom": 
            for __, proposed_predicate_values in value.items(): 
                for proposed_predicate, __ in proposed_predicate_values:
                    assert proposed_predicate in domain_predicates, f"Invalid predicate {proposed_predicate} in rule TODO in submap {submap}"
        else:
            raise ValueError(f"Unhandled parameter type {param_metadata[param]['type']}")


def no_misaligned_synsets_predicates(rule, submap, syns_to_props):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param, value in rule.items(): 
        if param_metadata[param]["type"] == "synset": continue 
        elif param_metadata[param]["type"] == "integer": continue
        elif param_metadata[param]["type"] == "string": continue
        elif value is None: continue
        elif param_metadata[param]["type"] == "atom": 
            for synsets, predicate_vals in value.items(): 
                for predicate, __ in predicate_vals:
                    synsets = synsets.split(",")
                    check_synset_predicate_alignment([predicate, *synsets], syns_to_props)
        else:
            raise ValueError(f"Unhandled parameter type {param_metadata[param]['type']}")


def no_substances_with_multiple_instances(rule, submap, syns_to_props):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param, value in rule.items(): 
        if param_metadata[param]["type"] == "synset":
            for synset, num_instances in value.items(): 
                if "substance" in syns_to_props[synset]:
                    assert num_instances == 1, f"Substance {synset} with {num_instances} instances instead of 1 in rule TODO in submap {submap}"
        elif param_metadata[param]["type"] == "atom": continue
        elif param_metadata[param]["type"] == "integer": continue
        elif param_metadata[param]["type"] == "string": continue
        else:
            raise ValueError(f"Unhandled parameter type {param_metadata[param]['type']}")


def verify_tms(): 
    with open(SYNS_TO_PROPS_JSON, "r") as f:
        syns_to_props = json.load(f)
    *__, domain_predicates = parse_domain("omnigibson")
    for submap in TM_SUBMAPS_TO_PARAMS:
        print(submap)
        print()
        with open(os.path.join(TRANSITION_MAP_DIR, submap + ".json"), "r") as f:
            submap_rules = json.load(f)
        for rule in submap_rules:
            no_missing_required_params(rule, submap)
            no_incorrectly_formatted_params(rule, submap)
            no_invalid_synsets(rule, submap, syns_to_props)
            no_invalid_predicates(rule, submap, domain_predicates)
            no_misaligned_synsets_predicates(rule, submap, syns_to_props)
            no_substances_with_multiple_instances(rule, submap, syns_to_props)
    

if __name__ == "__main__":
    verify_tms()