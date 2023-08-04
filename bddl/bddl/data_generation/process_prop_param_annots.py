import csv
from enum import IntEnum
import json
import os
import re
import pandas as pd
import pathlib


OBJECT_INSTANCE_RE = r"[A-Za-z-_]+\.n\.[0-9]+_[0-9]+"
OBJECT_CAT_RE = r"[A-Za-z-_]+\.n\.[0-9]+$"
OBJECT_CAT_AND_INST_RE = r"[A-Za-z-_]+\.n\.[0-9]+"

SYNS_TO_PROPS = pathlib.Path(__file__).parents[1] / "generated_data" / "propagated_annots_canonical.json"
PROP_PARAM_ANNOTS_DIR = pathlib.Path(__file__).parents[1] / "generated_data" / "prop_param_annots"
PARAMS_OUTFILE_FN = pathlib.Path(__file__).parents[1] / "generated_data" / "propagated_annots_params.json"
REMOVER_SYNSET_MAPPING = pathlib.Path(__file__).parents[1] / "generated_data" / "remover_synset_mapping.json"
LEAF_SYNSETS_FILE = pathlib.Path(__file__).parents[1] / "generated_data" / "synsets.csv"

# Helpers
LEAF_SYNSETS_SET = set(pd.read_csv(LEAF_SYNSETS_FILE)["synset"])

# Specific methods for applying / removing particles
class ParticleModifyMethod(IntEnum):
    ADJACENCY = 0
    PROJECTION = 1

# Specific condition types for applying / removing particles
class ParticleModifyCondition(IntEnum):
    FUNCTION = 0
    SATURATED = 1
    TOGGLEDON = 2
    GRAVITY = 3

PREDICATE_MAPPING = {
    "saturated": ParticleModifyCondition.SATURATED,
    "toggled_on": ParticleModifyCondition.TOGGLEDON,
    "function": ParticleModifyCondition.FUNCTION,
}

def parse_predicate(predicate):
    pred_type = PREDICATE_MAPPING[predicate.split(" ")[0]]
    if pred_type == ParticleModifyCondition.SATURATED:
        cond = (pred_type, predicate.split(" ")[1].split(".")[0])
    elif pred_type == ParticleModifyCondition.TOGGLEDON:
        cond = (pred_type, True)
    elif pred_type == ParticleModifyCondition.FUNCTION:
        raise ValueError("Not supported")
    else:
        raise ValueError(f"Unsupported condition type: {pred_type}")
    return cond


def parse_conditions_entry(unparsed_conditions):
    # print(f"Parsing: {unparsed_conditions}")
    if unparsed_conditions.isnumeric():
        always_true = bool(int(unparsed_conditions))
        conditions = [] if always_true else None
    else:
        conditions = [parse_predicate(predicate=pred) for pred in unparsed_conditions.lower().split(" or ")]
    return conditions



# particleRemovers

def get_synsets_to_particle_remover_params():
    synset_cleaning_mapping = {}
    records = []
    with open(os.path.join(PROP_PARAM_ANNOTS_DIR, "particleRemover.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    for record in records: 
        synset = re.match(OBJECT_CAT_AND_INST_RE, record["synset"])
        assert synset is not None, "prop_param_annots/particleRemover.csv has line without properly formed synset"
        synset = synset.group()

        # Skip washer and dryer
        if "not particleremover" in record["synset"].lower(): continue
        
        default_visual_conditions = parse_conditions_entry(record["other visualSubstances"])
        default_physical_conditions = parse_conditions_entry(record["other physicalSubstances"])
        if record["method"] == "projection": 
            method = ParticleModifyMethod.PROJECTION
        elif record["method"] == "adjacency":
            method = ParticleModifyMethod.ADJACENCY
        else:
            raise ValueError(f"Synset {record['synset']} prop particleRemover has invalid method {record['method']}")
        
        remover_kwargs = {
            "conditions": {},
            "default_visual_conditions": default_visual_conditions,
            "default_physical_conditions": default_physical_conditions,
            "method": method
        }

        # Iterate through all the columns headed by a substance, in no particular order since their ultimate location is a dict
        for dirtiness_substance_synset in [key for key in record if re.match(OBJECT_CAT_AND_INST_RE, key) is not None]:
            conditions = parse_conditions_entry(record[dirtiness_substance_synset])
            if conditions is not None: 
                og_cat = dirtiness_substance_synset.split(".")[0]
                remover_kwargs["conditions"][og_cat] = conditions
        
        synset_cleaning_mapping[synset] = remover_kwargs
    
    return synset_cleaning_mapping


# Main parameter method

def create_get_save_propagated_annots_params(syns_to_props):
    print("Processing param annots...")
    
    synsets_to_particleremover_params = get_synsets_to_particle_remover_params()
    for particleremover_syn, particleremover_params in synsets_to_particleremover_params.items():
        assert "particleRemover" in syns_to_props[particleremover_syn], f"Synset {particleremover_syn} has particleRemover params but is not annotated as a particleRemover"
        syns_to_props[particleremover_syn]["particleRemover"] = particleremover_params

    for prop_fn in os.listdir(PROP_PARAM_ANNOTS_DIR):
        prop = prop_fn.split(".")[0]

        if prop == "particleRemover": continue
        else:        
            param_annots = pd.read_csv(os.path.join(PROP_PARAM_ANNOTS_DIR, prop_fn)).to_dict(orient="records")
            for param_record in param_annots: 
                for param_name, param_value in param_record.items():

                    if param_name == "synset": continue

                    # NaN values
                    if pd.isna(param_value):
                        if prop == "coldSource":
                            raise ValueError(f"synset {param_record['synset']} coldSource annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value.")
                        elif prop == "cookable":
                            raise ValueError(f"synset {param_record['synset']} cookable annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value.")
                        elif prop == "flammable":
                            raise ValueError(f"synset {param_record['synset']} flammable annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value.")
                        elif prop == "heatable":
                            raise ValueError(f"synset {param_record['synset']} heatable annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value.")
                        elif prop == "heatSource":
                            raise ValueError(f"synset {param_record['synset']} heatSource annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value.")
                        elif prop == "particleApplier":
                            raise ValueError(f"synset {param_record['synset']} particleApplier annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value.")
                        elif prop == "particleSink":
                            if param_name == "conditions": 
                                formatted_param_value = {}
                            elif param_name == "default_physical_conditions": 
                                formatted_param_value = []
                            elif param_name == "default_visual_conditions":
                                formatted_param_value = None
                            else:
                                raise ValueError(f"synset {param_record['synset']} particleSink annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value.")
                        elif prop == "particleSource":
                            raise ValueError(f"synset {param_record['synset']} particleSource annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value.")
                    
                    # `conditions` values - format is keyword1:bool_value1;keyword2:bool_value2;...;keywordN:bool_valueN
                    elif param_name == "conditions": 
                        if (prop == "particleApplier") or (prop == "particleSource"):
                            # NOTE assumes that there may be multiple conditions but only handles gravity and toggled_on
                            conditions = param_value.split(";")
                            if len(conditions) > 1:
                                print(f"WARNING: multiple conditions for prop {prop} for synset {param_record['synset']}")
                            
                            # Checks that at most one of gravity and toggled_on are required
                            assert (not "gravity:True" in conditions) or (not "toggled_on:True" in conditions), f"synset {param_record['synset']} prop {prop} has contradictory conditions gravity:True and toggled_on:True"
                            
                            # Now that we know at most one of gravity and toggled_on are true, verify that it's one of them and determine which one
                            gravity_or_toggled_on = ""
                            for condition in conditions: 
                                if condition == "gravity:True": 
                                    gravity_or_toggled_on = "gravity"
                                elif condition == "toggled_on:True":
                                    gravity_or_toggled_on = "toggled_on"
                            if gravity_or_toggled_on not in ["gravity", "toggled_on"]:
                                raise ValueError(f"Synset {param_record['synset']} prop {prop} uses neither gravity nor toggled_on")

                            # Make the `conditions` entry
                            formatted_param_value = {
                                param_record["system"]: [(ParticleModifyCondition.GRAVITY if gravity_or_toggled_on == "gravity" else (ParticleModifyCondition.TOGGLEDON), True)]
                            }

                        else:
                            raise ValueError(f"prop {prop} not handled for parameter name `conditions`")
                    
                    # Can skip system since it's just part of handling `conditions`
                    elif param_name == "system": continue
                    
                    # `method` values
                    elif param_name == "method": 
                        if (prop == "particleApplier") or (prop == "particleSource"):
                            formatted_param_value = ParticleModifyMethod.PROJECTION
                        else:
                            raise ValueError(f"prop {prop} not handled for parameter name `method`")
                    
                    # Float values
                    else:
                        try: 
                            formatted_param_value = float(param_value)
                        except ValueError:
                            raise ValueError(f"Synset {param_record['synset']} property {prop} has param {param_name} that is not named `method`, `conditions`, or `system` and is not a NaN or a float. This is unhandled - please check.")

                    syns_to_props[param_record["synset"]][prop][param_name] = formatted_param_value
    

    with open(PARAMS_OUTFILE_FN, "w") as f:
        json.dump(syns_to_props, f, indent=4)
    
    print("Params parsed and added to flat and hierarchical files, saved.")


if __name__ == "__main__":
    create_get_save_propagated_annots_params()

