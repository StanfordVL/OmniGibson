import json 
import csv
import pandas as pd
import os
from collections import Counter
import re

from tm_submap_params import TM_SUBMAPS_TO_PARAMS

SHEETS_DIR = "tm_raw_data"
JSONS_DIR = "tm_jsons"

OBJECT_CAT_AND_INST_RE = r"[A-Za-z-_]+\.n\.[0-9]+"


def sheet_to_json(submap):
    print() 
    print(submap)
    params = TM_SUBMAPS_TO_PARAMS[submap]
    raw_data = pd.read_csv(os.path.join(SHEETS_DIR, submap + ".csv"))[params].to_json(orient="records")
    data = json.loads(raw_data)
    reformatted_data = []
    for rule in data:
        # print(rule["rule_name"])
        reformatted_rule = {}
        for param, value in rule.items():
            if TM_SUBMAPS_TO_PARAMS[submap][param]["type"] == "synset" and value is not None:
                value = value.split(" and ")
                value = Counter([re.match(OBJECT_CAT_AND_INST_RE, atom).group() for atom in value])
            elif TM_SUBMAPS_TO_PARAMS[submap][param]["type"] == "atom" and value is not None:
                value = value.split(" and ")
                unary_atoms = []
                binary_atoms = []
                for atom in value: 
                    elements = atom.split(" ")
                    assert len(elements) in range(2, 5), f"Malformed atom {atom}"
                    if len(elements) == 3:
                        if elements[0] == "not":
                            unary_atoms.append(elements)
                        else:
                            binary_atoms.append(elements)
                    elif len(elements) == 2:
                        unary_atoms.append(elements)
                    else:
                        binary_atoms.append(elements)
                reformatted_atoms = {}
                for atom in unary_atoms: 
                    synset = re.match(OBJECT_CAT_AND_INST_RE, atom[-1]).group()
                    if synset in reformatted_rule:
                        reformatted_atoms[synset].append((atom[-2], atom[0] != "not"))
                    else:
                        reformatted_atoms[synset] = [(atom[-2], atom[0] != "not")]
                for atom in binary_atoms:
                    synset1, synset2 = re.match(OBJECT_CAT_AND_INST_RE, atom[-2]).group(), re.match(OBJECT_CAT_AND_INST_RE, atom[-1]).group()
                    if f"{synset1},{synset2}" in reformatted_rule:
                        reformatted_atoms[f"{synset1},{synset2}"].append((atom[-3], atom[0] != "not"))
                    else:
                        reformatted_atoms[f"{synset1},{synset2}"] = [(atom[-3], atom[0] != "not")]
                print(reformatted_atoms)
                value = reformatted_atoms
            elif TM_SUBMAPS_TO_PARAMS[submap][param]["type"] == "string":
                value = value
            elif value is None:
                value = None
            else: 
                raise ValueError(f"Unhandled parameter type {TM_SUBMAPS_TO_PARAMS[submap][param]['type']}")
            reformatted_rule[param] = value
        reformatted_data.append(reformatted_rule)

    with open(os.path.join(JSONS_DIR, submap + ".json"), "w") as f:
        json.dump(reformatted_data, f, indent=4)


if __name__ == "__main__":
    for submap in TM_SUBMAPS_TO_PARAMS:
        sheet_to_json(submap)