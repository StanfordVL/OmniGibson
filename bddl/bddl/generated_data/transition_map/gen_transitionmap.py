import json 
import csv
import pandas as pd
import os

from tm_submap_params import TM_SUBMAPS_TO_PARAMS

SHEETS_DIR = "tm_raw_data"
JSONS_DIR = "tm_jsons"


def sheet_to_json(submap):
    formatted_map = {}
    params = TM_SUBMAPS_TO_PARAMS[submap]
    raw_data = pd.read_csv(os.path.join(SHEETS_DIR, submap + ".csv")).to_json(orient="records")
    data = json.loads(raw_data)
    reformatted_data = []
    for rule in data:
        reformatted_rule = {}
        for param, value in rule.items():
            if value is not None and " and " in value:
                value = value.split(" and ")
            reformatted_rule[param] = value
        reformatted_data.append(reformatted_rule)

    with open(os.path.join(JSONS_DIR, submap + ".json"), "w") as f:
        json.dump(reformatted_data, f, indent=4)


if __name__ == "__main__":
    for submap in TM_SUBMAPS_TO_PARAMS:
        sheet_to_json(submap)