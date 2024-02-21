import json
import os 
import pandas as pd 
from pprint import pprint


leaf_synsets = set(pd.read_csv("synsets.csv")["synset"])
with open("properties_to_synsets.json", "r") as f:
    props_to_syns = json.load(f)


for param_fn in os.listdir("prop_param_annots"):
    param = param_fn.split(".")[0]
    param_annotated_synsets = set(pd.read_csv(os.path.join("prop_param_annots", param_fn))["synset"])
    prop_annotated_synsets = set(props_to_syns[param])

    param_annotated_leaf_synsets = param_annotated_synsets.intersection(leaf_synsets)
    prop_annotated_leaf_synsets = prop_annotated_synsets.intersection(leaf_synsets)

    print(); print()
    print(f"Param: {param}")
    print("Param but no prop")
    pprint(param_annotated_leaf_synsets.difference(prop_annotated_leaf_synsets))
    print()
    print("Prop but no param")
    pprint(prop_annotated_leaf_synsets.difference(param_annotated_leaf_synsets))
    input()

    if param == "cookable":
        pd.Series(list(prop_annotated_leaf_synsets.difference(param_annotated_leaf_synsets))).to_csv("cookable_no_param.csv", index=False)

