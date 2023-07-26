import csv
import json
import os
import pandas as pd
import copy
from pprint import pprint


# PARAM_SHEETS_DIR = "prop_param_annots"
# NEW_PARAM_SHEETS_DIR = "prop_param_annots_new"
# with open("properties_to_synsets.json", "r") as f:
#     props_to_syns = json.load(f)
# leaf_synsets = set(pd.read_csv("synsets.csv")["synset"])


particle_applier_addition_annots_fn = "../generated_data/prop_param_annots/particleApplier_addition.csv"
particle_applier_annots_fn = "../generated_data/prop_param_annots/particleApplier.csv"

particle_applier_addition_annots = pd.read_csv(particle_applier_addition_annots_fn)
particle_applier_annots = pd.read_csv(particle_applier_annots_fn)

syn_to_sys = {}
for record in particle_applier_addition_annots.to_dict(orient="records"):
    syn_to_sys[record["synset"]] = record["system"]
    # print(syn_to_sys[record["synset"]])

particle_applier_annots_new = []
for record in particle_applier_annots.to_dict(orient="records"):
    new_record = copy.deepcopy(record)
    new_record["system"] = syn_to_sys[record["synset"]]
    # print(new_record["system"])
    particle_applier_annots_new.append(new_record)

particle_applier_annots_new_df = pd.DataFrame(particle_applier_annots_new, columns=["synset", "method", "conditions", "system"])
particle_applier_annots_new_df.to_csv("prop_param_annots/particleApplier_joint.csv", index=False)