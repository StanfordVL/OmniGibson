import pandas as pd
import json

heatable_params_fn = "prop_param_annots/heatable.csv"
heatable_params = pd.read_csv(heatable_params_fn)

with open("properties_to_synsets.json") as f:
    props_to_syns = json.load(f)
nonsubstances = set(props_to_syns["nonSubstance"])

heatable_params_nonsubstances = heatable_params[heatable_params["synset"].isin(nonsubstances)]
heatable_params_nonsubstances.to_csv("prop_param_annots/heatable.csv", index=False)