import json
import pathlib
from nltk.corpus import wordnet as wn
import pandas as pd
import copy

PARAMS_OUTFILE_FN = pathlib.Path(__file__).parents[1] / "generated_data" / "propagated_annots_params.json"
UNRESOLVED_DIR = pathlib.Path(__file__).parents[1] / "generated_data" / "unresolved"
PROP_PARAM_ANNOTS_DIR = pathlib.Path(__file__).parents[1] / "prop_param_annots"

def add_cookable_params(propagated_canonical, props_to_syns, synset_nonexistent, param_but_no_prop, prop_but_no_param):
    # Cooking
    print("COOKING")
    print()
    cooking_params = pd.read_csv(PROP_PARAM_ANNOTS_DIR / "cooking.csv")
    cooking_params = cooking_params[cooking_params["require_cookable/overcookable_temperature"] == True][["synset", "value_cook(C)", "value_overcook(C)"]]
    for i, [synset, cook_temp, overcook_temp] in cooking_params.iterrows():
        try:
            float(cook_temp)
        except ValueError:
            prop_but_no_param["cookable"].append(synset)
            continue
        if pd.isna(float(cook_temp)):
            prop_but_no_param["cookable"].append(synset)
            continue
        if synset not in propagated_canonical:
            # print(f"{synset} not in syn-to-prop file.")
            synset_nonexistent["cookable"].append(synset)
            continue
        if "cookable" not in propagated_canonical[synset]:
            # print(f"{synset} is not cookable but has cookable property param annotation.")
            param_but_no_prop["cookable"].append(synset)
            continue
        propagated_canonical[synset]["cookable"]["cook_temperature"] = float(cook_temp)
        # syns_to_props[synset]["cookable"]["overcook_temperature"] = float(overcook_temp)

    for cookable_synset in props_to_syns["cookable"]:
        try: 
            wn_syn = wn.synset(cookable_synset)
        except:
            continue
        if "cook_temperature" not in propagated_canonical[cookable_synset]["cookable"] and \
                        all(syn.name in propagated_canonical for syn in wn.synset(cookable_synset).hyponyms()):
            # print(f"{cookable_synset} is annotated as cookable but does not have cookable property param annotation")
            prop_but_no_param["cookable"].append(cookable_synset)


def add_coldsource_params(propagated_canonical, props_to_syns, synset_nonexistent, param_but_no_prop, prop_but_no_param):
    # coldSource
    print(); print()
    print("COLDSOURCE")
    print()
    coldSource_params = pd.read_csv(PROP_PARAM_ANNOTS_DIR / "coldSource.csv")
    coldSource_params = coldSource_params[coldSource_params["require_coldsource_temperature"] == True][["synset", "value(C)"]]
    for i, [synset, temp] in coldSource_params.iterrows():
        try:
            float(temp)
        except ValueError:
            prop_but_no_param["coldSource"].append(synset)
            continue
        if pd.isna(float(temp)):
            prop_but_no_param["coldSource"].append(synset)
            continue
        if synset not in propagated_canonical:
            # print(f"{synset} not in syn-to-prop file.")
            synset_nonexistent["coldSource"].append(synset)
            continue
        if "coldSource" not in propagated_canonical[synset]:
            # print(f"{synset} is not coldSource but has coldSource property param annotation.")
            param_but_no_prop["coldSource"].append(synset)
            continue
        propagated_canonical[synset]["coldSource"]["temperature"] = float(temp)
        propagated_canonical[synset]["coldSource"]["heating_rate"] = 0.01

    for coldSource_synset in props_to_syns["coldSource"]:
        try: 
            wn_syn = wn.synset(coldSource_synset)
        except:
            continue
        if "temperature" not in propagated_canonical[coldSource_synset]["coldSource"] and \
                        all(syn.name in propagated_canonical for syn in wn.synset(coldSource_synset).hyponyms()):
            # print(f"{coldSource_synset} is annotated as coldSource_synset but does not have coldSource_synset property param annotation")
            prop_but_no_param["coldSource"].append(coldSource_synset)


def add_heatsource_params(propagated_canonical, props_to_syns, synset_nonexistent, param_but_no_prop, prop_but_no_param):
    # heatSource
    print(); print()
    print("HEATSOURCE")
    print()
    heatSource_params = pd.read_csv(PROP_PARAM_ANNOTS_DIR / "heatSource.csv")

    heatSource_params = heatSource_params[heatSource_params["require_heatsource_temperature"] == True][["synset", "value(C)"]]
    # print(heatSource_params)
    # import sys; sys.exit()
    for i, [synset, temp] in heatSource_params.iterrows():
        try:
            float(temp)
        except ValueError:
            prop_but_no_param["heatSource"].append(synset)
            continue
        if pd.isna(float(temp)):
            print(synset)
            prop_but_no_param["heatSource"].append(synset)
            continue
        if synset not in propagated_canonical:
            # print(f"{synset} not in syn-to-prop file.")
            synset_nonexistent["heatSource"].append(synset)
            continue
        if "heatSource" not in propagated_canonical[synset]:
            # print(f"{synset} is not heatSource but has heatSource property param annotation.")
            param_but_no_prop["heatSource"].append(synset)
            continue
        propagated_canonical[synset]["heatSource"]["temperature"] = float(temp)
        propagated_canonical[synset]["heatSource"]["heating_rate"] = 0.01

    for heatSource_synset in props_to_syns["heatSource"]:
        try: 
            wn_syn = wn.synset(heatSource_synset)
        except:
            continue
        if "temperature" not in propagated_canonical[heatSource_synset]["heatSource"] and \
                        all(syn.name in propagated_canonical for syn in wn.synset(heatSource_synset).hyponyms()):        
            # print(f"{heatSource_synset} is annotated as heatSource but does not have heatSource property param annotation")
            prop_but_no_param["heatSource"].append(heatSource_synset)

# TODO check if that everything that SHOULD have a certain parameter annotation DOES have it 

def create_get_save_propagated_annots_params(propagated_canonical, props_to_syns):
    propagated_params_canonical = copy.deepcopy(propagated_canonical)
    synset_nonexistent, param_but_no_prop, prop_but_no_param = [{"cookable": [], "coldSource": [], "heatSource": []} for __ in range(3)]

    add_cookable_params(propagated_params_canonical, props_to_syns, synset_nonexistent, param_but_no_prop, prop_but_no_param)
    add_coldsource_params(propagated_params_canonical, props_to_syns, synset_nonexistent, param_but_no_prop, prop_but_no_param)
    add_heatsource_params(propagated_params_canonical, props_to_syns, synset_nonexistent, param_but_no_prop, prop_but_no_param)

    with open(PARAMS_OUTFILE_FN, "w") as f:
        json.dump(propagated_params_canonical, f, indent=4)

    with open(UNRESOLVED_DIR / "synset_nonexistent.json", "w") as f:
        json.dump(synset_nonexistent, f, indent=4)
    with open(UNRESOLVED_DIR / "param_but_no_prop.json", "w") as f:
        json.dump(param_but_no_prop, f, indent=4)
    with open(UNRESOLVED_DIR / "prop_but_no_param_or_malformed_param.json", "w") as f:
        json.dump(prop_but_no_param, f, indent=4)