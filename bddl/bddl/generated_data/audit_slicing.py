import json 
import pandas as pd
from pprint import pprint
import re


synsets_df = pd.read_csv("synsets.csv")
leaf_synsets = set(synsets_df["synset"])
syn_to_hypernym = dict(zip(synsets_df["synset"], synsets_df["hypernyms"]))


with open("properties_to_synsets.json", "r") as f:
    prop_to_syn = json.load(f)

sliceables = set(prop_to_syn["sliceable"])
leaf_sliceables = leaf_synsets.intersection(sliceables)

leaf_sliceables_without_halves = [
    syn for syn in leaf_sliceables
    if (f"half__{syn.split('.')[0]}.n.01" not in leaf_synsets) and (f"half__{syn.split('.')[0]}.n.02" not in leaf_synsets)
]


diceables = set(prop_to_syn["diceable"])
leaf_diceables = leaf_synsets.intersection(diceables)

leaf_substances = set(prop_to_syn["substance"]).intersection(leaf_synsets)
currently_sliceable_diceable_substances = leaf_substances.intersection(leaf_diceables.union(leaf_sliceables))


# NOTE do after removing currently-sliceable and diceable substances

syns_to_halves = {syn: "half__" + syn.split(".n.")[0] + ".n.01" for syn in leaf_sliceables_without_halves}
with open("propagated_annots_params.json", "r") as f:
    syn_to_prop = json.load(f)

if False: 
    halves_data = []
    seen_half_synsets = set()
    for syn, half_syn in syns_to_halves.items():
        assert half_syn not in seen_half_synsets, f"duplicate half syn {half_syn}" 
        seen_half_synsets.add(half_syn)
        record = {
            "synset": half_syn,
            "hypernyms": syn_to_hypernym[syn],
            "is_custom": 1,
            "breakable": int("breakable" in syn_to_prop[syn]),
            "fillable": int("fillable" in syn_to_prop[syn]),
            "flammable": int("flammable" in syn_to_prop[syn]),
            "openable": 0,
            "toggleable": 0,
            "cookable": int("cookable" in syn_to_prop[syn]),
            "heatSource": 0,
            "coldSource": 0,
            "sliceable": 0,
            "diceable": "TODO",
            "slicer": int("slicer" in syn_to_prop[syn]),
            "assembleable": int("assembleable" in syn_to_prop[syn]),
            "meltable": int("meltable" in syn_to_prop[syn]),
            "particleRemover": int("particleRemover" in syn_to_prop[syn]),
            "particleApplier": int("particleApplier" in syn_to_prop[syn]),
            "particleSource": int("particleSource" in syn_to_prop[syn]),
            "needsOrientation": int("needsOrientation" in syn_to_prop[syn]),
            "particleSink": int("particleSink" in syn_to_prop[syn]),
            "sceneObject": int("sceneObject" in syn_to_prop[syn]),
            "waterCook": int("waterCook" in syn_to_prop[syn]),
            "mixingTool": int("mixingTool" in syn_to_prop[syn])
        }
        assert ("rigidBody" in syn_to_prop[syn]) or ("cloth" in syn_to_prop[syn]) or ("softBody" in syn_to_prop[syn]) or ("rope" in syn_to_prop[syn]), f"Synset {syn} is a half-candidate but is not a rigid or deformable"
        if "rigidBody" in syn_to_prop[syn]:
            record["objectType"] = "rigidBody"
        elif "cloth" in syn_to_prop[syn]:
            record["objectType"] = "cloth"
        elif "rope" in syn_to_prop[syn]:
            record["objectType"] = "rope"
        elif "softBody" in syn_to_prop[syn]:
            record["objectType"] = "softBody"
        else:
            raise AssertionError("No object type coming from whole version")
        halves_data.append(record)

    halves_df = pd.DataFrame(columns=synsets_df.columns, data=halves_data)
    halves_df.to_csv("halfs_annots.csv", index=False)


# DICEABLES WITHOUT DICED

leaf_diceables = leaf_synsets.intersection(diceables)
leaf_diceable_root_syns = set([syn.split("__")[-1] for syn in leaf_diceables])
# leaf_diceds = set([syn for syn in leaf_synsets if "diced__" in syn])
leaf_diceds = set([syn for syn in leaf_synsets if re.match(r"^diced__", syn) is not None])

leaf_diced_root_syns = set([syn.split("__")[-1] for syn in leaf_diceds])
leaf_diceable_root_syns_without_diced = leaf_diceable_root_syns.difference(leaf_diced_root_syns)

# NOTE Do after sliceables
diceds_data = []
seen_diced_synsets = set()
root_syns_to_diceds = {syn: "diced__" + syn for syn in leaf_diceable_root_syns_without_diced}

if False:
    for root_syn, diced_syn in root_syns_to_diceds.items():
        assert diced_syn not in seen_diced_synsets, f"duplicate half syn {diced_syn}" 
        seen_diced_synsets.add(diced_syn)
        
        assert (f"half__{root_syn}" in syn_to_hypernym) or (f"sliced__{root_syn}" in syn_to_hypernym), f"Diceable root syn that does not have half- or sliced- version: {synset}"
        if f"half__{root_syn}" in syn_to_hypernym:
            # record["hypernym"] = syn_to_hypernym[f"half__{root_syn}"]
            diceable_version = f"half__{root_syn}"
        elif f"sliced__{root_syn}" in syn_to_hypernym:
            # record["hypernym"] = syn_to_hypernym[f"sliced__{root_syn}"]
            diceable_version = f"sliced__{root_syn}"
        else:
            raise AssertionError("No hypernym coming from sliced/half version")
        

        record = {
            "synset": diced_syn,
            "hypernyms": syn_to_hypernym[diceable_version],
            "is_custom": 1,
            "breakable": int("breakable" in syn_to_prop[diceable_version]),
            "fillable": int("fillable" in syn_to_prop[diceable_version]),
            "flammable": int("flammable" in syn_to_prop[diceable_version]),
            "openable": 0,
            "toggleable": 0,
            "cookable": int("cookable" in syn_to_prop[diceable_version]),
            "heatSource": 0,
            "coldSource": 0,
            "sliceable": 0,
            "diceable": 0,
            "slicer": int("slicer" in syn_to_prop[diceable_version]),
            "assembleable": int("assembleable" in syn_to_prop[diceable_version]),
            "meltable": int("meltable" in syn_to_prop[diceable_version]),
            "particleRemover": int("particleRemover" in syn_to_prop[diceable_version]),
            "particleApplier": int("particleApplier" in syn_to_prop[diceable_version]),
            "particleSource": int("particleSource" in syn_to_prop[diceable_version]),
            "needsOrientation": int("needsOrientation" in syn_to_prop[diceable_version]),
            "particleSink": int("particleSink" in syn_to_prop[diceable_version]),
            "sceneObject": int("sceneObject" in syn_to_prop[diceable_version]),
            "waterCook": int("waterCook" in syn_to_prop[diceable_version]),
            "mixingTool": int("mixingTool" in syn_to_prop[diceable_version]),
            "objectType": "macroPhysicalSubstance"
        }
        diceds_data.append(record)

    diceds_df = pd.DataFrame(columns=synsets_df.columns, data=diceds_data)
    diceds_df.to_csv("diceds_annots.csv", index=False)


# See which cooked__diced__s are missing
potentially_nonleaf_cookables = set(prop_to_syn["cookable"])
cookable_diceds = potentially_nonleaf_cookables.intersection(leaf_diceds)
existing_cooked_diceds = set([syn for syn in leaf_synsets if re.match(r"^cooked__diced__", syn) is not None])

cookablediceds_without_cookeddiced = set([syn for syn in cookable_diceds if f"cooked__{syn}" not in existing_cooked_diceds])

if False:

    cooked_diceds_data = []
    root_syns = set([syn.split("__")[-1] for syn in cookablediceds_without_cookeddiced])
    for root_syn in root_syns: 
        diced_version = f"diced__{root_syn}"
        record = {
            "synset": f"cooked__diced__{root_syn}",
            "hypernyms": syn_to_hypernym[diced_version],
            "is_custom": 1,
            "breakable": int("breakable" in syn_to_prop[diced_version]),
            "fillable": int("fillable" in syn_to_prop[diced_version]),
            "flammable": int("flammable" in syn_to_prop[diced_version]),
            "openable": 0,
            "toggleable": 0,
            "cookable": 0,
            "heatSource": 0,
            "coldSource": 0,
            "sliceable": 0,
            "diceable": 0,
            "slicer": int("slicer" in syn_to_prop[diced_version]),
            "assembleable": int("assembleable" in syn_to_prop[diced_version]),
            "meltable": int("meltable" in syn_to_prop[diced_version]),
            "particleRemover": int("particleRemover" in syn_to_prop[diced_version]),
            "particleApplier": int("particleApplier" in syn_to_prop[diced_version]),
            "particleSource": int("particleSource" in syn_to_prop[diced_version]),
            "needsOrientation": int("needsOrientation" in syn_to_prop[diced_version]),
            "particleSink": int("particleSink" in syn_to_prop[diced_version]),
            "sceneObject": int("sceneObject" in syn_to_prop[diced_version]),
            "waterCook": int("waterCook" in syn_to_prop[diced_version]),
            "mixingTool": int("mixingTool" in syn_to_prop[diced_version]),
            "objectType": "macroPhysicalSubstance"
        }
        cooked_diceds_data.append(record)

    cooked_diceds_df = pd.DataFrame(columns=synsets_df.columns, data=cooked_diceds_data)
    cooked_diceds_df.to_csv("cooked_diced_annots.csv", index=False)


# See which cooked__ for cookable substances are missing 
cookable_substances = leaf_synsets.intersection(set(prop_to_syn["substance"]).intersection(set(prop_to_syn["cookable"])))
existing_cooked_substance_syns = set([syn for syn in leaf_synsets if re.match(r"^cooked__", syn) is not None])

print(len(cookable_substances))
print(len(existing_cooked_substance_syns))

# cookable_substances_without_synset_roots = set([syn.split(".n.")[0] + ".n.01" for syn in cookable_substances]).difference(set([
#     syn.split("cooked__")[-1] for syn in existing_cooked_substance_syns
# ]))

# cooked_substs_data = []
# for root_syn in cookable_substances_without_synset_roots:

cookable_subst_to_cooked_subst = {}
for cookable_subst in cookable_substances:
    root = cookable_subst.split(".n.")[0] + ".n.01"
    cooked_subst = "cooked__" + root 
    if cooked_subst not in existing_cooked_substance_syns: 
        cookable_subst_to_cooked_subst[cookable_subst] = cooked_subst

cooked_substs_data = []
for cookable_subst, cooked_subst in cookable_subst_to_cooked_subst.items():
    record = {
        "synset": cooked_subst,
        "hypernyms": syn_to_hypernym[cookable_subst],
        "is_custom": 1,
        "breakable": int("breakable" in syn_to_prop[cookable_subst]),
        "fillable": int("fillable" in syn_to_prop[cookable_subst]),
        "flammable": int("flammable" in syn_to_prop[cookable_subst]),
        "openable": 0,
        "toggleable": 0,
        "cookable": 0,
        "heatSource": 0,
        "coldSource": 0,
        "sliceable": 0,
        "diceable": 0,
        "slicer": int("slicer" in syn_to_prop[cookable_subst]),
        "assembleable": int("assembleable" in syn_to_prop[cookable_subst]),
        "meltable": int("meltable" in syn_to_prop[cookable_subst]),
        "particleRemover": int("particleRemover" in syn_to_prop[cookable_subst]),
        "particleApplier": int("particleApplier" in syn_to_prop[cookable_subst]),
        "particleSource": int("particleSource" in syn_to_prop[cookable_subst]),
        "needsOrientation": int("needsOrientation" in syn_to_prop[cookable_subst]),
        "particleSink": int("particleSink" in syn_to_prop[cookable_subst]),
        "sceneObject": int("sceneObject" in syn_to_prop[cookable_subst]),
        "waterCook": int("waterCook" in syn_to_prop[cookable_subst]),
        "mixingTool": int("mixingTool" in syn_to_prop[cookable_subst]),
        # "objectType": "macroPhysicalSubstance"
    }

    assert ("liquid" in syn_to_prop[cookable_subst]) or ("microPhysicalSubstance" in syn_to_prop[cookable_subst]) or ("macroPhysicalSubstance" in syn_to_prop[cookable_subst]) or ("visualSubstance" in syn_to_prop[cookable_subst]), f"No valid object type: {cookable_subst}"
    if "liquid" in syn_to_prop[cookable_subst]:
        record["objectType"] = "liquid"
    elif "microPhysicalSubstance" in syn_to_prop[cookable_subst]:
        record["objectType"] = "microPhysicalSubstance"
    elif "macroPhysicalSubstance" in syn_to_prop[cookable_subst]:
        record["objectType"] = "macroPhysicalSubstance"
    elif "visualSubstance" in syn_to_prop[cookable_subst]:
        record["objectType"] = "visualSubstance"
    else:
        raise AssertionError("No valid object type")
    
    cooked_substs_data.append(record)

cooked_substs_df = pd.DataFrame(columns=synsets_df.columns, data=cooked_substs_data)
cooked_substs_df.to_csv("cooked_substs_annots.csv", index=False)