import json
import pandas as pd 


with open("properties_to_synsets.json", "r") as f:
    props_to_syns = json.load(f)

with open("propagated_annots_canonical.json", "r") as f:
    syns_to_props = json.load(f)

leaf_synsets = set(pd.read_csv("synsets.csv")["synset"])

if False:
    cookable_df = pd.read_csv("prop_param_annots/cookable.csv")
    cookable_substances = set(props_to_syns["cookable"]).intersection(props_to_syns["substance"])

    # uncooked_to_cooked_data = {
    #     cookable_substance: f"cooked__{cookable_substance.split('.n.')[0]}.n.01"
    #     for cookable_substance in cookable_substances
    # }.items()
    uncooked_to_cooked_data = {}
    for cookable_substance in cookable_substances:
        cooked_version = f"cooked__{cookable_substance.split('.n.')[0]}.n.01"
        assert cooked_version in syns_to_props, cooked_version
        uncooked_to_cooked_data[cookable_substance] = cooked_version

    uncooked_to_cooked_df = pd.DataFrame(data=uncooked_to_cooked_data, columns=["synset", "substance_cooked_version"])

    cookable_df["synset"] = cookable_df["synset"].astype(str)
    # cookable_df["cook_temperature"] = cookable_df["cook_temperature"].astype(str)
    uncooked_to_cooked_df["synset"] = uncooked_to_cooked_df["synset"].astype(str)

    # import sys; sys.exit()
    augmented_df = cookable_df.merge(uncooked_to_cooked_df, on="synset", how="left",)

    augmented_df.to_csv("cooked_substance_synset_version.csv", index=False)


if False:
    sliceables = props_to_syns["sliceable"]
    sliceable_df = pd.DataFrame(data=sorted(sliceables), columns=["synset"])
    whole_to_half_data = {}
    print("tape.n.03" in set(sliceables).intersection(leaf_synsets))
    for sliceable in set(sliceables).intersection(leaf_synsets):
        if sliceable == "tape.n.04":
            continue
        half_version = f"half__{sliceable.split('.n.')[0]}.n.01"
        assert half_version in syns_to_props, half_version
        whole_to_half_data[sliceable] = half_version

    sliceable_to_sliced_df = pd.DataFrame(data=whole_to_half_data.items(), columns=["synset", "sliceable_derivative_synset"])

    augmented_df = sliceable_df.merge(sliceable_to_sliced_df, on="synset", how="left")
    augmented_df.to_csv("sliceable_synset_version.csv", index=False)


diceables = set(props_to_syns["diceable"])
cookables = set(props_to_syns["cookable"])
cookable_diceables = diceables.intersection(cookables)

half_to_diced_data = []
for diceable in diceables:
    record = [diceable]
    record.append(f"diced__{diceable.split('__')[-1].split('.n.')[0]}.n.01")
    if not (diceable == "half__papaya.n.01"): 
        assert f"diced__{diceable.split('__')[-1].split('.n.')[0]}.n.01" in syns_to_props, f"diced__{diceable.split('__')[-1].split('.n.')[0]}.n.01"
    if diceable in cookable_diceables:
        record.append(f"cooked__diced__{diceable.split('__')[-1].split('.n.')[0]}.n.01")
        if not (diceable == "half__papaya.n.01") and not (diceable == "sliced__papaya.n.01"): 
            assert f"cooked__diced__{diceable.split('__')[-1].split('.n.')[0]}.n.01" in syns_to_props, f"cooked__diced__{diceable.split('__')[-1].split('.n.')[0]}.n.01" + " " + diceable
    half_to_diced_data.append(record)


diced_derivatives_df = pd.DataFrame(data=sorted(half_to_diced_data, key=lambda x: x[0]), columns=["synset", "uncooked_diceable_derivative_synset", "cooked_diceable_derivative_synset"])
diced_derivatives_df.to_csv("diced_derivatives_df.csv", index=False)



