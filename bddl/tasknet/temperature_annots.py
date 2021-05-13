import json 
import pandas 
from tasknet.object_taxonomy import ObjectTaxonomy

object_taxonomy = ObjectTaxonomy()

def get_leaf_synsets(synsets):
    # return [synset for synset in synsets if not object_taxonomy.get_descendants(synset)]
    leaf_syns = []
    missing_syns = []
    for synset in synsets:
        try:
            if not object_taxonomy.get_descendants(synset):
                leaf_syns.append(synset)
        except AssertionError:
            missing_syns.append(synset)
    
    return leaf_syns, missing_syns


with open("../utils/synsets_to_filtered_properties.json", "r") as f:
    syns_to_props = json.load(f)

cookable_burnable = []
cookable = []
burnable = []
for synset, props in syns_to_props.items():
    if "cookable" in props and "burnable" in props:
        cookable_burnable.append(synset)
    if "cookable" in props and not "burnable" in props:
        cookable.append(synset)
    if not "cookable" in props and "burnable" in props:
        burnable.append(synset)

leaf_cookable_burnable, missing = get_leaf_synsets(cookable_burnable)
leaf_cookable, missing2 = get_leaf_synsets(cookable)
leaf_burnable, missing3 = get_leaf_synsets(burnable)

print(len(leaf_cookable + leaf_cookable_burnable))
print(len(leaf_burnable))
# with open("temperature_annots_needed.json", "w") as f:

cookables_augmented = leaf_cookable + leaf_cookable_burnable
burnables_augmented = leaf_burnable
num_cookables = len(cookables_augmented)
num_burnables = len(burnables_augmented)
if num_burnables < num_cookables:
    burnables_augmented = burnables_augmented + [0 for __ in range(num_cookables - num_burnables)]
else:
    cookables_augmented = cookables_augmented + [0 for __ in range(num_burnables - num_cookables)]

data = {"cookable": cookables_augmented, "burnable": burnables_augmented}
df = pandas.DataFrame.from_dict(data)
df.to_csv("temperature_annots_needed.csv")


# with open("missing_synsets.json", "w") as f:
#     json.dump(missing, f, indent=4)