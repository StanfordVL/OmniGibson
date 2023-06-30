import csv
import json
from enum import IntEnum
from bddl.object_taxonomy import ObjectTaxonomy
import pathlib

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

PARTICLE_SOURCE_MAPPING = {
    "bathtub.n.01": "water",
    "bidet.n.01": "water",
    "sink.n.01": "water",
    "soap_dispenser.n.01": "liquid_soap",
    "tub.n.02": "water",
    "watering_can.n.01": "water",
    "squeeze_bottle.n.01": "water",
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
    print(f"Parsing: {unparsed_conditions}")
    if unparsed_conditions.isnumeric():
        always_true = bool(int(unparsed_conditions))
        conditions = [] if always_true else None
    else:
        conditions = [parse_predicate(predicate=pred) for pred in unparsed_conditions.lower().split(" or ")]
    return conditions

def parse_tm_cleaning_csv():
    synset_cleaning_mapping = dict()

    PROP_PARAM_ANNOTS_DIR = pathlib.Path(__file__).parents[1] / "generated_data" / "prop_param_annots"
    TM_CLEANING_FILE = PROP_PARAM_ANNOTS_DIR / "tm_cleaning.csv"
    REMOVER_SYNSET_MAPPING = pathlib.Path(__file__).parents[1] / "generated_data" / "remover_synset_mapping.json"
    OUTPUT_HIERARCHY_PROPERTIES = pathlib.Path(__file__).parents[1] / "generated_data" / "output_hierarchy_properties.json"

    rows = []
    with open(TM_CLEANING_FILE) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            rows.append(row)

    # Remove first row
    header, rows = rows[1], rows[2:]

    start_idx = 0
    for idx, head in enumerate(header):
        if head == "water.n.06":
            start_idx = idx
            break
    assert start_idx != 0

    for row in rows:
        synset_entry = row[start_idx - 4]
        synset = synset_entry.split(" ")[0]

        if synset == "":
            break

        if "not particleremover" in synset_entry.lower():
            continue

        default_visual_conditions = parse_conditions_entry(row[start_idx - 2])
        default_physical_conditions = parse_conditions_entry(row[start_idx - 1])

        remover_kwargs = {
            "conditions": dict(),
            "default_physical_conditions": default_physical_conditions,
            "default_visual_conditions": default_visual_conditions,
            "method": ParticleModifyMethod.PROJECTION if "vacuum" in synset.lower() else ParticleModifyMethod.ADJACENCY,
        }

        for idx, substance_synset in enumerate(header[start_idx:]):
            # Grab condition
            conditions = parse_conditions_entry(row[start_idx + idx])
            if conditions is not None:
                og_cat = substance_synset.split(".")[0]
                remover_kwargs["conditions"][og_cat] = conditions

        synset_cleaning_mapping[synset] = remover_kwargs

    ot = ObjectTaxonomy()
    pruned_synset_cleaning_mapping = dict()
    for synset, remover_kwargs in synset_cleaning_mapping.items():
        if not ot.is_valid_synset(synset):
            continue
        leaf_synsets = ot.get_leaf_descendants(synset=synset)
        leaf_synsets = [synset] if len(leaf_synsets) == 0 else leaf_synsets
        for leaf_synset in leaf_synsets:
            abilities = ot.get_abilities(leaf_synset)
            if "particleRemover" in abilities:
                pruned_synset_cleaning_mapping[leaf_synset] = remover_kwargs


    with open(REMOVER_SYNSET_MAPPING, "w+") as f:
        json.dump(synset_cleaning_mapping, f, indent=2)


    # Modify the output hierarchy properties
    with open(OUTPUT_HIERARCHY_PROPERTIES, "r") as f:
        ohp = json.load(f)

    not_annotated_removers = set()

    def find_and_replace_synsets_recursively(ohp_root):
        # global pruned_synset_cleaning_mapping, not_annotated_removers
        if isinstance(ohp_root, dict):
            # Leaf node
            if "name" in ohp_root.keys() and "children" not in ohp_root.keys():
                name = ohp_root["name"]
                # Make sure category mapping exists
                if "substance" in ohp_root["abilities"]:
                    correct_category = name.split(".")[0].replace("-", "_")
                    if "categories" in ohp_root:
                        assert len(ohp_root["categories"]) == 1
                        assert ohp_root["categories"][0] == correct_category
                    else:
                        ohp_root["categories"] = [correct_category]
                
                # Make sure particleRemover annotation aligns
                if "particleRemover" in ohp_root["abilities"]:
                    if name not in pruned_synset_cleaning_mapping:
                        print(f"no particleRemover annotated for {name}")
                        not_annotated_removers.add(name)
                if "particleSink" in ohp_root["abilities"]:
                    print(f"Adding particleSink kwargs for: {name}")
                    ohp_root["abilities"]["particleSink"] = {
                        "conditions": {},
                        "default_physical_conditions": [],
                        "default_visual_conditions": None,
                    }
                if "particleApplier" in ohp_root["abilities"]:
                    print(f"Adding particleApplier kwargs for: {name}")
                    # assert len(name.split("__")) > 1
                    system_name = name.split("__")[0]
                    ohp_root["abilities"]["particleApplier"] = {
                        "conditions": {system_name: [(ParticleModifyCondition.GRAVITY, True) if "needsOrientation" in ohp_root["abilities"] else (ParticleModifyCondition.TOGGLEDON, True)]},
                        "method": ParticleModifyMethod.PROJECTION,
                    }
                if "particleSource" in ohp_root["abilities"]:
                    print(f"Adding particleSource kwargs for: {name}")
                    assert name in PARTICLE_SOURCE_MAPPING
                    ohp_root["abilities"]["particleSource"] = {
                        "conditions": {PARTICLE_SOURCE_MAPPING[name]: [(ParticleModifyCondition.GRAVITY, True) if "needsOrientation" in ohp_root["abilities"] else (ParticleModifyCondition.TOGGLEDON, True)]},
                        "method": ParticleModifyMethod.PROJECTION,
                    }
            for k, v in ohp_root.items():
                if k == "name" and v in pruned_synset_cleaning_mapping:
                    # print(f"found: {v}")
                    assert "children" not in ohp_root.keys()
                    assert "particleRemover" in ohp_root["abilities"]
                    ohp_root["abilities"]["particleRemover"] = pruned_synset_cleaning_mapping[v]
                    break
                elif k == "children":
                    for child in v:
                        find_and_replace_synsets_recursively(ohp_root=child)
                else:
                    find_and_replace_synsets_recursively(ohp_root=v)

    find_and_replace_synsets_recursively(ohp_root=ohp)

    with open(OUTPUT_HIERARCHY_PROPERTIES, "w+") as f:
        json.dump(ohp, f, indent=2)


if __name__ == "__main__":
    parse_tm_cleaning_csv()
