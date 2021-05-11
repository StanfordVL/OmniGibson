import os
import csv
import xml.etree.ElementTree as ET
from collections import Counter

import gibson2
import tasknet

from gibson2.objects.articulated_object import URDFObject
from gibson2.utils import urdf_utils
from gibson2.utils.assets_utils import get_ig_category_path
from IPython import embed
import json

import hierarchy_generator


INPUT_SYNSET_FILE = os.path.join(os.path.dirname(
    tasknet.__file__), '..', 'utils', 'synsets_to_filtered_properties.json')
MODELS_CSV_PATH = os.path.join(os.path.dirname(
    tasknet.__file__), '..', 'utils', 'objectmodeling.csv')
OUTPUT_SYNSET_FILE = os.path.join(os.path.dirname(
    tasknet.__file__), '..', 'utils', 'synsets_to_filtered_properties_pruned_igibson.json')

NON_MODEL_CATEGORIES = ["floor"]

def get_categories():
    obj_dir = os.path.join(gibson2.ig_dataset_path, 'objects')
    return [cat for cat in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, cat))]

def categories_to_synsets(categories):
    with open(MODELS_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        cat_to_syn = {row["Object"].strip(): row["Synset"].strip() 
                      for row in reader}
    synsets = []
    for cat in categories:
        # Check that the category has this label.
        class_name = cat_to_syn[cat]
        assert class_name is not None, cat
        synsets.append(class_name)
    return synsets


def prune_openable():
    '''
    Returns a list of categories in iGibson that has openable.
    '''
    # Require all models of the category to have revolute or prismatic joints
    allowed_joints = frozenset(["revolute", "prismatic"])
    with open(MODELS_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        all_categories = [row["Object"].strip() for row in reader]
    allowed_categories = []
    # for cat in categories:
    for cat in all_categories:
        if cat in NON_MODEL_CATEGORIES:
            continue
        cat_dir = get_ig_category_path(cat)
        success = True
        for obj_name in os.listdir(cat_dir):
            obj_dir = os.path.join(cat_dir, obj_name)
            urdf_file = os.path.join(obj_dir, obj_name + '.urdf')
            tree = ET.parse(urdf_file)
            joints = [joint for joint in tree.findall('joint')
                      if joint.attrib['type'] in allowed_joints]
            if len(joints) == 0:
                success = False
                break
        if success:
            allowed_categories.append(cat)
    # Manuall remove toilet because even thought they have joints, they can't necessarily open the lid of the toilet.
    skip_openable = [
        "toilet",
        "console_table",
        "monitor",
        "standing_tv",
        "coffee_maker"
    ]
    for skip_category in skip_openable:
        if skip_category in allowed_categories:
            allowed_categories.remove(skip_category)

    return allowed_categories


def prune_heat_source():
    # Heat sources are confined to kitchen appliance that we have articulated models for
    allowed_categories = [
        'microwave',
        'stove',
        'oven',
    ]
    return allowed_categories


def prune_water_source():
    # Water sources are confined to only sink for now. May add bathtub later?
    allowed_categories = [
        'sink',
    ]
    return allowed_categories


def prune_sliceable():
    # Sliceable are confined to objects that we have half_* models for
    allowed_categories = []
    for cat in get_categories():
        if 'half_' in cat:
            allowed_categories.append(cat.replace('half_', ''))
    return allowed_categories

def get_owned_hierarchy():
    '''
    Builds a simply synset hierarchy using owned models. We can use 
    this to find the leaf nodes.
    '''
    owned_synsets = {}
    with open(MODELS_CSV_PATH) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            synset = row["Synset"].strip()
            obj = row["Object"].strip()
            if synset in owned_synsets:
                owned_synsets[synset].append(obj)
            else:
                owned_synsets[synset] = [obj]

    # Every synset we have should theoretically lead up to `entity.n.01`.
    hierarchy = {"name": 'entity.n.01', "children": []}

    for synset in owned_synsets:
        synset = wn.synset(synset)
        synset_paths = []
        hierarchy_generator.generate_paths(synset_paths, [synset], synset)
        for synset_path in synset_paths:
            # The last word should always be `entity.n.01`, so we can just take it out.
            hierarchy_generator.add_path(synset_path[:-1], hierarchy)

    return hierarchy
    

def get_leaf_synsets(hierarchy, leaf_synsets):
    '''
    Goes through a synset hierarchy and adds all of the
    leaf node synsets into a set.
    '''
    if "children" not in hierarchy:
        leaf_synsets.add(hierarchy["name"])
    else:
        for sub_hierarchy in hierarchy["children"]:
            get_leaf_synsets(sub_hierarchy, leaf_synsets)

def update_synsets_to_properties(hierarchy, synsets_to_properties):
    if "children" not in hierarchy:
        if synsets_to_properties[hierarchy["name"]] != hierarchy["abilities"]:
            print(f"{hierarchy['name']} has conflicting properties. Please investigate.")
    else:
        if hierarchy["name"] in synsets_to_properties:
            synsets_to_properties[hierarchy["name"]] = hierarchy["abilities"]
            print(f"Abilities updated for {hierarchy['name']}")
        for sub_hierarchy in hierarchy["children"]:
            update_synsets_to_properties(sub_hierarchy, synsets_to_properties)

def main():
    properties_to_synsets = {}
    properties_to_synsets['openable'] = categories_to_synsets(prune_openable())
    properties_to_synsets['heatSource'] = categories_to_synsets(
        prune_heat_source())
    properties_to_synsets['waterSource'] = categories_to_synsets(
        prune_water_source())
    properties_to_synsets['sliceable'] = categories_to_synsets(
        prune_sliceable())

    with open(INPUT_SYNSET_FILE) as f:
        synsets_to_properties = json.load(f)

    # We build a hierarchy of only owned models, but with oracle properties because
    #   igibson properties are not yet available. 
    hierarchy = hierarchy_generator.generate_hierarchy("owned", "oracle")
    # We store a set of leaf synset nodes.
    leaf_synsets = set()
    get_leaf_synsets(hierarchy, leaf_synsets)

    for synset in synsets_to_properties:
        curr_properties = synsets_to_properties[synset]
        for prop in properties_to_synsets:
            # We add a property to a synset if iGibson has it but it is not in the annotation.
            # TODO: Should we update the oracle json file then?
            if synset in properties_to_synsets[prop] and prop not in curr_properties:
                assert synset in leaf_synsets, f"Synset {synset} not in leaf synsets"
                curr_properties[prop] = {}
                print('add', synset, prop)
            # We remove a property from a synset if:
            # 1. The synset does not have this property in iGibson.
            # 2. The annotation has the property.
            # 3. The synset is a leaf node in the hierarchy.
            elif synset not in properties_to_synsets[prop] and prop in curr_properties and synset in leaf_synsets:
                curr_properties.pop(prop)
                print('remove', synset, prop)

    # We propogate the leaf nodes' updated properties up the hierarchy.
    hierarchy_generator.add_abilities(hierarchy, ability_map=synsets_to_properties)
    # We go through the hierarchy to update synsets_to_properties.
    update_synsets_to_properties(hierarchy, synsets_to_properties)

    with open(OUTPUT_SYNSET_FILE, 'w') as f:
        json.dump(synsets_to_properties, f, indent=2)

if __name__ == "__main__":
    main()
