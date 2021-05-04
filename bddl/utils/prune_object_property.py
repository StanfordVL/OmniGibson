import os
import xml.etree.ElementTree as ET
from collections import Counter

import gibson2
import tasknet
from tasknet.object_taxonomy import ObjectTaxonomy

from gibson2.objects.articulated_object import URDFObject
from gibson2.utils import urdf_utils
from gibson2.utils.assets_utils import get_ig_category_path
from IPython import embed
import json


OBJECT_TAXONOMY = ObjectTaxonomy()

SYNSET_FILE = os.path.join(os.path.dirname(
    tasknet.__file__), '..', 'utils', 'synsets_to_filtered_properties.json')


def get_categories():
    obj_dir = os.path.join(gibson2.ig_dataset_path, 'objects')
    return [cat for cat in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, cat))]


def get_category_with_ability(ability):
    categories = []
    for cat in get_categories():
        # Check that the category has this label.
        class_name = OBJECT_TAXONOMY.get_class_name_from_igibson_category(cat)
        if not class_name:
            continue

        if not OBJECT_TAXONOMY.has_ability(class_name, ability):
            continue

        categories.append(cat)
    return categories


def categories_to_synsets(categories):
    synsets = []
    for cat in categories:
        # Check that the category has this label.
        class_name = OBJECT_TAXONOMY.get_class_name_from_igibson_category(cat)
        assert class_name is not None
        synsets.append(class_name)
    return synsets


def prune_openable():
    # Require all models of the category to have revolute or prismatic joints
    allowed_joints = frozenset(["revolute", "prismatic"])
    categories = get_category_with_ability('openable')
    print('openable', categories)
    allowed_categories = []
    for cat in categories:
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
    if 'toilet' in allowed_categories:
        allowed_categories.remove('toilet')

    return allowed_categories


def prune_heat_source():
    # Heat sources are confined to kitchen appliance that we have articulated models for
    categories = get_category_with_ability('heatSource')
    print('heatSource', categories)
    allowed_categories = [
        'microwave',
        'stove',
        'oven',
    ]
    return allowed_categories


def prune_water_source():
    # Water sources are confined to only sink for now. May add bathtub later?
    categories = get_category_with_ability('waterSource')
    print('waterSource', categories)
    allowed_categories = [
        'sink',
    ]
    return allowed_categories


def prune_sliceable():
    # Sliceable are confined to objects that we have half_* models for
    categories = get_category_with_ability('sliceable')
    print('sliceable', categories)
    allowed_categories = []
    for cat in get_categories():
        if 'half_' in cat:
            allowed_categories.append(cat.replace('half_', ''))
    return allowed_categories


def main():
    properties_to_synsets = {}
    properties_to_synsets['openable'] = categories_to_synsets(prune_openable())
    properties_to_synsets['heatSource'] = categories_to_synsets(
        prune_heat_source())
    properties_to_synsets['waterSource'] = categories_to_synsets(
        prune_water_source())
    properties_to_synsets['sliceable'] = categories_to_synsets(
        prune_sliceable())

    with open(SYNSET_FILE) as f:
        obj = json.load(f)

    for synset in obj:
        curr_properties = obj[synset]
        for prop in properties_to_synsets:
            if synset in properties_to_synsets[prop] and prop not in curr_properties:
                curr_properties.append(prop)
                print('add', synset, prop)
            elif synset not in properties_to_synsets[prop] and prop in curr_properties:
                curr_properties.remove(prop)
                print('remove', synset, prop)

    with open(SYNSET_FILE, 'w+') as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    main()
