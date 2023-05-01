"""
Purpose:
This is the script that generates the 'hierarchy.json' file that will be necessary
for many purposes like initial/goal state annotation, object shopping list annotation,
and scene generation in iGATUS. 

Generate 3 types of hierarchy.json:
    - Hierarchy of just the owned models for scene generation and object sampling.
    - Hierarchy of the union of all of the owned models + objects extracted from online articles.
    - Hierarchy of just the objects from online articles, this is our most unbiased object distribution.

To Run:
- Make sure that the .csv files and the .json file are updated (see ### Dependencies).
- python3 hierarchy_generator.py

Last Change: 05/05/2021
Created By: Zheng Lian & Cem Gokmen
"""

import csv
import json
import os
from collections import OrderedDict
import pandas as pd
import copy

from nltk.corpus import wordnet as wn

### Dependencies
'''
This .csv file should contain all of the b100 models we currently own.
Should contain an `Object` column and a `Synset` column.
'''
MODELS_CSV_PATH = "objectmodeling.csv"
'''
This .csv file should contain all of the objects we extracted from
online articles.
Should contain an `Object` column and a `Synset` column.
'''
OBJECT_STATS_PATH = "object_stats.csv"
'''
This .csv file should contain all of the objects and words used 
in B-1K. Should contain a `synset` column and a `words` column.
'''
B1K_SYNSET_MASTERLIST = "b1k_synset_masterlist.tsv"
'''
This .csv file should contain all of the objects and words modeled
in B-1K. Should contain a `category` column and a `synset` column.
'''
B1K_MODELED_SYNSET_MASTERLIST = "b1k_objectmodeling.csv"
'''
This .json file should contain all of the synsets from the .csv files above
as well as their associated iGibson abilities.
NOTE: Please contact Sanjana (sanjana2@stanford.edu) or Zheng (zhengl@stanford.edu) if
any of the property annotations is missing.
'''
IGIBSON_ABILITY_JSON_PATH = "synsets_to_filtered_properties_pruned_igibson.json"
ORACLE_ABILITY_JSON_PATH = "synsets_to_filtered_properties.json"
B1K_ABILITY_JSON_PATH = "synsets_to_filtered_properties_b1k.json"
'''
This .csv file should contain all of the custom synsets (synsets that do not
exist in WordNet) that may be in the activities being loaded, as well as their
assigned hypernym. Should contain columns `word`, `custom_synset`, and 
`hypernyms`.
'''
CUSTOM_SYNSETS = "custom_synsets.csv"

# Uses iGibson abilities.
OUTPUT_JSON_PATH1 = os.path.join(os.path.dirname(__file__), "..", "bddl", "hierarchy_owned.json")
# Uses oracle abilities.
OUTPUT_JSON_PATH2 = os.path.join(os.path.dirname(__file__), "..", "bddl", "hierarchy_articles.json")
# Uses oracle abilities.
OUTPUT_JSON_PATH3 = os.path.join(os.path.dirname(__file__), "..", "bddl", "hierarchy_all.json")
# Uses B-1K abilities
OUTPUT_JSON_PATH4 = os.path.join(os.path.dirname(__file__), "..", "bddl", "hierarchy_b1k.json")
# Uses both B-1K and B-100 abilities, with common synsets taking from B-1K
OUTPUT_JSON_PATH5 = os.path.join(os.path.dirname(__file__), "..", "bddl", "hierarchy_b1k_modeled.json")

'''
Load in all of the owned models. Map the synsets to their corresponding object names.
'''
owned_synsets = {}
with open(MODELS_CSV_PATH) as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        synset = row["Synset"].strip()
        obj = row["Object"].strip()
        if synset in owned_synsets:
            owned_synsets[synset]["objects"].append(obj)
        else:
            owned_synsets[synset] = {"objects": [obj]}

'''
Load in all of the synsets that appeared in articles.
'''
article_synsets = {}
with open(OBJECT_STATS_PATH) as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        synset = row["Synset"].strip()
        synset = synset[synset.find('\'')+1: synset.rfind('\'')]
        obj = row["Object"].strip()
        if synset in article_synsets:
            article_synsets[synset]["objects"].append(obj)
        else:
            article_synsets[synset] = {"objects": [obj]}

'''
Load in all of the synsets from B-1K
'''
b1k_synset_df = pd.read_csv(B1K_SYNSET_MASTERLIST, sep="\t")
b1k_synsets = {}
for i, [synset, words, *__] in b1k_synset_df.iterrows():
    b1k_synsets[synset] = {"objects": 
        json.loads(words.replace("'", '"')) if not pd.isna(words) else []}

'''
Load in all of the owned synsets from B-1K, plus substances since they 
don't require a model
'''
b1k_modeled_synset_df = pd.read_csv(B1K_MODELED_SYNSET_MASTERLIST)
b1k_modeled_synsets = {}
for __, [__, category, __, synset, *__] in b1k_modeled_synset_df.iterrows():
    b1k_modeled_synsets[synset] = {"objects": [category]}
with open(B1K_ABILITY_JSON_PATH, "r") as f:
    b1k_syns_to_props = json.load(f)
try:
    b1k_modeled_synsets.update(
        {syn: objs for syn, objs in b1k_synsets.items() if "substance" in b1k_syns_to_props[syn]})
except KeyError as e:
    print(f"{e} not in synset-to-filtered-property file")

'''
Synsets from B-1K and owned B-100 models
'''
b1k_modeled_synsets = copy.deepcopy(owned_synsets)
b1k_modeled_synsets.update(b1k_modeled_synsets)

'''
Combined version of owned and article.
'''
all_synsets = {key: value for key, value in owned_synsets.items()}
for synset in article_synsets:
    if synset not in all_synsets:
        all_synsets[synset] = article_synsets[synset]
    else:
        all_synsets[synset] = list(set(all_synsets[synset]) | set(article_synsets[synset]))


######################################################

def add_path(path, node, custom_synsets):
    """
    Takes in list of synsets and add it into the overall hierarchy.

    @param path: List of strings, where each string is a synset.
      - The list is in the form of [child, parent, grandparent, ...]
    @param node: Dictionary, contains the overall hierarchy we are building
    @param custom_synsets: Dictionary, maps name to hypernym name
    """
    if not path:
        return
    # Take out the oldest ancestor in the path.
    oldest_synset = path[-1]

    try:
        name = oldest_synset.name()
    except:
        name = oldest_synset[8:-2]
        if "children" not in node:
            node["children"] = []
        for child_node in node["children"]:
            if child_node["name"] == name:
                # child_node["hasModel"] = True 
                add_path(path[:-1], child_node, custom_synsets)
                return 
        # node["children"].append({"name": name, "hasModel": True})
        node["children"].append({"name": name})
        add_path(path[:-1], node["children"][-1], custom_synsets)
    else:
        # If the current node never had a child before, initialize a list to store them.
        if "children" not in node:
            node["children"] = []
        # Look for an existing child that matches the current synset.
        for child_node in node["children"]:
            if child_node["name"] == name:
                # child_node["hasModel"] = True
                add_path(path[:-1], child_node, custom_synsets)
                return 
            # node["children"].append({"name": name, "hasModel": True})
        # If this is a child we have yet to find, append it into our hierarchy.
        node["children"].append({"name": name})
        add_path(path[:-1], node["children"][-1], custom_synsets)


def generate_paths(paths, path, word, custom_synsets):
    """
    Given a synset, generate all paths this synset can take up to 'entity.n.01'.

    @param paths, List of lists, will be populated with the paths we create here.
    @param path: List, the current path we are building.
    @param word: The current synset we are searching parents for.
    @param custom_synsets: Dictionary, maps name to hypernym name
    """
    # if str(word)[8:-2] in custom_synsets:
    #     hypernyms = wn.synset(custom_synsets[word[8:-2]]["hypernyms"])
    #     generate_paths(paths, path + [hypernyms], hypernyms, custom_synsets)
    # else:
    #     hypernyms = word.hypernyms()
    #     if not hypernyms:
    #         paths.append(path)
    #     else:
    #         for parent in hypernyms:
    #             generate_paths(paths, path + [parent], parent, custom_synsets)
    try:
        if str(word[8:-2]) in custom_synsets:
            pass 
    except:
        hypernyms = word.hypernyms()
        if not hypernyms:
            paths.append(path)
        else:
            for word in hypernyms:
                generate_paths(paths, path + [word], word, custom_synsets)
    else:
        if str(word[8:-2]) not in custom_synsets:       # TODO remove when new custom synsets have been added
            pass
        else:
            hypernyms = wn.synset(custom_synsets[word[8:-2]]["hypernyms"])
            generate_paths(paths, path + [hypernyms], hypernyms, custom_synsets)

'''
Below is the script that creates the .json hierarchy
'''

def add_igibson_objects(node, synsets):
    '''
    Go through the hierarchy and add the words associated with the synsets as attributes.
    '''
    categories = []
    if node["name"] in synsets:
        categories = synsets[node["name"]]["objects"]

    node["igibson_categories"] = sorted(categories)

    if "children" in node:
        for child_node in node["children"]:
            add_igibson_objects(child_node, synsets)


def add_abilities(node, ability_type=None, ability_map=None):
    if ability_type is None and ability_map is None:
        raise ValueError("No abilities specified. Abilities can be specified through the ability_type kwarg to get a pre-existing ability map, or the ability_map kwarg to override with custom abilities.")
    if ability_map is None: 
        if ability_type == "igibson":
            with open(IGIBSON_ABILITY_JSON_PATH) as f:
                ability_map = json.load(f)
        elif ability_type == "oracle": 
            with open(ORACLE_ABILITY_JSON_PATH) as f:
                ability_map = json.load(f)
        elif ability_type == "b1k":
            with open(B1K_ABILITY_JSON_PATH) as f:
                ability_map = json.load(f)
        elif ability_type == "b1k_modeled": 
            with open(IGIBSON_ABILITY_JSON_PATH) as f:
                ability_map = json.load(f)
            with open(B1K_ABILITY_JSON_PATH) as f:
                b1k_ability_map = json.load(f)
            ability_map.update(b1k_ability_map)
        else:
            raise ValueError("Invalid ability type given.")

    # At leaf
    if "children" not in node:
        name = node["name"]
        if name in ability_map:
            if isinstance(ability_map[name], dict):
                abilities = ability_map[name]
            else:
                # Support legacy format ability annotations where params are not
                # supported and abilities are in list format.
                abilities = {ability: dict() for ability in ability_map[name]}

            node["abilities"] = OrderedDict(sorted(abilities.items(), key=lambda pair: pair[0]))
            return abilities
        else:
            node["abilities"] = OrderedDict()
            print(f"{name} not found in ability list!")
            return None
    else:
        init = False
        abilities = {}
        for child_node in node["children"]:
            child_abilities = add_abilities(child_node, ability_map=ability_map)
            if child_abilities is not None:
                if init:
                    # First merge the ability annotations themselves
                    common_keys = set(abilities.keys()) & set(child_abilities.keys())

                    # Then add the ability annotations & merge common parameters
                    new_abilities = {}
                    for ability_key in common_keys:
                        current_params = set(abilities[ability_key].items())
                        child_params = set(
                            child_abilities[ability_key].items())

                        # Note that this intersection finds pairs where both the param
                        # key and the param value are equal.
                        common_params = current_params & child_params

                        new_abilities[ability_key] = dict(common_params)

                    abilities = new_abilities
                else:
                    abilities = child_abilities
                    init = True
        node["abilities"] = OrderedDict(sorted(abilities.items(), key=lambda pair: pair[0]))
        return abilities

def generate_hierarchy(hierarchy_type, ability_type):
    # Every synset we have should theoretically lead up to `entity.n.01`.
    hierarchy = {"name": 'entity.n.01', "children": []}

    if hierarchy_type == "owned": 
        synsets = owned_synsets
    elif hierarchy_type == "article":
        synsets = article_synsets
    elif hierarchy_type == "all": 
        synsets = all_synsets
    elif hierarchy_type == "b1k":
        synsets = b1k_synsets
    elif hierarchy_type == "b1k_modeled":
        synsets = b1k_modeled_synsets
    else:
        raise ValueError("Invalid hierarchy type given.")
    
    custom_synsets = {}
    with open(CUSTOM_SYNSETS, "r") as custom_map:
        output = csv.DictReader(custom_map)
        for row in output: 
            custom_synsets[row["custom_synset"]] = {
                "hypernyms": row["hypernyms"],
                "objects": [row["word"]]
            }
    
    synsets.update(custom_synsets)

    objects = set()
    for synset in synsets: 
        try:
            syn = wn.synset(synset)
        except:
            syn = f"Synset('{synset}')"
        synset_paths = []
        objects.add(synset)
        generate_paths(synset_paths, [syn], syn, custom_synsets)
        for synset_path in synset_paths:
            # The last word should always be `entity.n.01`, so we can just take it out.
            add_path(synset_path[:-1], hierarchy, custom_synsets)

    add_igibson_objects(hierarchy, synsets)
    add_abilities(hierarchy, ability_type=ability_type)
    return hierarchy

def save_hierarchies():
    """Save all three hierarchy types 
    """
    # hierarchy_owned = generate_hierarchy("owned", "b1k")
    # with open(OUTPUT_JSON_PATH1, "w") as f:
    #     json.dump(hierarchy_owned, f, indent=2)

    # hierarchy_articles = generate_hierarchy("article", "b1k")
    # with open(OUTPUT_JSON_PATH2, "w") as f:
    #     json.dump(hierarchy_articles, f, indent=2)

    # hierarchy_all = generate_hierarchy("all", "b1k")
    # with open(OUTPUT_JSON_PATH3, "w") as f:
    #     json.dump(hierarchy_all, f, indent=2)
    
    # hierarchy_b1k = generate_hierarchy("b1k", "b1k")
    # with open(OUTPUT_JSON_PATH4, "w") as f:
    #     json.dump(hierarchy_b1k, f, indent=2)
    
    hierarchy_b1k_modeled = generate_hierarchy("b1k_modeled", "b1k_modeled")
    with open(OUTPUT_JSON_PATH5, "w") as f:
        json.dump(hierarchy_b1k_modeled, f, indent=2)


if __name__ == "__main__":
    save_hierarchies()

