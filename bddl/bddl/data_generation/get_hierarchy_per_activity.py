'''
Iteration one: 
Strict multi-way tree (without no count)
We consider each unique path to a synset "different", with the obvious
problem that each synset can have multiple hypernyms
'''

import csv
import pathlib
from nltk.corpus import wordnet as wn
import pprint
import json
import glob

DATA_ROOT = pathlib.Path(__file__).parents[1] / "generated_data"
ACTIVITIES_DICT = DATA_ROOT / "activities_to_synset_lists.json"
CUSTOM_SYNSETS = DATA_ROOT / "custom_synsets.csv"
OBJECT_PATH = DATA_ROOT / 'owned_models.csv'
JSON_DIRECTORY = DATA_ROOT / 'hierarchy_per_activity'

def load_activities(): # Loads activities_to_synset_lists.json as dictionary, return dictionary
  with open(ACTIVITIES_DICT) as dict:
    activities_dict = json.load(dict)
  return activities_dict

def load_customs(): # creates dictionary of custom synset keys matched to hypernym values
  custom_synsets = {}
  with open(CUSTOM_SYNSETS) as custom_map:
    output = csv.DictReader(custom_map)
    for row in output:
      custom_synsets[row["custom_synset"]] = row['hypernyms']
  return custom_synsets

def set_owned_synsets(): # creates a set of all synsets we have models for
  owned_synsets = set()
  with open(OBJECT_PATH) as csvfile: 
    reader = csv.DictReader(csvfile) # opening csv with dictreader
    for row in reader:
      # if row["hasModel?"].strip() == "1":
      owned_synsets.add(row["Synset"].strip()) # populating type:set owned_synsets() by grabbing the value of key synset for each row, and eliminating extraneous spaces"""
  return owned_synsets

def add_path(path, hierarchy,custom_synsets):
  if not path: # if path is false
    return
  synset = path[-1]
  try: # catch crashes of trying to use custom_synset elements (strings) as synset-class-objects
    name = synset.name()
  except: 
    name = synset[8:-2]
    if "children" not in hierarchy:
      hierarchy["children"] = []
    for subhierarchy in hierarchy["children"]:
      if subhierarchy["name"] == name:
        subhierarchy["hasModel"] = True # set hasModel to true
        add_path(path[:-1], subhierarchy,custom_synsets) # iteratively run this function until reaching the end of the branches
        return
    hierarchy["children"].append({"name": name, "hasModel": True})
    add_path(path[:-1], hierarchy["children"][-1],custom_synsets)
  else:
    name = synset.name() # this is like "water.n.06" from 'water'
    if "children" not in hierarchy:
      hierarchy["children"] = []
    for subhierarchy in hierarchy["children"]: # for every subhierarchy of hierarchy
      if subhierarchy["name"] == name: # if the name of the subhierarchy is a name we have in owned_models
        subhierarchy["hasModel"] = True # set hasModel to true
        add_path(path[:-1], subhierarchy,custom_synsets) # recursively run this function until reaching the end of the branches
        return
    hierarchy["children"].append({"name": name, "hasModel": True})
    add_path(path[:-1], hierarchy["children"][-1],custom_synsets)
  """if not path: # if there isn't a path, don't do anything
    return
  synset = path[-1] # the synset is the last term in the path list
  name = synset.name() # recover name of the synset
  #if name == "rack.n.01": # set hasRack to true, if the name is rack
  #  hasRack[0] = True
  if "children" not in hierarchy: # if the hierarchy doesn't say children
    hierarchy["children"] = [] # create any empty value with key "childredn"
  for subhierarchy in hierarchy["children"]:
    if subhierarchy["name"] == name: # if the subhierarchy section matches the name of the synset
      if hasModel: # and if we have the model
        subhierarchy["hasModel"] = True # if we have the model for something a step up the ladder, then we have the model for anything downstream
      add_path(path[:-1], subhierarchy, hasModel) # perform add_path on the next step down
      return
  hierarchy["children"].append({"name": name})
  hierarchy["children"][-1]["hasModel"] =  hasModel
  add_path(path[:-1], hierarchy["children"][-1], hasModel)"""

def generate_paths(paths, path, word, custom_synsets):
  try: # should only work for customs
    if str(word[8:-2]) in custom_synsets:
        pass
  except: # for normal synsets
    hypernyms = word.hypernyms() 
    if not hypernyms: # if there aren't any hypernyms append path as it is
      paths.append(path)
    else:
      for word in hypernyms: # generate a path for each hypernym
        generate_paths(paths, path + [word], word, custom_synsets) # calls itself
  else:
    hypernyms = wn.synset(custom_synsets[word[8:-2]])
    generate_paths(paths, path + [hypernyms], hypernyms, custom_synsets)
    
def generate_all_activities(dict_in, owned_synsets, custom_synsets):
  for key in dict_in:
    synsets_in_activity = []
    activity = key
    for synset in dict_in[key]:
      synsets_in_activity.append(synset)
    add_one_activity(activity, synsets_in_activity, owned_synsets, custom_synsets)

def add_one_activity(activity, synsets_in_activity, owned_synsets, custom_synsets):
  hierarchy = {"name": 'entity.n.01', "children": [], "hasModel": True} # define hierarchy
  for synset in synsets_in_activity: # for every synset in the list we just compiled
    try:
      word = wn.synset(synset) # word is the type:synset of a synset-formatted string
    except:
      word = "Synset('" + synset + "')"
    else:
      word = wn.synset(synset)
    paths = [] # resetting paths to 0
    generate_paths(paths, [word], word, custom_synsets) # creates all potential paths and selects one
    hasModel = False # defaults hasModel to False
    if synset in owned_synsets: # unless we own it
      hasModel = True # checking if we have a model
    # We will only take the first path. # how is the first path determined?
    for path in paths[:1]:
      add_path(path[:-1], hierarchy, custom_synsets) # add a slice of the path from beginning to -1
    JSON_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with open(JSON_DIRECTORY / f"{activity}.json", "w") as f:
      json.dump(hierarchy, f, indent=2)


# API

def create_save_activity_specific_hierarchies():
  activities_dictionary = load_activities()
  customs_dict = load_customs()
  owned_models = set_owned_synsets()
  generate_all_activities(activities_dictionary, owned_models, customs_dict)


if __name__ == "__main__":
  activities_dictionary = load_activities()
  customs_dict = load_customs()
  owned_models = set_owned_synsets()
  generate_all_activities(activities_dictionary, owned_models,customs_dict)
