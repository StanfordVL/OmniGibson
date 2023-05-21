'''
Iteration one: 
Strict multi-way tree (without no count)
We consider each unique path to a synset "different", with the obvious
problem that each synset can have multiple hypernyms
'''
import json
from nltk.corpus import wordnet as wn

HIERARCHY_OUTPUT_FN = "output_hierarchy.json"


def add_path(path, hierarchy):
  if not path: 
    return 
  # print("Path is nontrivial")
  synset = path[-1]
  if "children" not in hierarchy:
    hierarchy["children"] = []
  for subhierarchy in hierarchy["children"]:
    if subhierarchy["name"] == synset:
      subhierarchy["hasModel"] = True 
      add_path(path[:-1], subhierarchy)
      return 
  hierarchy["children"].append({"name": synset, "hasModel": True})
  add_path(path[:-1], hierarchy["children"][-1])
  

def generate_paths(paths, path, synset, syn_prop_dict):
  # Annotated as not custom, or not present (i.e. not leaf, i.e. not custom)
  if (synset not in syn_prop_dict) or (not syn_prop_dict[synset]["is_custom"]):
    hypernyms = wn.synset(synset).hypernyms()   # This fails iff a custom synset is incorrectly stored as non-custom
    if not hypernyms:
      paths.append(path)
    else:
      for hypernym in hypernyms:
        generate_paths(paths, path + [hypernym.name()], hypernym.name(), syn_prop_dict)
  else:
    hypernym = syn_prop_dict[synset]["hypernym"]
    # NOTE assumes every custom synset's hypernym is a WordNet synset
    generate_paths(paths, path + [hypernym], hypernym, syn_prop_dict)


# API

def get_hierarchy(syn_prop_dict): 
  hierarchy = {"name": 'entity.n.01', "children": [], "hasModel": False}
  objects = set()
  for synset in syn_prop_dict.keys():
    # NOTE skip custom synsets without hypernym annotations
    if syn_prop_dict[synset]["hypernym"] == "":
      continue
    objects.add(synset)
    paths = []
    generate_paths(paths, [synset], synset, syn_prop_dict)
    for path in paths: 
      add_path(path[:-1], hierarchy)

  with open(HIERARCHY_OUTPUT_FN, "w") as f:
    json.dump(hierarchy, f, indent=2)
  return hierarchy


if __name__ == "__main__":
  pass

"""
Graph using http://bl.ocks.org/eyaler/10586116#graph.json
"""