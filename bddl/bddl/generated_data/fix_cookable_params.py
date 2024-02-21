import json 
import pdb
import pandas as pd 

with open("cookable_noparams.json", "r") as f:
    cookable_noparams = json.load(f)

cookable_params = pd.read_csv("prop_param_annots/cookable.csv")
cookable_to_cooktemp = dict(zip(cookable_params["synset"], cookable_params["cook_temperature"]))
lemma_to_cooktemp = {synset.split(".")[0]: cooktemp for synset, cooktemp in cookable_to_cooktemp.items()}
# pdb.set_trace()

with open("cookable_noparams.json", "r") as f:
    cookable_noparams = json.load(f)

records = []
for syn in cookable_noparams:
    lemma = syn.split("__")[-1].split(".")[0]
    if lemma in [
        "melon", 
        "onion", 
        "pomegranate", 
        "pomelo", 
        "roast_beef", 
        "tomato"
    ]:
        print(syn)
        continue
    record = {
        "synset": syn,
        "cook_temperature": lemma_to_cooktemp[lemma]
    }
    records.append(record)

cooktemp_df = pd.DataFrame(columns=["synset", "cook_temperature"], data=records)
cooktemp_df.to_csv("cooktemp_annots.csv", index=False)