import pandas as pd 
import os

tasklist = pd.read_csv("b1k_master_planning.csv")
invalid_tasks = set(os.listdir("."))
invalid_tasks.remove("domain_igibson.bddl")
invalid_tasks.remove("domain_omnigibson.bddl")
invalid_tasks.remove("b1k_master_planning.csv")
invalid_tasks.remove("remove_unwanted_tasks.py")
invalid_tasks.remove("non_b1k_tasks")
original_length = len(invalid_tasks)

not_in_folder = set()

num_approved_tasks = 0
for __, [task_name, approve1, approve2, approve3, note, *__] in tasklist.iterrows():
    # print(task_name)
    task_internal_name = "_".join("".join("_or_".join("".join("_".join(task_name.split(" ")).split(",")).split("/")).split("'")).split("-"))
    # print(note)
    if (pd.isna(note)) or (("remove" not in note) and ("duplicate" not in note)) and (approve3 == 1):
        # print()
        # print(task_internal_name)
        # print("crossed first gate")
        if task_internal_name in invalid_tasks:
            invalid_tasks.remove(task_internal_name)
        else:
            not_in_folder.add(task_internal_name)

print("Num approved tasks in folder:", original_length - len(invalid_tasks))
from pprint import pprint
pprint(invalid_tasks)
print()
pprint(not_in_folder)
print("Num valid tasks not in definitions folder:", len(not_in_folder))