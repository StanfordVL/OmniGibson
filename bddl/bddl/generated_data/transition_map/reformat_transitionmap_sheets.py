import csv 
import json 
import pandas as pd
import os

from tm_submap_params import TM_SUBMAPS_TO_PARAMS

OLD_SHEETS_DIR = "oldformat_csvs" # TODO get rid of this once sheets are reformatted
NEW_SHEETS_DIR = "newformat_sheets"

def make_heatcook():
    with open(os.path.join(OLD_SHEETS_DIR, "heat_cook.csv"), "r") as f:
        lines = [line for line in csv.reader(f)]

    with open(os.path.join(NEW_SHEETS_DIR, "heat_cook.csv"), "w") as f:
        fieldnames = ["rule_name"] + list(TM_SUBMAPS_TO_PARAMS["heat_cook"].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        record = {param: [] for param in fieldnames}
        section = ""
        for line in lines: 
            if all(elem == "" for elem in line):
                for field in record: 
                    record[field] = " and ".join(record[field])
                writer.writerow(record)
                record = {param: [] for param in fieldnames}
                continue
            # Section header
            if line[1] == "" and line[2] == "":
                if line[0][0] == "#":
                    record["rule_name"].append(line[0])
                    continue
                section = line[0]
                if section not in ["input", "container", "container-input relation", "heat source", "output", "container-output relation"]:
                    raise ValueError(f"Invalid section header {section}")
            # Section body 
            else: 
                if section == "input":
                    record["input_objects"].append(" ".join([elem for elem in line if elem != ""]))
                elif section == "container": 
                    record["container"].append(line[1])
                elif section == "container-input relation":
                    record["container_input_relation"].append(" ".join([elem for elem in line if elem != ""]))
                elif section == "heat source":
                    record["heat_source"].append(line[1])
                elif section == "container-output relation":
                    record["container_output_relation"].append(" ".join([elem for elem in line if elem != ""]))
                elif section == "output":
                    record["output_objects"].append(" ".join([elem for elem in line if elem != ""]))
                    

def make_electricmixer():
    with open(os.path.join(OLD_SHEETS_DIR, "electric_mixer.csv"), "r") as f:
        lines = [line for line in csv.reader(f)]

    with open(os.path.join(NEW_SHEETS_DIR, "electric_mixer.csv"), "w") as f:
        fieldnames = ["rule_name"] + list(TM_SUBMAPS_TO_PARAMS["electric_mixer"].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        record = {param: [] for param in fieldnames}
        section = ""
        for line in lines: 
            if all(elem == "" for elem in line):
                for field in record: 
                    record[field] = " and ".join(record[field])
                writer.writerow(record)
                record = {param: [] for param in fieldnames}
                continue
            # Section header
            if line[1] == "" and line[2] == "":
                if line[0][0] == "#":
                    record["rule_name"].append(line[0])
                    continue
                section = line[0]
                if section not in ["input", "output"]:
                    raise ValueError(f"Invalid section header {section}")
            # Section body 
            else: 
                if section == "input":
                    record["input_objects"].append(" ".join([elem for elem in line if elem != ""]))
                elif section == "output":
                    record["output_objects"].append(" ".join([elem for elem in line if elem != ""]))
                    

def make_mixingstick():
    with open(os.path.join(OLD_SHEETS_DIR, "mixing_stick.csv"), "r") as f:
        lines = [line for line in csv.reader(f)]

    with open(os.path.join(NEW_SHEETS_DIR, "mixing_stick.csv"), "w") as f:
        fieldnames = ["rule_name"] + list(TM_SUBMAPS_TO_PARAMS["mixing_stick"].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        record = {param: [] for param in fieldnames}
        section = ""
        for line in lines: 
            if all(elem == "" for elem in line):
                for field in record: 
                    record[field] = " and ".join(record[field])
                writer.writerow(record)
                record = {param: [] for param in fieldnames}
                continue
            # Section header
            if line[1] == "" and line[2] == "":
                if line[0][0] == "#":
                    record["rule_name"].append(line[0])
                    continue
                section = line[0]
                if section not in ["input", "output", "transition requirement"]:
                    raise ValueError(f"Invalid section header {section}")
            # Section body 
            else: 
                if section == "input":
                    record["input_objects"].append(" ".join([elem for elem in line if elem != ""]))
                elif section == "output":
                    record["output_objects"].append(" ".join([elem for elem in line if elem != ""]))

            
def make_singletoggleablemachine():
    with open(os.path.join(OLD_SHEETS_DIR, "single_toggleable_machine.csv"), "r") as f:
        lines = [line for line in csv.reader(f)]

    with open(os.path.join(NEW_SHEETS_DIR, "single_toggleable_machine.csv"), "w") as f:
        fieldnames = ["rule_name"] + list(TM_SUBMAPS_TO_PARAMS["single_toggleable_machine"].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        record = {param: [] for param in fieldnames}
        section = ""
        for line in lines: 
            if all(elem == "" for elem in line):
                for field in record: 
                    record[field] = " and ".join(record[field])
                writer.writerow(record)
                record = {param: [] for param in fieldnames}
                continue
            # Section header
            if line[1] == "" and line[2] == "":
                if line[0][0] == "#":
                    record["rule_name"].append(line[0])
                    continue
                section = line[0]
                if section not in ["input", "output", "transition machine"]:
                    raise ValueError(f"Invalid section header {section}")
            # Section body 
            else: 
                if section == "input":
                    record["input_objects"].append(" ".join([elem for elem in line if elem != ""]))
                elif section == "output":
                    record["output_objects"].append(" ".join([elem for elem in line if elem != ""]))
                elif section == "transition machine":
                    record["machine"].append(" ".join([elem for elem in line if elem != ""]))

            
if __name__ == "__main__":
    make_singletoggleablemachine()