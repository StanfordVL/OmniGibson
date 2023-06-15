import bisect
import sys

sys.path.append(r"D:\ig_pipeline")
import csv
import glob
import json
import pathlib
import re
from collections import defaultdict

import numpy as np
import pymxs
from scipy.spatial.transform import Rotation as R

import b1k_pipeline.utils

rt = pymxs.runtime

ignore_messages = ["Confirm reasonable bounding box size:", "Confirm object synset assignment."]

def file_eligible(objdir):
    complaint_path = objdir / "complaints.json"
    
    if not complaint_path.exists():
        return False

    with open(complaint_path, "r") as f:
        x = json.load(f)
        if any(not y["processed"] for y in x if not any(y["message"].startswith(ign) for ign in ignore_messages)):
            return True

    return False


def next_failed():
    with open(rf"C:\Users\Vince\Downloads\task_required_objects.csv", "r") as f:
        reader = csv.reader(f)
        task_required_objects = {row[0].split("-")[-1] for row in reader}

    with open(b1k_pipeline.utils.PIPELINE_ROOT / "artifacts/pipeline/object_inventory_future.json", "r") as f:
        providers = json.load(f)["providers"]
    provided = defaultdict(set)
    for object, provider in providers.items():
        provided[provider].add(object.split("-")[-1])

    eligible_max = []
    for target in b1k_pipeline.utils.get_targets("objects"):
        # Get the objects from this target
        target_provided = provided[target]

        if target_provided & task_required_objects:
            objdir = b1k_pipeline.utils.PIPELINE_ROOT / "cad" / target
            if objdir.exists() and file_eligible(objdir):
                eligible_max.append(objdir / "processed.max")

    eligible_max.sort()
    eligible_max = eligible_max
    print(len(eligible_max), "files remaining.")
    print("\n".join(str(x) for x in eligible_max))
    if eligible_max:
        # Find current file in the list
        current_max = pathlib.Path(rt.maxFilePath).resolve() / "processed.max"
        next_idx = 0
        try:
            next_idx = bisect.bisect(eligible_max, current_max) % len(eligible_max)
        except:
            pass

        scene_file = eligible_max[next_idx]
        assert (
            not scene_file.is_symlink()
        ), f"File {scene_file} should not be a symlink."
        assert rt.loadMaxFile(
            str(scene_file), useFileUnits=False, quiet=True
        ), f"Could not load {scene_file}"


def next_failed_button():
    try:
        next_failed()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return


if __name__ == "__main__":
    next_failed_button()
