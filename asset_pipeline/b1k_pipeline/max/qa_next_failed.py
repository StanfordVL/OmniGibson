import sys

sys.path.append(r"D:\ig_pipeline")

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


def file_eligible(objdir):
    complaint_path = objdir / "complaints.json"
    
    if not complaint_path.exists():
        return False

    with open(complaint_path, "r") as f:
        x = json.load(f)
        if any(not y["processed"] for y in x if y["message"].startswith("Confirm object visual appearance.")):
            return True

    return False


def next_failed():
    eligible_max = []
    for target in b1k_pipeline.utils.get_targets("objects"):
        objdir = b1k_pipeline.utils.PIPELINE_ROOT / "cad" / target
        if objdir.exists() and file_eligible(objdir):
            eligible_max.append(objdir / "processed.max")

    eligible_max.sort()
    print(len(eligible_max), "files remaining.")
    print("\n".join(str(x) for x in eligible_max))
    if eligible_max:
        # Find current file in the list
        current_max_dir = pathlib.Path(rt.maxFilePath).resolve()
        next_idx = 0
        try:
            next_idx = (next(i for i, max in enumerate(eligible_max) if max.parent.resolve() == current_max_dir) + 1) % len(eligible_max)
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
