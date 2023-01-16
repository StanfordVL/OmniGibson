import glob
import json
import pathlib
import re
from collections import defaultdict

import numpy as np
import pymxs
from scipy.spatial.transform import Rotation as R

rt = pymxs.runtime


def file_eligible(objdir):
    objlist = objdir / "artifacts" / "object_list.json"
    sanitycheck = objdir / "artifacts" / "sanitycheck.json"

    if not objlist.exists() or not sanitycheck.exists():
        print(
            f"{str(objdir)} is missing object list or sanitycheck. Unsure if it failed."
        )
        return False

    with open(objlist, "r") as f:
        x = json.load(f)
        if not x["success"]:
            return True

    with open(sanitycheck, "r") as f:
        x = json.load(f)
        if not x["success"]:
            return True

    return False


def next_failed():
    eligible_max = []
    for objdir in glob.glob(r"D:\ig_pipeline\cad\objects\*"):
        objdir = pathlib.Path(objdir)
        if objdir.exists() and file_eligible(objdir):
            eligible_max.append(objdir / "processed.max")

    eligible_max.sort()
    print(len(eligible_max), "files remaining.")
    print("\n".join(str(x) for x in eligible_max))
    if eligible_max:
        scene_file = eligible_max[0]
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
