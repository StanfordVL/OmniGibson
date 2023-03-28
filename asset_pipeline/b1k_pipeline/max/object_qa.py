import json
import os
import sys
sys.path.append(r"D:\ig_pipeline")

import pymxs
rt = pymxs.runtime

from b1k_pipeline.utils import parse_name, PIPELINE_ROOT

PASS_NAME = "ruohan-1"
RECORD_PATH = PIPELINE_ROOT / "qa-logs" / f"{PASS_NAME}.json"

def main():
    # Read completed groups record
    completed_groups = set()
    if os.path.exists(RECORD_PATH):
        with open(RECORD_PATH) as f:
            completed_groups = {tuple(obj) for obj in json.load(f)}

    # Find incomplete groups
    parsed_names = [parse_name(x.name) for x in rt.objects]
    available_groups = {x.group("category") + "-" + x.group("model_id") for x in parsed_names if x is not None and not x.group("bad") and int(x.group("instance_id")) == 0}
    remaining_groups = sorted(available_groups - completed_groups)
    if not remaining_groups:
        rt.messageBox("Scene complete. Move to next scene.")
        return

    # Find objects corresponding to the next remaining group's instance zero
    next_group = remaining_groups[0]
    next_group_objects = []
    for obj in rt.objects:
        n = parse_name(obj)
        if n is None:
            continue
        if n.group("bad") or int(n.group("instance_id")) != 0:
            continue
        if n.group("category") + "-" + n.group("model_id") != next_group:
            continue
        next_group_objects.append(obj)

    # Select that object and print
    rt.select(next_group_objects)
    rt.IsolateSelection.EnterIsolateSelectionMode()
    rt.execute("max tool zoomextents")
    print(f"{len(remaining_groups) + 1} / {len(available_groups)}: {next_group}")

    # Record that object as completed
    with open(RECORD_PATH, "w") as f:
        json.dump(completed_groups | {next_group}, f)