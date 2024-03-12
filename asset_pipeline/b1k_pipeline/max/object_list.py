import sys

import numpy as np

sys.path.append(r"D:\ig_pipeline")

import json
import pathlib
from collections import Counter, defaultdict

import pymxs

import b1k_pipeline.utils

rt = pymxs.runtime

OUTPUT_FILENAME = "object_list.json"


def main():
    object_names = [x.name for x in rt.objects if rt.classOf(x) == rt.Editable_Poly]
    matches = [b1k_pipeline.utils.parse_name(name) for name in object_names]
    nomatch = [name for name, match in zip(object_names, matches) if match is None]

    success = len(nomatch) == 0
    needed = sorted(
        {
            x.group("category") + "-" + x.group("model_id")
            for x in matches
            if x is not None and not x.group("meta_type")
        }
    )
    provided = sorted(
        {
            x.group("category") + "-" + x.group("model_id")
            for x in matches
            if x is not None and not x.group("bad") and not x.group("meta_type")
        }
    )
    counts = Counter(
        [
            x.group("category") + "-" + x.group("model_id")
            for x in matches
            if x is not None and not x.group("meta_type")
        ]
    )
    max_tree = [(x.name, str(rt.classOf(x)), x.parent.name if x.parent else None) for x in rt.objects]

    meshes = sorted(
        name for match, name in zip(matches, object_names) if match is not None and not match.group("meta_type")
    )

    meta_links = defaultdict(set)
    for obj, _, parent in max_tree:
        if not parent:
            continue
        parent_match = b1k_pipeline.utils.parse_name(parent)
        obj_match = b1k_pipeline.utils.parse_name(obj)
        if not parent_match or not obj_match:
            continue
        if not obj_match.group("meta_type") or parent_match.group("bad"):
            continue
        parent_id = parent_match.group("category") + "-" + parent_match.group("model_id")
        meta_type = obj_match.group("meta_type")
        meta_links[parent_id].add(meta_type)

    meta_links = {k: sorted(v) for k, v in sorted(meta_links.items())}

    # Compute attachment points
    bad_attachments = set()
    attachment_points = defaultdict(lambda: defaultdict(lambda: {"M": 0, "F": 0}))
    for obj, _, parent in max_tree:
        if not parent:
            continue
        parent_match = b1k_pipeline.utils.parse_name(parent)
        obj_match = b1k_pipeline.utils.parse_name(obj)
        if not parent_match or not obj_match:
            continue
        if obj_match.group("meta_type") != "attachment":
            continue
        parent_id = parent_match.group("category") + "-" + parent_match.group("model_id")
        attachment_type = obj_match.group("meta_id")
        if len(attachment_type) == 0 or attachment_type[-1] not in "MF":
            bad_attachments.add(obj)
            continue
        attachment_gender = attachment_type[-1]
        attachment_type = attachment_type[:-1]
        if len(attachment_type) == 0:
            bad_attachments.add(obj)
            continue
        attachment_points[parent_id][attachment_type][attachment_gender] += 1

    # Add attachment points for connected parts
    for obj, _, parent in max_tree:
        if not parent:
            continue
        parent_match = b1k_pipeline.utils.parse_name(parent)
        obj_match = b1k_pipeline.utils.parse_name(obj)
        if not parent_match or not obj_match:
            continue
        if not obj_match.group("tag") or "Tconnectedpart" not in obj_match.group("tag"):
            continue
        parent_id = parent_match.group("category") + "-" + parent_match.group("model_id")
        child_id = obj_match.group("category") + "-" + obj_match.group("model_id")
        attachment_type = f"{obj_match.group('model_id')}parent"
        attachment_points[parent_id][attachment_type]["F"] += 1
        attachment_points[child_id][attachment_type]["M"] += 1

    meta_links = {k: sorted(v) for k, v in sorted(meta_links.items())}

    output_dir = pathlib.Path(rt.maxFilePath) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / OUTPUT_FILENAME
    results = {
        "success": success,
        "needed_objects": needed,
        "provided_objects": provided,
        "meshes": meshes,
        "meta_links": meta_links,
        "attachment_points": attachment_points,
        "max_tree": max_tree, 
        "object_counts": counts,
        "error_invalid_name": sorted(nomatch),
        "error_bad_attachments": sorted(bad_attachments),
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print("success:", success)
    print("error_invalid_name:", sorted(nomatch))


if __name__ == "__main__":
    main()
