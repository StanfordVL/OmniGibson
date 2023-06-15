import sys

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

    output_dir = pathlib.Path(rt.maxFilePath) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / OUTPUT_FILENAME
    results = {
        "success": success,
        "needed_objects": needed,
        "provided_objects": provided,
        "meshes": meshes,
        "meta_links": meta_links,
        "max_tree": max_tree, 
        "object_counts": counts,
        "error_invalid_name": sorted(nomatch),
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print("success:", success)
    print("error_invalid_name:", sorted(nomatch))


if __name__ == "__main__":
    main()
