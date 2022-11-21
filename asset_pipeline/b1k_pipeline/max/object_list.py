import sys
sys.path.append(r"D:\ig_pipeline")

from collections import Counter
import json
import pathlib
import b1k_pipeline.utils

import pymxs
rt = pymxs.runtime

OUTPUT_FILENAME = "object_list.json"

def main():
    object_names = [x.name for x in rt.objects if rt.classOf(x) == rt.Editable_Poly]
    matches = [b1k_pipeline.utils.parse_name(name) for name in object_names]

    nomatch = [name for name, match in zip(object_names, matches) if match is None]
    success = len(nomatch) == 0
    needed = sorted({x.group("category") + "-" + x.group("model_id") for x in matches if x is not None})
    provided = sorted({x.group("category") + "-" + x.group("model_id") for x in matches if x is not None and not x.group("bad")})
    counts = Counter([x.group("category") + "-" + x.group("model_id") for x in matches if x is not None])

    meshes = sorted(name for match, name in zip(matches, object_names) if match is not None)

    output_dir = pathlib.Path(rt.maxFilePath) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / OUTPUT_FILENAME
    with open(filename, "w") as f:
        json.dump({"success": success, "needed_objects": needed, "provided_objects": provided, "meshes": meshes, "object_counts": counts, "error_invalid_name": sorted(nomatch)}, f, indent=4)

if __name__ == "__main__":
    main()