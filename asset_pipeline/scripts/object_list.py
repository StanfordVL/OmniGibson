import json
import re

import pymxs
rt = pymxs.runtime

DEFAULT_PATH = "artifacts/object_list.json"
PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6}+)-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?$")

def main():
    object_names = [x.name for x in rt.objects if rt.classOf(x) == rt.Editable_Poly]
    matches = [rt.fullmatch(name) for name in object_names]

    success = all(x is not None for x in matches)
    needed = [x.group("category") + "-" + x.group("model_id") for x in matches]
    provided = [x.group("category") + "-" + x.group("model_id") for x in matches if not x.group("bad")]

    with open(DEFAULT_PATH, "w") as f:
        json.dump({"success": success, "needed_objects": needed, "provided_objects": provided}, f, indent=4)

if __name__ == "__main__":
    main()