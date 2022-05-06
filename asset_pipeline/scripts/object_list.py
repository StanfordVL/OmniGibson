import json
import os
import re

import pymxs
rt = pymxs.runtime

PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?$")
OUTPUT_FILENAME = "object_list.json"
SUCCESS_FILENAME = "object_list.success"

def main():
    object_names = [x.name for x in rt.objects if rt.classOf(x) == rt.Editable_Poly]
    matches = [PATTERN.fullmatch(name) for name in object_names]

    success = all(x is not None for x in matches)
    needed = sorted({x.group("category") + "-" + x.group("model_id") for x in matches})
    provided = sorted({x.group("category") + "-" + x.group("model_id") for x in matches if not x.group("bad")})

    output_dir = rt.maxops.mxsCmdLineArgs[rt.name('dir')]
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, OUTPUT_FILENAME)
    with open(filename, "w") as f:
        json.dump({"success": success, "needed_objects": needed, "provided_objects": provided}, f, indent=4)

    if success:
        with open(os.path.join(output_dir, SUCCESS_FILENAME), "w") as f:
            pass

if __name__ == "__main__":
    main()