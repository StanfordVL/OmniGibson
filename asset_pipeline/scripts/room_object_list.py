import json
import os
import re
from collections import defaultdict, Counter

import pymxs
rt = pymxs.runtime

PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?$")
OUTPUT_FILENAME = "room_object_list.json"
SUCCESS_FILENAME = "room_object_list.success"

def main():
    objects_by_room = defaultdict(Counter)
    nomatch = []
    for obj in rt.objects:
        if rt.classOf(x) != rt.Editable_Poly:
            continue

        match = PATTERN.fullmatch(obj.name)
        if not match:
            nomatch.append(obj.name)
            continue

        room_strs = obj.layer.name
        for room_str in room_strs.split(","):
            link_name = match.group()
            if link_name == "base_link" or not link_name:
                objects_by_room[room_str.strip()][match.group("category")] += 1

    success = len(nomatch) == 0

    output_dir = rt.maxops.mxsCmdLineArgs[rt.name('dir')]
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, OUTPUT_FILENAME)
    with open(filename, "w") as f:
        json.dump({"success": success, "objects_by_room": objects_by_room, "error_invalid_name": sorted(nomatch)}, f, indent=4)

    if success:
        with open(os.path.join(output_dir, SUCCESS_FILENAME), "w") as f:
            pass

if __name__ == "__main__":
    main()