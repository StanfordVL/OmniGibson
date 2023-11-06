import sys

import numpy as np

sys.path.append(r"D:\ig_pipeline")

import json
import os
from collections import Counter, defaultdict

import pymxs

import b1k_pipeline.utils

rt = pymxs.runtime

OUTPUT_FILENAME = "room_object_list.json"
SUCCESS_FILENAME = "room_object_list.success"


def main():
    objects_by_room = defaultdict(Counter)
    nomatch = []

    for obj in rt.objects:
        if rt.classOf(obj) not in [rt.Editable_Poly, rt.Editable_Mesh, rt.VrayProxy]:
            continue

        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            nomatch.append(obj.name)
            continue

        model = match.group("category") + "-" + match.group("model_id")

        room_strs = obj.layer.name
        for room_str in room_strs.split(","):
            room_str = room_str.strip()
            if room_str == "0":
                continue
            link_name = match.group("link_name")
            if link_name == "base_link" or not link_name:
                objects_by_room[room_str][model] += 1

    # Separately process portals
    portals = [x for x in rt.objects if rt.classOf(x) == rt.Plane]
    incoming_portal = None
    outgoing_portals = {}
    for portal in portals:
        portal_match = b1k_pipeline.utils.parse_portal_name(portal.name)
        if not portal_match:
            nomatch.append(portal.name)
            continue

        # Get portal info
        position = list(portal.position)
        rotation = portal.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        scale = np.array(list(portal.scale))
        size = list(np.array([portal.width, portal.length]) * scale[:2])

        portal_info = [position, quat, size]

        # Process incoming portal
        if portal_match.group("partial_scene") is None:
            assert incoming_portal is None, "Duplicate incoming portal"
            incoming_portal = portal_info
        else:
            scene_name = portal_match.group("partial_scene")
            assert scene_name not in outgoing_portals, f"Duplicate outgoing portal for scene {scene_name}"
            outgoing_portals[scene_name] = portal_info

    success = len(nomatch) == 0

    output_dir = os.path.join(rt.maxFilePath, "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, OUTPUT_FILENAME)
    with open(filename, "w") as f:
        json.dump(
            {
                "success": success,
                "objects_by_room": objects_by_room,
                "incoming_portal": incoming_portal,
                "outgoing_portals": outgoing_portals,
                "error_invalid_name": sorted(nomatch),
            },
            f,
            indent=4,
        )

    if success:
        with open(os.path.join(output_dir, SUCCESS_FILENAME), "w") as f:
            pass


if __name__ == "__main__":
    main()
