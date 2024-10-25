from collections import defaultdict
import json
import pathlib
import sys
import traceback

sys.path.append(r"D:\ig_pipeline")

import pymxs

rt = pymxs.runtime

from b1k_pipeline.utils import mat2arr, parse_name, PIPELINE_ROOT

import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation
import tqdm
import trimesh

MINIMUM_COLLISION_DEPTH_METERS = 0.01  # 1cm of collision


inventory_path = PIPELINE_ROOT / "artifacts" / "pipeline" / "object_inventory.json"
with open(inventory_path, "r") as f:
    providers = {k.split("-")[-1]: v for k, v in json.load(f)["providers"].items()}


def quat2arr(q):
    return np.array([q.x, q.y, q.z, q.w])


def import_bad_model_originals(model_id):
    # Get the provider file
    provider = providers[model_id]
    filename = str(PIPELINE_ROOT / "cad" / provider / "processed.max")

    # Get the right object names from the file
    f_objects = rt.getMAXFileObjectNames(filename, quiet=True)
    visual_objects = {}
    for obj in f_objects:
        m = parse_name(obj)
        if not m:
            continue
        if (
            m.group("model_id") != model_id
            or m.group("instance_id") != "0"
            or m.group("joint_side") == "upper"
        ):
            continue

        # We have no way of checking if the object is an editable poly so we just apply some heuristics
        if m.group("meta_type") and m.group("meta_type") != "collision":
            continue
        if m.group("light_id"):
            continue

        # If we get here this is a lower link of an instance zero
        assert not m.group("bad"), "Bad objects should not be in the inventory"

        link_name = m.group("link_name") if m.group("link_name") else "base_link"
        if "Mcollision" in obj:
            continue

        visual_objects[link_name] = obj

    objects_to_import = set(visual_objects.values())

    success, imported_meshes = rt.mergeMaxFile(
        filename,
        objects_to_import,
        rt.Name("select"),
        rt.Name("autoRenameDups"),
        rt.Name("useSceneMtlDups"),
        rt.Name("neverReparent"),
        rt.Name("noRedraw"),
        quiet=True,
        mergedNodes=pymxs.byref(None),
    )
    assert success, f"Failed to import {model_id}."
    imported_objs_by_name = {obj.name: obj for obj in imported_meshes}
    assert set(objects_to_import) == set(
        imported_objs_by_name.keys()
    ), "Not all objects were imported. Missing: " + str(
        set(objects_to_import) - set(imported_objs_by_name.keys())
    )

    return {k: imported_objs_by_name[v] for k, v in visual_objects.items()}


def replace_object_instances(obj):
    # Parse the name and assert it's the base link of a bad object and instance 0
    parsed_name = parse_name(obj.name)
    assert parsed_name, f"Object {obj.name} has no parsed name"
    assert parsed_name.group("bad"), f"Object {obj.name} is not a bad object"
    assert parsed_name.group("instance_id") == "0", f"Object {obj.name} is not instance 0"
    assert not (parsed_name.group("link_name") and parsed_name.group("link_name") != "base_link"), f"Object {obj.name} is not the base link"

    # Get the model ID
    model_id = parsed_name.group("model_id")

    # First, find all the instances of this object and validate that their pivots are the same
    all_instances = [x for x in rt.objects if x.baseObject == obj.baseObject]
    all_same_model_id = [x for x in rt.objects if parse_name(x.name) and parse_name(x.name).group("model_id") == model_id]
    instances_parsed_names = [parse_name(x.name) for x in all_same_model_id]
    same_model_id_parsed_names = [parse_name(x.name) for x in all_same_model_id]
    assert all(instances_parsed_names), "All instances should have a parsed name"
    assert all(same_model_id_parsed_names), "All same model ID objects should have a parsed name"

    # Check that they have the same model ID and same category
    unique_model_ids_for_instance = {x.group("model_id") for x in instances_parsed_names}
    assert len(unique_model_ids_for_instance) == 1, f"All instances of {obj.name} do not share category/model ID."

    # Check that once grouped by instance ID, they all have the same set of links
    links_by_instance_id = defaultdict(dict)
    for inst, inst_parsed_name in zip(all_same_model_id, same_model_id_parsed_names):
        link_name = inst_parsed_name.group("link_name")
        if not link_name:
            link_name = "base_link"
        links_by_instance_id[inst_parsed_name.group("instance_id")][link_name] = inst
    all_links = set(frozenset(x.keys()) for x in links_by_instance_id.values())
    assert len(all_links) == 1, f"All instances of {obj.name} do not share the same set of links: {links_by_instance_id}"

    # Check that they are all marked as BAD
    assert all(x.group("bad") for x in instances_parsed_names), "All instances should be marked as bad."
    assert all(x.group("bad") for x in same_model_id_parsed_names), "All instances of the same model ID should be marked as bad."

    # Check that they do NOT have any children
    assert not any(x.children for x in all_instances), "Instances should not have children."
    assert not any(x.children for x in all_same_model_id), "Instances of the same model ID should not have children."

    # Check that they all have the same object offset rotation and pos/scale and shear.
    desired_offset_pos = np.array(obj.objectOffsetPos) / np.array(obj.objectOffsetScale)
    desired_offset_rot_inv = Rotation.from_quat(quat2arr(obj.objectOffsetRot)).inv()
    for inst in all_instances:
        this_offset_pos = np.array(inst.objectOffsetPos) / np.array(inst.objectOffsetScale)
        pos_diff = this_offset_pos - desired_offset_pos
        assert np.allclose(pos_diff, 0, atol=5e-2), f"{inst.name} has different pivot offset position (by {pos_diff}). Match pivots on each instance.",

        this_offset_rot = Rotation.from_quat(quat2arr(inst.objectOffsetRot))
        rot_diff = (this_offset_rot * desired_offset_rot_inv).magnitude()
        assert np.allclose(rot_diff, 0, atol=1e-3), f"{inst.name} has different pivot offset rotation (by {rot_diff}). Match pivots on each instance."

    # Parent all links under their bases for efficient computation of the bounding box
    all_parented = set()
    for instance, links in links_by_instance_id.items():
        base_link = links.pop("base_link")
        assert base_link in all_instances, f"Base link {base_link.name} not found in instances."
        for link in links.values():
            link.parent = base_link
        all_parented.add(base_link)
    assert all_parented == set(all_instances), "Not all instances were parented."

    # Record the transforms of all of the instances as well as their local and world BBs
    obj_transform = obj.transform
    rotation_only_transform = rt.Matrix3(
    obj_local_bb = rt.NodeGetBoundingBox(obj, obj_transform)
    instance_transforms = [inst.transform for inst in all_instances]
    instance_local_bbs = [rt.NodeGetBoundingBox(inst, inst.transform) for inst in all_instances]
    instance_world_bbs = [rt.NodeGetBoundingBox(inst, rt.Matrix3(1)) for inst in all_instances]
    
    # Delete all the objects
    for inst in all_instances:
        rt.delete(inst)
    for same_model_id in all_same_model_id:
        rt.delete(same_model_id)

    # Import the original mesh for just the zero instance
    imported = import_bad_model_originals(model_id)

    # Parent all the imported meshes under the base link
    base_link = imported.pop("base_link")
    for link in imported.values():
        link.parent = base_link

    # Get the bounding box of the imported mesh. Here we use the local-oriented-world-bb,
    # which means we assume that the imported mesh and the original one have the same
    # orientation relative to their pivots.
    imported_local_bb = rt.NodeGetBoundingBox(base_link, base_link.transform)
    imported_world_bb = rt.NodeGetBoundingBox(base_link, rt.Matrix3(1))


def main():
    replace_object_instances(rt.selection[0])


if __name__ == "__main__":
    main()
