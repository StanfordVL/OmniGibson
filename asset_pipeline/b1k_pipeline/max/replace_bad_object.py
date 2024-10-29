from collections import defaultdict
import json
import sys

import tqdm

sys.path.append(r"D:\ig_pipeline")

import pymxs

rt = pymxs.runtime

from b1k_pipeline.utils import mat2arr, parse_name, PIPELINE_ROOT

import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation

MINIMUM_COLLISION_DEPTH_METERS = 0.01  # 1cm of collision


inventory_path = PIPELINE_ROOT / "artifacts" / "pipeline" / "object_inventory.json"
with open(inventory_path, "r") as f:
    providers = {k.split("-")[-1]: v for k, v in json.load(f)["providers"].items()}


def quat2arr(q):
    return np.array([q.x, q.y, q.z, q.w])


def get_vert_sets_including_children(node):
    vert_sets = [
        np.array(
            [rt.polyop.getVert(node, i + 1) for i in range(rt.polyop.GetNumVerts(node))]
        )
    ]
    for child in node.children:
        vert_sets.extend(get_vert_sets_including_children(child))
    return vert_sets


def node_bounding_box_incl_children(node, transform=None):
    verts_world = np.concatenate(get_vert_sets_including_children(node), axis=0)
    if transform is None or transform == rt.Matrix3(1):
        verts = verts_world
    else:
        transform = np.hstack([mat2arr(transform), [[0], [0], [0], [1]]]).T
        inv_transform = np.linalg.inv(transform)
        verts = np.concatenate(
            [verts_world, np.ones((verts_world.shape[0], 1))], axis=1
        )
        verts = (inv_transform @ verts.T).T[:, :3]
    bb_min = np.min(verts, axis=0)
    bb_max = np.max(verts, axis=0)
    return bb_min, bb_max


def rotation_only_transform(transform):
    rot_only = rt.Matrix3(1)
    rot_only.rotation = transform.rotation
    return rot_only


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
        joint_type = m.group("joint_type")
        if "Mcollision" in obj:
            continue

        visual_objects[(link_name, m.group("parent_link_name"), joint_type)] = obj

    objects_to_import = sorted(set(visual_objects.values()))

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
    assert (
        parsed_name.group("instance_id") == "0"
    ), f"Object {obj.name} is not instance 0"
    assert not (
        parsed_name.group("link_name") and parsed_name.group("link_name") != "base_link"
    ), f"Object {obj.name} is not the base link"

    # Get the model ID
    loose = parsed_name.group("loose")
    if not loose:
        loose = ""
    category = parsed_name.group("category")
    model_id = parsed_name.group("model_id")
    tags = parsed_name.group("tag")
    if (
        not tags
    ):  # This is for retaining any part tag on the base link. Other tags dont care.
        tags = ""

    # First, find all the instances of this object and validate that their pivots are the same
    all_same_model_id = [
        x
        for x in rt.objects
        if parse_name(x.name) and parse_name(x.name).group("model_id") == model_id
    ]
    all_base_links = [
        x
        for x in all_same_model_id
        if parse_name(x.name).group("link_name") == parsed_name.group("link_name")
    ]
    same_model_id_parsed_names = [parse_name(x.name) for x in all_same_model_id]
    assert all(
        same_model_id_parsed_names
    ), "All same model ID objects should have a parsed name"

    # Check that once grouped by instance ID, they all have the same set of links
    links_by_instance_id = defaultdict(dict)
    for inst, inst_parsed_name in zip(all_same_model_id, same_model_id_parsed_names):
        link_name = inst_parsed_name.group("link_name")
        if not link_name:
            link_name = "base_link"
        links_by_instance_id[inst_parsed_name.group("instance_id")][link_name] = inst
    all_links = set(frozenset(x.keys()) for x in links_by_instance_id.values())
    assert (
        len(all_links) == 1
    ), f"All instances of {obj.name} do not share the same set of links: {links_by_instance_id}"

    # Check that they are all marked as BAD
    assert all(
        x.group("bad") for x in same_model_id_parsed_names
    ), "All instances of the same model ID should be marked as bad."

    # Check that they do NOT have any children
    assert not any(
        len(x.children) > 0 for x in all_same_model_id
    ), "Instances of the same model ID should not have children: {all_same_model_id}"

    # Assert they are all unit scale
    # assert all(
    #     np.allclose(np.array(x.scale), np.ones(3)) for x in all_same_model_id
    # ), "Instances of the same model ID should all have unit scale."

    # # Check that they all have the same object offset rotation and pos/scale and shear.
    # desired_offset_pos = np.array(obj.objectOffsetPos) / np.array(obj.objectOffsetScale)
    # desired_offset_rot_inv = Rotation.from_quat(quat2arr(obj.objectOffsetRot)).inv()
    # for inst in all_base_links:
    #     this_offset_pos = np.array(inst.objectOffsetPos) / np.array(
    #         inst.objectOffsetScale
    #     )
    #     pos_diff = this_offset_pos - desired_offset_pos
    #     assert np.allclose(
    #         pos_diff, 0, atol=5e-2
    #     ), f"{inst.name} has different pivot offset position (by {pos_diff}). Match pivots on each instance."

    #     this_offset_rot = Rotation.from_quat(quat2arr(inst.objectOffsetRot))
    #     rot_diff = (this_offset_rot * desired_offset_rot_inv).magnitude()
    #     assert np.allclose(
    #         rot_diff, 0, atol=1e-3
    #     ), f"{inst.name} has different pivot offset rotation (by {rot_diff}). Match pivots on each instance."

    # Parent all links under their bases for efficient computation of the bounding box
    all_parented = set()
    for instance, links in links_by_instance_id.items():
        imported_base_link = links.pop("base_link")
        for link in links.values():
            link.parent = imported_base_link
        all_parented.add(imported_base_link)
    assert all_parented == set(all_base_links), "Not all instances were parented."

    # Record the transforms of all of the instances as well as their local and world BBs
    instance_parents = [inst.parent for inst in all_base_links]
    instance_transforms = [inst.transform for inst in all_base_links]
    instance_lowbbs = [
        node_bounding_box_incl_children(inst, rotation_only_transform(inst.transform))
        for inst in all_base_links
    ]
    instance_world_bbs = [
        node_bounding_box_incl_children(inst) for inst in all_base_links
    ]

    # Delete all the objects
    for same_model_id in all_same_model_id:
        rt.delete(same_model_id)

    # Import the original mesh for just the zero instance
    imported = import_bad_model_originals(model_id)
    print("Imported originals for", model_id)

    # Parent all the imported meshes under the base link
    imported_base_link = imported[("base_link", None, None)]
    for (link_name, parent_link, joint_type), link in imported.items():
        if link_name == "base_link":
            continue
        link.parent = imported_base_link

    # Set the rotation of the imported base link to be the identity rotation
    imported_base_link.transform = rt.Matrix3(1)

    # Get the bounding box of the imported mesh. Here we use the local-oriented-world-bb,
    # which means we assume that the imported mesh and the original one have the same
    # orientation relative to their pivots. Here we can use the world bb because the
    # imported mesh right now has the identity transform.
    imported_lowbb = node_bounding_box_incl_children(imported_base_link)
    imported_lowbb_size = imported_lowbb[1] - imported_lowbb[0]

    # Flatten the copyables into a list
    copyables = [("base_link", None, None, imported_base_link)] + [
        (*k, v) for k, v in imported.items() if k[0] != "base_link"
    ]
    copyable_meshes = [x[-1] for x in copyables]

    # Now for each of the original bozos make a copy of the whole thing, scale it up, rotate it, and shift it into place
    for instance_id, (
        instance_parent,
        instance_transform,
        instance_lowbb,
        instance_world_bb,
    ) in enumerate(
        zip(instance_parents, instance_transforms, instance_lowbbs, instance_world_bbs)
    ):
        # First copy the object.
        success, child_copy = rt.maxOps.cloneNodes(
            copyable_meshes,
            cloneType=rt.name("instance"),
            newNodes=pymxs.byref(None),
        )
        assert success, f"Could not clone {imported_base_link.name}"
        print("Cloned", model_id, instance_id)

        # Find the base link in the child copies. It's the one whose parent is not one of the child copies
        (base_copy,) = [x for x in child_copy if x.parent not in child_copy]
        assert (
            base_copy == child_copy[0]
        ), "The base link should be the first element in the list of child"

        # Get the rotation-only transform of the instance and apply it to this object too
        base_copy.transform = base_copy.transform * rotation_only_transform(
            instance_transform
        )

        # Scale the imported mesh to match the instance
        instance_lowbb_size = instance_lowbb[1] - instance_lowbb[0]
        instance_lowbb_center = (instance_lowbb[0] + instance_lowbb[1]) / 2
        print(
            "Target instance has lowbb size",
            instance_lowbb_size,
            "and center",
            instance_lowbb_center,
        )
        relative_scale_from_now = instance_lowbb_size / 1000
        scale_transform = rt.Matrix3(1)
        scale_transform.scale = rt.Point3(*relative_scale_from_now.tolist())
        print("Applying relative scale", relative_scale_from_now)
        print("Previous scale was ", base_copy.scale)
        base_copy.transform = (
            scale_transform * base_copy.transform
        )  # this means scale first, then the real transform
        print("New scale is", base_copy.scale)

        # Finally position it to match the center
        base_copy_worldbb = node_bounding_box_incl_children(base_copy)
        base_copy_worldbb_center = (base_copy_worldbb[0] + base_copy_worldbb[1]) / 2
        instance_worldbb_center = (instance_world_bb[0] + instance_world_bb[1]) / 2
        move_center_by = rt.Point3(
            *(instance_worldbb_center - base_copy_worldbb_center).tolist()
        )
        base_copy.position = base_copy.position + move_center_by
        print(
            "Moving in world by ",
            move_center_by,
        )

        base_copy_lowbb = node_bounding_box_incl_children(
            base_copy, rotation_only_transform(base_copy.transform)
        )
        print(
            "Base copy new lowbb center", (base_copy_lowbb[0] + base_copy_lowbb[1]) / 2
        )

        # Assert the new world bb is the same as the original world bb
        base_copy_world_bb = node_bounding_box_incl_children(base_copy)
        orig_min, orig_max = np.array(instance_world_bb[0]), np.array(
            instance_world_bb[1]
        )
        orig_center = (orig_min + orig_max) / 2
        orig_size = orig_max - orig_min
        new_min, new_max = np.array(base_copy_world_bb[0]), np.array(
            base_copy_world_bb[1]
        )
        new_center = (new_min + new_max) / 2
        new_size = new_max - new_min
        print("Bounding box info")
        print("  Original min", orig_min, "vs New min", new_min)
        print("  Original max", orig_max, "vs New max", new_max)
        print("  Original size", orig_size, "vs New size", new_size)
        print("  Original center", orig_center, "vs New center", new_center)
        assert np.allclose(
            orig_min, new_min, atol=10
        ), "New world min is not the same as the original world min"
        assert np.allclose(
            orig_max, new_max, atol=10
        ), "New world max is not the same as the original world max"

        # Name each of the children correctly, and parent them to the original owner's parents
        for (link_name, parent_link, joint_type, _), link in zip(copyables, child_copy):
            link.parent = instance_parent
            if link_name == "base_link":
                link.name = (
                    f"B-{loose}{category}-{model_id}-{instance_id}-{link_name}{tags}"
                )
            else:
                link.name = f"B-{loose}{category}-{model_id}-{instance_id}-{link_name}-{parent_link}-{joint_type}-lower"
        print("Replaced", model_id, instance_id)

    # Finally, after all is done, remove all of the copyables
    rt.delete(copyable_meshes)


def replace_all_bad_legacy_objects_in_open_file():
    bad_objects_to_remove = {}
    for obj in rt.objects:
        # Check that it's an editable poly
        if rt.classOf(obj) != rt.Editable_Poly:
            continue

        # Check that the parsed name contains the B- tag
        parsed_name = parse_name(obj.name)
        if not parsed_name or not parsed_name.group("bad"):
            continue

        # Check that the link is the base link
        if (
            parsed_name.group("link_name")
            and parsed_name.group("link_name") != "base_link"
        ):
            continue

        # Check that the instance ID is 0
        if parsed_name.group("instance_id") != "0":
            continue

        # Check that its not a meta link
        if parsed_name.group("meta_type"):
            continue

        # Check that the provider is legacy_
        model_id = parsed_name.group("model_id")
        provider = providers[model_id]
        if "legacy_" not in provider:
            continue

        # Check that we havent picked an object for this model ID yet.
        assert (
            model_id not in bad_objects_to_remove
        ), f"Found multiple base objects for {model_id}. First: {bad_objects_to_remove[model_id].name}, Second: {obj.name}"

        # Add to the list of bad objects to remove
        bad_objects_to_remove[model_id] = obj

    print(
        "Found",
        len(bad_objects_to_remove),
        "bad objects to remove:",
        ", ".join(bad_objects_to_remove.keys()),
    )

    # Go through all the bad objects to remove
    for model_id, example_obj in tqdm.tqdm(bad_objects_to_remove.items()):
        replace_object_instances(example_obj)

    print("\nDon't forget to reset scale!")


def main():
    replace_object_instances(rt.selection[0])


if __name__ == "__main__":
    main()
