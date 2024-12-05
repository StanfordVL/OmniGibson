from collections import defaultdict
import pathlib
import re
import traceback
import numpy as np
import json
from fs.zipfs import ZipFS

import pymxs
rt = pymxs.runtime

import sys

sys.path.append(r"D:\ig_pipeline")

from b1k_pipeline.utils import PipelineFS, get_targets, parse_name, load_mesh, PIPELINE_ROOT

FILLABLE_DIR = pathlib.Path(r"D:\fillable-10-21")
JUST_THIS_FILE = True
REMOVE_EXISTING = True

def import_fillable_volumes(model_id, object_links, fillable_assignment):
    print("Importing collision mesh for", model_id)

    # Remove any fillable volumes that this object already has
    if REMOVE_EXISTING:
        for cand_obj in rt.objects:
            match = parse_name(cand_obj.name)
            if not match:
                continue
            if match.group("model_id") != model_id:
                continue
            if match.group("meta_type") != "collision":
                continue
            rt.delete(cand_obj)

    # Find the directory corresponding to this object
    model_dir, = list(FILLABLE_DIR.glob(f"objects/*/{model_id}"))

    # Match the files
    if not collision_filenames:
        print("No collision meshes found for", obj.name)
        return

    selection_matches = [(selection_matching_pattern.fullmatch(x), x) for x in collision_filenames]
    indexed_matches = {int(match.group(1)): fn for match, fn in selection_matches if match}
    expected_keys = set(range(len(indexed_matches)))
    found_keys = set(indexed_matches.keys())
    assert expected_keys == found_keys, f"Missing collision meshes for {node_key}: {expected_keys - found_keys}"
    ordered_collision_filenames = [indexed_matches[i] for i in range(len(indexed_matches))]

    collision_meshes = []
    for collision_filename in ordered_collision_filenames:
        collision_mesh = load_mesh(collision_fs, collision_filename, force="mesh", skip_materials=True)
        if not collision_mesh.is_volume:
            collision_mesh = load_mesh(collision_fs, collision_filename, force="mesh", process=False, skip_materials=True)
        collision_meshes.append(collision_mesh)

    # Get a flattened list of vertices and faces
    all_vertices = []
    all_faces = []
    for split in collision_meshes:
        vertices = [rt.Point3(*v.tolist()) for v in split.vertices]
        # Offsetting here by the past vertex count
        faces = [[v + len(all_vertices) + 1 for v in f.tolist()] for f in split.faces]
        all_vertices.extend(vertices)
        all_faces.extend(faces)

    # Create a new node for the collision mesh
    collision_obj = rt.Editable_Mesh()
    rt.ConvertToPoly(collision_obj)
    collision_obj.name = f"{parsed_name.group('mesh_basename')}-Mcollision"
    collision_obj.position = obj.position
    collision_obj.rotation = obj.rotation
    
    # Add the vertices
    for v in all_vertices:
        rt.polyop.createVert(collision_obj, v)

    # Add the faces
    for f in all_faces:
        rt.polyop.createPolygon(collision_obj, f)

    # Optionally set its wire color
    collision_obj.wirecolor = rt.yellow

    # Update the mesh to reflect changes
    rt.update(collision_obj)

    # Parent the mesh
    collision_obj.parent = obj

    # Check that the new element count is the same as the split count
    elems = {tuple(rt.polyop.GetElementsUsingFace(collision_obj, i + 1)) for i in range(rt.polyop.GetNumFaces(collision_obj))}
    assert len(elems) == len(collision_meshes), f"{obj.name} has different number of faces in collision mesh than in splits"
    elems = np.array(list(elems))
    assert not np.any(np.sum(elems, axis=0) > 1), f"{obj.name} has same face appear in multiple elements"

    # Hide the mesh
    collision_obj.isHidden = True


def process_target(pipeline_fs, target, fillable_assignments):
    print("Processing", target)
    
    if not JUST_THIS_FILE:
        filename = str(pipeline_fs.target(target).getsyspath("processed.max"))
        assert rt.loadMaxFile(filename, useFileUnits=False, quiet=True)

    # Iterate through the objects in the file and build the link lists
    object_links = defaultdict(dict)
    for obj in rt.objects:
        match = parse_name(obj.name)
        if not match:
            continue

        if match.group("instance_id") != "0":
            continue

        if match.group("bad"):
            continue

        if match.group("meta_type"):
            continue

        if match.group("joint_side") == "upper":
            continue

        link_name = match.group("link_name") if match.group("link_name") else "base_link"
        object_links[match.group("model_id")][link_name] = obj

    # For each object, try to import the fillable volumes
    availables = set(fillable_assignments.keys()) & set(object_links.keys())
    for model_id in sorted(availables):
        import_fillable_volumes(model_id, object_links[model_id], fillable_assignments[model_id])

    if not JUST_THIS_FILE:
        rt.saveMaxFile(filename)


def main():
    # Load the fillable selection
    with open(FILLABLE_DIR / "fillable_assignments_2.json", "r") as f:
        fillable_assignments = json.load(f)

    # Filter it down to just the fillable objects
    fillable_assignments = {k: v for k, v in fillable_assignments.items() if v in {"dip", "ray", "combined", "generated"}}

    with PipelineFS() as pipeline_fs:
        current_target_dir = pathlib.Path(rt.maxFilePath)
        current_target_rel = current_target_dir.relative_to(PIPELINE_ROOT)
        cad, target_type, target_name = current_target_rel.parts
        assert cad == "cad", f"Current file is not in the cad directory: {current_target_rel}"
        assert target_type in ("scenes", "objects"), f"Current file is not in the scenes or objects directory: {current_target_rel}"
        current_target = target_type + "/" + target_name

        targets = get_targets("combined_unfiltered") if not JUST_THIS_FILE else [current_target]

        for target in targets:
            process_target(pipeline_fs, target, fillable_assignments)

if __name__ == "__main__":
    main()
