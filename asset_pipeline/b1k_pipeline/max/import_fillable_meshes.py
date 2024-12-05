from collections import defaultdict
import pathlib
import tempfile
import numpy as np
import json
import pxr
from fs.osfs import OSFS
from cryptography.fernet import Fernet

import pymxs
rt = pymxs.runtime

import sys

sys.path.append(r"D:\ig_pipeline")

from b1k_pipeline.utils import PipelineFS, get_targets, parse_name, load_mesh, PIPELINE_ROOT
from b1k_pipeline.max.replace_bad_object import node_bounding_box_incl_children, rotation_only_transform

FILLABLE_DIR = pathlib.Path(r"D:\fillable-10-21")
KEY_PATH = pathlib.Path(r"C:\Users\cgokmen\research\OmniGibson\omnigibson\data\omnigibson.key")
JUST_THIS_FILE = True
REMOVE_EXISTING = True


def decrypt_file(encrypted_filename, decrypted_filename):
    with open(KEY_PATH, "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def get_bounding_box_from_usd(obj_dir):
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        encrypted_filename = obj_dir / "usd" / f"{obj_dir.name}.encrypted.usd"
        decrypted_filename = tempdir / "obj.usd"
        decrypt_file(encrypted_filename, decrypted_filename)
        stage = pxr.Usd.Stage.Open(str(decrypted_filename))
        prim = stage.GetDefaultPrim()

        base_link_size = np.array(prim.GetAttribute("ig:nativeBB").Get()) * 1000
        base_link_offset = np.array(prim.GetAttribute("ig:offsetBaseLink").Get()) * 1000

        return base_link_offset, base_link_size


def import_fillable_volumes(model_id, object_links):
    print("Importing fillable mesh for", model_id)

    # Remove any fillable volumes that this object already has
    if REMOVE_EXISTING:
        for cand_obj in rt.objects:
            match = parse_name(cand_obj.name)
            if not match:
                continue
            if match.group("model_id") != model_id:
                continue
            if match.group("meta_type") not in ("fillable", "openfillable"):
                continue
            rt.delete(cand_obj)

    # Find the directory corresponding to this object
    model_dir, = list(FILLABLE_DIR.glob(f"objects/*/{model_id}"))

    # Find the fillable mesh files
    fillable_files = list(model_dir.glob("fillable---*---*---*.obj"))
    assert len(fillable_files) >= 1, f"Expected at least one fillable mesh for {model_id}"
    by_link_and_kind = defaultdict(lambda: defaultdict(list))
    for fillable_file in fillable_files:
        link, idx, kind = fillable_file.stem.split("---")[1:]
        by_link_and_kind[link][kind].append(fillable_file)

    # From the USD read the bounding box info
    usd_bbox_offset, usd_bbox_size = get_bounding_box_from_usd(model_dir)

    # Get information about the base link of the object
    base_link = object_links["base_link"]
    base_rot = base_link.rotation
    bbox_min, bbox_max = node_bounding_box_incl_children(base_link, rotation_only_transform(base_link.transform), only_canonical=True)
    bbox_ctr = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min

    # Compare the bounding box sizes
    assert np.allclose(bbox_size, usd_bbox_size), f"Bounding box sizes do not match for {model_id}. USD has {usd_bbox_size}, max has {bbox_size}"

    # Compute the theoretical base link params using the current bbox center and the offset from the USD
    bbox_offset = rt.Point3(*usd_bbox_offset)
    rotated_bbox_offset = base_rot * bbox_offset
    theoretical_base_link_pos = bbox_ctr - rotated_bbox_offset
    theoretical_base_link_rot = base_rot

    for link_name, kind_files in by_link_and_kind.items():
        link_obj = object_links[link_name]
        parsed_link_name = parse_name(link_obj.name)
        for kind, files in kind_files.items():
            fillable_meshes = []
            for path in files:
                fs = OSFS(str(path.parent))
                filename = path.name
                fillable_mesh = load_mesh(fs, filename, force="mesh", skip_materials=True)
                if not fillable_mesh.is_volume:
                    fillable_mesh = load_mesh(fs, filename, force="mesh", process=False, skip_materials=True)
                fillable_meshes.append(fillable_mesh)

            # Get a flattened list of vertices and faces
            all_vertices = []
            all_faces = []
            for split in fillable_mesh:
                vertices = [rt.Point3(*v.tolist()) for v in split.vertices]
                # Offsetting here by the past vertex count
                faces = [[v + len(all_vertices) + 1 for v in f.tolist()] for f in split.faces]
                all_vertices.extend(vertices)
                all_faces.extend(faces)

            # Create a new node for the fillable mesh
            fillable_obj = rt.Editable_Mesh()
            rt.ConvertToPoly(fillable_obj)
            fillable_type = "Mfillable" if kind == "enclosed" else "Mopenfillable"
            fillable_obj.name = f"{parsed_link_name.group('mesh_basename')}-{fillable_type}"
            
            # Add the vertices
            for v in all_vertices:
                rt.polyop.createVert(fillable_obj, v)

            # Add the faces
            for f in all_faces:
                rt.polyop.createPolygon(fillable_obj, f)

            # So far, everything is in the local coordinate system, e.g. where they would go if the
            # object base link was at the origin and had no rotation. We need to move it to where we
            # think the base link is (this is unknown in 3ds Max but we read it from the USD).
            fillable_obj.position = theoretical_base_link_pos
            fillable_obj.rotation = theoretical_base_link_rot

            # Optionally set its wire color
            fillable_obj.wirecolor = rt.yellow

            # Update the mesh to reflect changes
            rt.update(fillable_obj)

            # Parent the mesh
            fillable_obj.parent = link_obj

            # Check that the new element count is the same as the split count
            elems = {tuple(rt.polyop.GetElementsUsingFace(fillable_obj, i + 1)) for i in range(rt.polyop.GetNumFaces(fillable_obj))}
            assert len(elems) == len(fillable_meshes), f"{fillable_obj.name} has different number of faces in fillable mesh than in splits"
            elems = np.array(list(elems))
            assert not np.any(np.sum(elems, axis=0) > 1), f"{fillable_obj.name} has same face appear in multiple elements"

            # Hide the mesh
            fillable_obj.isHidden = True


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
        import_fillable_volumes(model_id, object_links[model_id])

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
