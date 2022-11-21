import sys
sys.path.append(r"D:\ig_pipeline")

import json
import shutil
import string
import glob
from typing import Dict, List
import pymxs
import random
import os
from b1k_pipeline.urdfpy import URDF, Link, Joint
import trimesh.transformations
import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm
import yaml

rt = pymxs.runtime

JOINT_TYPES = {"prismatic": "P", "revolute": "R", "fixed": "F"}
IN_DATASET_ROOT = r"C:\Users\Cem\research\iGibson-dev\igibson\data\ig_dataset"
OUTPUT_ROOT = r"D:\test" # r"D:\ig_pipeline\cad\objects"

def get_maps_from_mat(mat):
    results = set()
    if mat:
        for i in range(mat.numsubs):
            submat = mat[i]
            if submat:
                if rt.classOf(submat) == rt.SubAnim:
                    if rt.isKindOf(submat.object, rt.textureMap):
                        results.add(submat.object)
                results.update(get_maps_from_mat(submat))
    return results
    
def get_all_maps():
    materials = {x.material for x in rt.objects}
    return {map for mat in materials for map in get_maps_from_mat(mat)}

def process_object_dir(model_dir):
    # Reset the scene
    rt.resetMaxFile(rt.name("noPrompt"))

    # Process the model name
    old_category_name = os.path.basename(os.path.dirname(model_dir))
    old_model_name = os.path.basename(model_dir)

    # Convert to new model name.
    translation_path = os.path.join(IN_DATASET_ROOT, "metadata", "model_rename.yaml")
    with open(translation_path, "r") as f:
        translation_dict = yaml.load(f, Loader=yaml.SafeLoader)
    new_category_name, new_model_name = translation_dict[old_category_name + "/" + old_model_name].split("/")

    # Get output path.
    obj_output_dir = os.path.join(OUTPUT_ROOT, f"legacy_{new_category_name}_{new_model_name}")

    # If the path already exists, assume the file has already been processed.
    if os.path.exists(obj_output_dir):
        return

    # Otherwise, start processing.
    try:
        # Load the URDF file into urdfpy
        urdf_filename = old_model_name + ".urdf"
        urdf_path = os.path.join(model_dir, urdf_filename)
        robot = URDF.load(urdf_path)

        # Load the metadata file too
        metadata_path = os.path.join(model_dir, "misc", "metadata.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)

        # Check that all of the object's joints are valid.
        joint_types = set(j.joint_type for j in robot.joints)
        invalid_joint_types = joint_types - set(JOINT_TYPES.keys())
        assert not invalid_joint_types, f"Invalid joint type found in list of joints: {invalid_joint_types}"

        # Check the list of links that end at openable joints.
        openable_joint_ids = metadata.get("openable_joint_ids", [])
        openable_links = {
            robot.link_map[robot.joint_map[j_name].child]
            for _, j_name in openable_joint_ids
        }

        # Do FK with everything at the lower position
        lfk : Dict[Link, np.ndarray] = robot.link_fk(cfg={joint.name: joint.limit.lower for joint in robot.joints})

        # Get the base link base mesh transform, revert it
        base_link = robot.base_link
        assert len(base_link.visuals) >= 1, "Base link has no visual mesh. How do we decide where the origin is?"
        real_base_link_transform = np.linalg.inv(base_link.visuals[0].origin)

        # Start tracking objects per link
        lower_objects_by_link = {}

        def process_link(link, link_pose, is_upper_side):
            found_identity_transform = False
            assert len(link.visuals) > 0, "Encountered link with no visual meshes."

            # Actually start the processing.
            first_mesh = None
            for visual in link.visuals:
                assert visual.geometry.mesh is not None, "Encountered link with primitive visuals."
                filename = os.path.abspath(os.path.join(model_dir, visual.geometry.mesh.filename))

                # Compute the pose
                pose = real_base_link_transform @ link_pose @ visual.origin
                if visual.geometry.mesh.scale is not None:
                    S = np.eye(4, dtype=np.float64)
                    S[:3,:3] = np.diag(visual.geometry.mesh.scale)
                    pose = pose @ S

                # Count this for check that at least one thing has trivial pose
                if np.allclose(pose, trimesh.transformations.identity_matrix()):
                    found_identity_transform = True

                # Convert the pose to a 3ds Max transform
                pose_formatted = pose[:3].T.tolist()
                mat = rt.Matrix3(rt.Point3(*pose_formatted[0]), rt.Point3(*pose_formatted[1]), rt.Point3(*pose_formatted[2]), rt.Point3(*pose_formatted[3]))

                # Compute the name
                if link == robot.base_link:
                    obj_name = f"{new_category_name}-{new_model_name}-0-base_link"
                else:
                    # Get the name based on the joint type/side.
                    link_name = link.name
                    joints : List[Joint] = robot.joints
                    j, = [j for j in joints if j.child == link_name]
                    parent_name = j.parent
                    if parent_name == robot.base_link.name:
                        parent_name = "base_link"
                    joint_type = JOINT_TYPES[j.joint_type]
                    side = "upper" if is_upper_side else "lower"
                    obj_name = f"{new_category_name}-{new_model_name}-0-{link_name}-{parent_name}-{joint_type}-{side}"

                    # Add openable tag to relevant joints
                    if link in openable_links:
                        obj_name = obj_name + "-Topenable"

                # Check if we are processing the 2nd instance (upper) of a link. If so, just copy an instance and we can return now since we just need
                # to process the first visual mesh in that case.
                if link in lower_objects_by_link:
                    lower_obj = lower_objects_by_link[link]
                    instance_objs = []
                    success, instance_objs = rt.maxOps.cloneNodes(lower_obj, cloneType=rt.name("instance"), newNodes=pymxs.byref(instance_objs))
                    assert success, "Could not clone object for upper end."
                    obj, = instance_objs
                    obj.transform = mat
                    obj.name = obj_name
                    return obj, found_identity_transform

                # Otherwise, import the object into 3ds Max
                rt.importFile(filename, pymxs.runtime.Name("noPrompt"))
                obj, = rt.selection
                obj.transform = mat
                obj.name = obj_name

                # If this is not the first mesh, we just attach it to the first mesh which is already fully processed.
                if first_mesh is not None:
                    rt.polyop.attach(first_mesh, obj)
                    continue

                # Add meta-links.
                if link == robot.base_link:
                    meta_links = metadata.get("links", {})
                    for metalink_name, metalink_info in meta_links.items():
                        metalink_type = metalink_info["geometry"]

                        # Get its position w.r.t. the URDF base link, transform it
                        metalink_orig_pos = np.array(metalink_info["xyz"])
                        metalink_pos = trimesh.transformations.transform_points(metalink_orig_pos[None, :], real_base_link_transform).flatten()

                        if metalink_type is None:
                            # For point metalinks, we use a PointHelper.
                            point = rt.Point()
                            point.cross = False
                            point.name = f"{new_category_name}-{new_model_name}-0-base_link-M{metalink_name}"
                            point.position = rt.Point3(*metalink_pos.tolist())
                            point.rotation = obj.rotation
                            point.parent = obj
                        elif metalink_type == "box":
                            # For box metalinks, we use a Box.
                            raise ValueError("Unsupported box metalink")
                            box = rt.Box()
                            box.name = f"{new_category_name}-{new_model_name}-0-base_link-M{metalink_name}"
                            box.rotation = obj.rotation * rt.Quat(*R.from_euler("XYZ", metalink_info["rpy"]).as_quat().tolist())
                            box.position = rt.Point3(*metalink_pos.tolist())
                            box.parent = obj
                            box.width, box.length, box.height = metalink_info["size"]
                            box.objectoffsetpos = rt.Point3(0, 0, box.height / 2.)
                        else:
                            # We don't know how to handle anything else really.
                            raise ValueError(f"Unknown metalink type {metalink_type}")

                # Keep track of the first mesh we process for attachment of later meshes.
                first_mesh = obj

            return first_mesh, found_identity_transform

        # First, get the lower range of everything based on the default lfk.
        found_identity_transform = False
        for link, link_pose in lfk.items():
            obj, any_identity_transform = process_link(link, link_pose, is_upper_side=False)
            lower_objects_by_link[link] = obj
            found_identity_transform = found_identity_transform or any_identity_transform
        assert found_identity_transform, "No object has identity transform."

        # Then, for each joint, add the upper limit.
        joints : List[Joint] = robot.joints
        for joint in joints:
            if joint.joint_type not in {"prismatic", "revolute"}:
                continue

            # Get some joint info.
            joint_name = joint.name
            joint_upper = joint.limit.upper
            child_link_name = joint.child
            link = robot.link_map[child_link_name]

            # Apply FK with only this joint set to upper position.
            joint_lfk = robot.link_fk(cfg={joint_name: joint_upper})
            link_pose = joint_lfk[link]

            # Process the child link with the joint's upper position.
            process_link(link, link_pose, is_upper_side=True)

        # Fix any bad materials, set the reflection color of the vray materials to be 0
        rt.convertToVRay(False)
        for obj in rt.objects:
            if obj.material:
                obj.material.reflection = rt.Color(0, 0, 0)

        # Update the image paths.
        texture_dir = os.path.join(obj_output_dir, "textures")
        os.makedirs(texture_dir, exist_ok=True)
        for map in get_all_maps():
            source_path = map.filename
            assert os.path.exists(source_path), f"Could not find texture map {source_path}"
            source_name, source_ext = os.path.splitext(os.path.basename(source_path))

            # Pick a semi-random target path
            while True:
                target_name = source_name + "-" + ''.join(random.choices(string.ascii_letters, k=6))
                target_path = os.path.join(texture_dir, target_name + source_ext)
                if not os.path.exists(target_path):
                    break

            # Copy the file to its new spot
            shutil.copy2(source_path, target_path)

            # Update the bitmap to have the right reference.
            map.filename = target_path

        # Finally, save the file.
        max_path = os.path.join(obj_output_dir, "processed.max")
        assert rt.saveMaxFile(max_path), "Could not save the file at {max_path}."
    except:
        # Remove the incomplete file.
        shutil.rmtree(obj_output_dir, ignore_errors=True)
        raise
    finally:
        # Don't leave some file hanging.
        rt.resetMaxFile(rt.name("noPrompt"))

if __name__ == "__main__":
    # Grab all the files from the dataset.
    obj_dirs = sorted(p for p in glob.glob(os.path.join(IN_DATASET_ROOT, "objects/*/*")) if os.path.isdir(p))
    # obj_dirs = [x for x in obj_dirs if r"monitor\3393" in x]

    failures = {}
    for obj_dir in tqdm.tqdm(obj_dirs):
        try:
            process_object_dir(obj_dir)
        except Exception as e:
            failures[obj_dir] = e

    print("Finished importing files.")
    print(f"Failures: {len(failures)} / {len(obj_dirs)}")
    sorted_failure_fns = sorted(failures.keys())
    for i, failure_fn in enumerate(sorted_failure_fns):
        print(f"{i+1} / {len(failures)} - {failure_fn}: {failures[failure_fn]}")