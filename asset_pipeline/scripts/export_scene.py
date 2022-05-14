import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from xml.dom import minidom

import cv2
import numpy as np
import pybullet as p
import trimesh
from IPython import embed
from scipy.spatial.transform import Rotation as R

PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?$")


vray_mapping = {
    "VRayRawDiffuseFilterMap": "albedo",
    "VRayNormalsMap": "normal",
    "VRayMtlReflectGlossinessBake": "roughness",
    "VRayMetalnessMap": "metalness",
    "VRayRawRefractionFilterMap": "opacity",
    "VRaySelfIlluminationMap": "emission",
    "VRayAOMap": "ao",
}

mtl_mapping = {
    "map_Kd": "albedo",
    "map_bump": "normal",
    "map_Pr": "roughness",
    "map_": "metalness",
    "map_Tf": "opacity",
    "map_Ke": "emission",
    "map_Ks": "ao",
}

processed_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/objects")
scene_processed_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/scenes")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def build_object_hierarchy(scene_obj_root):
    """
    return: parent_to_children
    Key: (obj_cat, obj_model, obj_inst_id, is_broken, is_loose)
    Value: Dict[str: parent link name, set: a set of children links]
    """
    parent_to_children = OrderedDict()
    for obj in sorted(os.listdir(scene_obj_root)):
        obj_folder = os.path.join(scene_obj_root, obj)
        if not os.path.isdir(obj_folder):
            continue
        groups = PATTERN.match(obj).groups()
        (
            is_broken,
            is_randomization_fixed,
            is_loose,
            obj_cat,
            obj_model,
            obj_inst_id,
            link_name,
            parent_link_name,
            joint_type,
            joint_limit,
            light_id
        ) = groups

        # Only store the lower limit link
        if joint_limit == "upper":
            continue

        link_name = "base_link" if link_name is None else link_name

        obj_inst = (obj_cat, obj_model, obj_inst_id, is_broken, is_loose)
        if obj_inst not in parent_to_children:
            parent_to_children[obj_inst] = dict()

        if parent_link_name is None:
            parent_to_children[obj_inst]["root"] = [(obj, link_name, None)]
        else:
            if parent_link_name not in parent_to_children[obj_inst]:
                parent_to_children[obj_inst][parent_link_name] = []
            parent_to_children[obj_inst][parent_link_name].append((obj, link_name, joint_type))
    return parent_to_children


def fix_all_mtl_files(scene_obj_root):
    # Fix all the MTL files and texture maps in the root folder
    for obj in sorted(os.listdir(scene_obj_root)):
        obj_dir = os.path.join(scene_obj_root, obj)
        if not os.path.isdir(obj_dir):
            continue

        for mtl_file in os.listdir(obj_dir):
            if mtl_file.endswith(".mtl"):
                fix_mtl_file(obj_dir, mtl_file)


def fix_mtl_file(obj_dir, mtl_file):
    mtl_file = os.path.join(obj_dir, mtl_file)
    new_lines = []
    with open(mtl_file, "r") as f:
        for line in f.readlines():
            if "map" in line:
                line = line.replace("material\\", "material/")
                map_path = os.path.join(obj_dir, line.split(" ")[1].strip())
                # For some reason, pybullet won't load the texture files unless we save it again with OpenCV
                img = cv2.imread(map_path)
                # These two maps need to be flipped
                # glossiness -> roughness
                # translucency -> opacity
                if "VRayMtlReflectGlossinessBake" in map_path or "VRayRawRefractionFilterMap" in map_path:
                    img = 255 - img

                # TODO: maybe some of them also need to be squeezed into (H, W, 1) channels
                cv2.imwrite(map_path, img)

            new_lines.append(line)

    with open(mtl_file, "w") as f:
        for line in new_lines:
            f.write(line)


def get_pybullet_transformations():
    # Correct coordinate system change in pybullet
    coordinate_matrix = np.eye(4)
    coordinate_matrix[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()

    # Original mesh 1 unit = 1mm
    scale_matrix = trimesh.transformations.scale_matrix(0.001)

    return coordinate_matrix, scale_matrix


def scale_rotate_mesh(mesh, translation, rotation):
    coordinate_matrix, scale_matrix = get_pybullet_transformations()
    mesh.apply_transform(scale_matrix).apply_transform(coordinate_matrix)
    # Rotate the object around the CoM of the base link frame by the transpose of its canonical orientation
    # so that the object has identity orientation in the end
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R.from_quat(rotation).as_matrix().T
    mesh.apply_transform(trimesh.transformations.translation_matrix(-translation))
    mesh.apply_transform(rotation_matrix)
    mesh.apply_transform(trimesh.transformations.translation_matrix(translation))


def scale_rotate_meta_links(meta_links, link_name, translation, rotation):
    rotation_inv = R.from_quat(rotation).inv()
    for meta_link_type in meta_links:
        if link_name in meta_links[meta_link_type]:
            for meta_link in meta_links[meta_link_type][link_name]:
                meta_link["position"] = np.array(meta_link["position"])
                meta_link["position"] *= 0.001
                meta_link["position"] -= translation
                meta_link["position"] = np.dot(rotation_inv.as_matrix(), meta_link["position"])
                meta_link["position"] += translation
                meta_link["orientation"] = (rotation_inv * R.from_quat(meta_link["orientation"])).as_quat()
                meta_link["length"] *= 0.001
                meta_link["width"] *= 0.001


def normalize_meta_links(meta_links, link_name, offset):
    for meta_link_type in meta_links:
        if link_name in meta_links[meta_link_type]:
            for meta_link in meta_links[meta_link_type][link_name]:
                meta_link["position"] += offset


def get_base_link_center(mesh):
    mesh_copy = mesh.copy()
    coordinate_matrix, scale_matrix = get_pybullet_transformations()
    mesh_copy.apply_transform(scale_matrix).apply_transform(coordinate_matrix)
    return get_mesh_center(mesh_copy)


def get_mesh_center(mesh):
    if mesh.is_watertight:
        return mesh.center_mass
    else:
        return mesh.centroid


def main():
    scene_name = sys.argv[1]
    scene_obj_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad/scenes", scene_name, "artifacts/objects")

    p.connect(p.DIRECT)
    parent_to_children = build_object_hierarchy(scene_obj_root)
    fix_all_mtl_files(scene_obj_root)

    obj_model_names = {}
    existing_obj_model_names = {}
    broken_obj_folders = []
    scene_dir = os.path.join(scene_processed_folder, scene_name)
    scene_urdf_dir = os.path.join(scene_dir, "urdf")
    os.makedirs(scene_urdf_dir, exist_ok=True)

    scene_tree_root = ET.Element("robot")
    scene_tree_root.attrib = {"name": "igibson_scene"}
    world_link = ET.SubElement(scene_tree_root, "link")
    world_link.attrib = {"name": "world"}

    for obj_inst in parent_to_children:
        obj_cat, obj_model, obj_inst_id, is_broken, is_loose = obj_inst

        should_save_model = obj_inst_id == "0"

        obj_parent_to_children = parent_to_children[obj_inst]

        if should_save_model:
            obj_model_names[(obj_cat, obj_model)] = obj_model

        assert (obj_cat, obj_model) in obj_model_names, "missing instance 0 for this model: {}_{}".format(
            obj_cat, obj_model
        )
        obj_model_name = obj_model_names[(obj_cat, obj_model)]

        obj_link_name = "-".join([obj_cat, obj_model, obj_inst_id])
        is_building_structure = obj_cat in ["floors", "ceilings", "walls"]

        # TODO: get rid of test_
        if not is_building_structure:
            category = obj_cat
            processed_obj_inst_folder = os.path.join(processed_folder, category, obj_model_name)
        else:
            category = obj_cat
            processed_obj_inst_folder = scene_dir
            obj_model_name = scene_name

        # Extract base link orientation and position
        if not is_building_structure:
            obj_name = list(obj_parent_to_children["root"])[0][0]
            obj_dir = os.path.join(scene_obj_root, obj_name)
            json_file = os.path.join(obj_dir, "{}.json".format(obj_name))
            assert os.path.isfile(json_file)
            with open(json_file, "r") as f:
                metadata = json.load(f)
            canonical_orientation = np.array(metadata["orientation"])
            meta_links = metadata["meta_links"]

            obj_file = os.path.join(obj_dir, "{}.obj".format(obj_name))
            mesh = trimesh.load(obj_file, process=False)
            base_link_center = get_base_link_center(mesh)
        else:
            canonical_orientation = np.array([0.0, 0.0, 0.0, 1.0])
            base_link_center = np.zeros(3)

        if should_save_model:
            tree_root = ET.Element("robot")

            os.makedirs(processed_obj_inst_folder, exist_ok=True)
            if is_broken:
                broken_obj_folders.append(processed_obj_inst_folder)

            parent_sets = ["root"]
            parent_centers = {}
            while len(parent_sets) > 0:
                next_parent_sets = []
                for parent_link_name in parent_sets:
                    # Leaf nodes are skipped
                    if parent_link_name not in obj_parent_to_children:
                        continue

                    for obj_name, link_name, joint_type in obj_parent_to_children[parent_link_name]:
                        next_parent_sets.append(link_name)
                        print("\nprocessing", obj_name)
                        obj_dir = os.path.join(scene_obj_root, obj_name)
                        obj_file = os.path.join(obj_dir, "{}.obj".format(obj_name))
                        mesh = trimesh.load(obj_file, process=False)

                        scale_rotate_mesh(mesh, base_link_center, canonical_orientation)
                        scale_rotate_meta_links(meta_links, link_name, base_link_center, canonical_orientation)

                        center = get_mesh_center(mesh)

                        parent_centers[link_name] = center

                        # Cache "lower" mesh before translation
                        lower_mesh = mesh.copy()

                        # Make the mesh centered at its CoM
                        if not is_building_structure:
                            mesh.apply_translation(-center)
                            normalize_meta_links(meta_links, link_name, -center)

                        # Somehow we need to manually write the vertex normals to cache
                        mesh._cache.cache["vertex_normals"] = mesh.vertex_normals

                        # Save the mesh
                        obj_link_folder = os.path.join(processed_obj_inst_folder, obj_name)
                        os.makedirs(obj_link_folder, exist_ok=True)
                        obj_relative_path = "{}.obj".format(obj_name)
                        obj_path = os.path.join(obj_link_folder, obj_relative_path)
                        mesh.export(obj_path, file_type="obj")

                        # Move the mesh to the correct path
                        obj_link_mesh_folder = os.path.join(processed_obj_inst_folder, "shape")
                        os.makedirs(obj_link_mesh_folder, exist_ok=True)
                        obj_link_visual_mesh_folder = os.path.join(obj_link_mesh_folder, "visual")
                        os.makedirs(obj_link_visual_mesh_folder, exist_ok=True)
                        obj_link_collision_mesh_folder = os.path.join(obj_link_mesh_folder, "collision")
                        os.makedirs(obj_link_collision_mesh_folder, exist_ok=True)
                        obj_link_material_folder = os.path.join(processed_obj_inst_folder, "material")
                        os.makedirs(obj_link_material_folder, exist_ok=True)

                        src_mtl_file = os.path.join(obj_link_folder, "material_0.mtl")
                        dst_mtl_file = os.path.join(obj_link_visual_mesh_folder, "{}.mtl".format(obj_name))
                        shutil.copy(src_mtl_file, dst_mtl_file)

                        if not is_broken:
                            # Only non-broken models have texture baking
                            original_material_folder = os.path.join(obj_dir, "material")
                            for fname in os.listdir(original_material_folder):
                                # fname is in the same format as room_light-0-0_VRayAOMap.png
                                vray_name = fname[fname.index("VRay") : -4]
                                if vray_name in vray_mapping:
                                    dst_fname = vray_mapping[vray_name]
                                else:
                                    raise ValueError("Unknown texture map: {}".format(fname))

                                src_texture_file = os.path.join(original_material_folder, fname)
                                dst_texture_file = os.path.join(
                                    obj_link_material_folder, "{}_{}_{}.png".format(obj_link_name, link_name, dst_fname)
                                )
                                shutil.copy(src_texture_file, dst_texture_file)

                        src_obj_file = obj_path
                        visual_shape_file = os.path.join(obj_link_visual_mesh_folder, obj_relative_path)
                        shutil.copy(src_obj_file, visual_shape_file)

                        # Generate collision mesh
                        collision_shape_file = os.path.join(obj_link_collision_mesh_folder, obj_relative_path)
                        vhacd = os.path.join(os.path.dirname(__file__), "vhacd.exe")
                        subprocess.call([vhacd, "--input", visual_shape_file, "--output", collision_shape_file, "--log", "NUL"], shell=False, stdout=subprocess.DEVNULL)

                        # Remove the original saved OBJ folder
                        shutil.rmtree(obj_link_folder)

                        # Modify MTL reference in OBJ file
                        with open(visual_shape_file, "r") as f:
                            new_lines = []
                            for line in f.readlines():
                                if "mtllib material_0.mtl" in line:
                                    line = "mtllib {}.mtl\n".format(obj_name)
                                new_lines.append(line)

                        with open(visual_shape_file, "w") as f:
                            for line in new_lines:
                                f.write(line)

                        # Modify texture reference in MTL file
                        with open(dst_mtl_file, "r") as f:
                            new_lines = []
                            for line in f.readlines():
                                # TODO: bake multi-channel PBR texture
                                if "map_Kd material_0.png" in line:
                                    line = ""
                                    for key in mtl_mapping:
                                        line += "{} ../../material/{}_{}_{}.png\n".format(
                                            key, obj_link_name, link_name, mtl_mapping[key]
                                        )
                                new_lines.append(line)

                        with open(dst_mtl_file, "w") as f:
                            for line in new_lines:
                                f.write(line)

                        # Create the link in URDF
                        tree_root.attrib = {"name": obj_model_name}
                        link = ET.SubElement(tree_root, "link")
                        link.attrib = {"name": link_name}
                        visual = ET.SubElement(link, "visual")
                        visual_origin = ET.SubElement(visual, "origin")
                        visual_origin.attrib = {"xyz": " ".join([str(item) for item in [0.0] * 3])}
                        visual_geometry = ET.SubElement(visual, "geometry")
                        visual_mesh = ET.SubElement(visual_geometry, "mesh")
                        visual_mesh.attrib = {"filename": os.path.join("shape", "visual", obj_relative_path).replace("\\", "/")}

                        collision = ET.SubElement(link, "collision")
                        collision_origin = ET.SubElement(collision, "origin")
                        collision_origin.attrib = {"xyz": " ".join([str(item) for item in [0.0] * 3])}
                        collision_geometry = ET.SubElement(collision, "geometry")
                        collision_mesh = ET.SubElement(collision_geometry, "mesh")
                        collision_mesh.attrib = {"filename": os.path.join("shape", "collision", obj_relative_path).replace("\\", "/")}

                        if joint_type is not None:
                            # Find the upper joint limit mesh
                            upper_mesh_file = obj_file.replace("lower", "upper")
                            upper_mesh = trimesh.load(upper_mesh_file, process=False)
                            scale_rotate_mesh(upper_mesh, base_link_center, canonical_orientation)

                            # Find the center of the parent link
                            parent_center = parent_centers[parent_link_name]

                            if joint_type == "R":
                                # Revolute joint
                                num_v = lower_mesh.vertices.shape[0]
                                random_index = np.random.choice(num_v, min(num_v, 20), replace=False)
                                from_vertices = lower_mesh.vertices[random_index]
                                to_vertices = upper_mesh.vertices[random_index]

                                # Find joint axis and joint limit
                                r = R.align_vectors(
                                    to_vertices - np.mean(to_vertices, axis=0),
                                    from_vertices - np.mean(from_vertices, axis=0),
                                )[0]
                                upper_limit = r.magnitude()
                                assert upper_limit < np.deg2rad(
                                    175
                                ), "upper limit of revolute joint should be <175 degrees"
                                joint_axis_xyz = r.as_rotvec() / r.magnitude()

                                # Let X = from_vertices_mean, Y = to_vertices_mean, R is rotation, T is translation
                                # R * (X - T) + T = Y
                                # => (I - R) T = Y - R * X
                                # Find the translation part of the joint origin
                                r_mat = r.as_matrix()
                                from_vertices_mean = from_vertices.mean(axis=0)
                                to_vertices_mean = to_vertices.mean(axis=0)
                                left_mat = np.eye(3) - r_mat
                                t = np.linalg.lstsq(
                                    left_mat, (to_vertices_mean - np.dot(r_mat, from_vertices_mean)), rcond=None
                                )[0]

                                # The joint origin has infinite number of solutions along the joint axis
                                # Find the translation part of the joint origin that is closest to the CoM of the link
                                # by projecting the CoM onto the joint axis
                                a = t
                                b = t + joint_axis_xyz
                                pt = center

                                ap = pt - a
                                ab = b - a
                                joint_origin_xyz = a + np.dot(ap, ab) / np.dot(ab, ab) * ab

                                # Assign visual and collision mesh origin based on the diff between CoM and joint origin
                                mesh_offset = center - joint_origin_xyz
                                visual_origin.attrib = {"xyz": " ".join([str(item) for item in mesh_offset])}
                                collision_origin.attrib = {"xyz": " ".join([str(item) for item in mesh_offset])}

                                normalize_meta_links(meta_links, mesh_offset)

                                # Assign the joint origin relative to the parent CoM
                                joint_origin_xyz = joint_origin_xyz - parent_center
                            else:
                                # Prismatic joint
                                diff = upper_mesh.centroid - lower_mesh.centroid

                                # Find joint axis and joint limit
                                upper_limit = np.linalg.norm(diff)
                                joint_axis_xyz = diff / upper_limit

                                # Assign the joint origin relative to the parent CoM
                                joint_origin_xyz = center - parent_center

                            # Create the joint in the URDF
                            joint = ET.SubElement(tree_root, "joint")
                            joint.attrib = {
                                "name": "j_{}".format(link_name),
                                "type": "revolute" if joint_type == "R" else "prismatic",
                            }
                            joint_origin = ET.SubElement(joint, "origin")
                            joint_origin.attrib = {"xyz": " ".join([str(item) for item in joint_origin_xyz])}
                            joint_axis = ET.SubElement(joint, "axis")
                            joint_axis.attrib = {"xyz": " ".join([str(item) for item in joint_axis_xyz])}
                            joint_parent = ET.SubElement(joint, "parent")
                            joint_parent.attrib = {"link": parent_link_name}
                            joint_child = ET.SubElement(joint, "child")
                            joint_child.attrib = {"link": link_name}
                            joint_limit = ET.SubElement(joint, "limit")
                            joint_limit.attrib = {"lower": str(0.0), "upper": str(upper_limit)}

                parent_sets = next_parent_sets

            if not is_building_structure:
                urdf_path = os.path.join(processed_obj_inst_folder, "{}.urdf".format(obj_model_name))
            else:
                urdf_path = os.path.join(processed_obj_inst_folder, "urdf", "{}_{}.urdf".format(scene_name, category))

            xmlstr = minidom.parseString(ET.tostring(tree_root)).toprettyxml(indent="   ")
            with open(urdf_path, "w") as f:
                f.write(xmlstr)
            tree = ET.parse(urdf_path)
            tree.write(urdf_path, xml_declaration=True)
            print("\nsaving", processed_obj_inst_folder)

            if not is_building_structure:
                # Save medadata json
                body_id = p.loadURDF(urdf_path)
                lower, upper = p.getAABB(body_id)
                base_link_offset = ((np.array(lower) + np.array(upper)) / 2.0).tolist()
                bbox_size = (np.array(upper) - np.array(lower)).tolist()
                metadata = {
                    "base_link_offset": base_link_offset,
                    "bbox_size": bbox_size,
                    "meta_links": meta_links,
                }
                obj_misc_folder = os.path.join(processed_obj_inst_folder, "misc")
                os.makedirs(obj_misc_folder, exist_ok=True)
                metadata_file = os.path.join(obj_misc_folder, "metadata.json")
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, cls=NumpyEncoder)
        else:
            # If re-using a previously saved model, just load its metadata
            obj_misc_folder = os.path.join(processed_obj_inst_folder, "misc")
            metadata_file = os.path.join(obj_misc_folder, "metadata.json")
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            base_link_offset = metadata["base_link_offset"]
            bbox_size = metadata["bbox_size"]

        if is_broken:
            # The original model in the scene is broken, needs to use an existing model as an alternative
            if should_save_model:
                existing_model_folder = os.path.join(processed_folder, obj_cat)
                obj_model_name = random.choice(os.listdir(existing_model_folder))
                existing_obj_model_names[(obj_cat, obj_model)] = obj_model_name

            assert (obj_cat, obj_model) in obj_model_names, "missing instance 0 for this model: {}_{}".format(
                obj_cat, obj_model
            )
            obj_model_name = existing_obj_model_names[(obj_cat, obj_model)]
            category = obj_cat

            obj_misc_folder = os.path.join(processed_folder, obj_cat, obj_model_name, "misc")
            metadata_file = os.path.join(obj_misc_folder, "metadata.json")
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            model_base_link_offset = metadata["base_link_offset"]
            model_bbox_size = metadata["bbox_size"]

            # Scale base_link_offset by the ratio of the two bounding boxes
            base_link_offset = np.array(model_base_link_offset) * (np.array(bbox_size) / np.array(model_bbox_size))

        if not is_building_structure:
            # Save the object into scene URDF
            rotated_offset = p.multiplyTransforms(
                [0, 0, 0], canonical_orientation, -np.array(base_link_offset), [0, 0, 0, 1]
            )[0]
            bbox_center = base_link_center - rotated_offset
        else:
            # Building structure always have [0, 0, 0] offset
            bbox_center = np.zeros(3)

        # Save pose to scene URDF
        scene_link = ET.SubElement(scene_tree_root, "link")
        scene_link.attrib = {
            "category": category,
            "model": obj_model_name,
            "name": obj_link_name,
        }
        if not is_building_structure:
            scene_link.attrib["bounding_box"] = " ".join([str(item) for item in bbox_size])
        joint = ET.SubElement(scene_tree_root, "joint")
        joint.attrib = {
            "name": "j_{}".format(obj_link_name),
            "type": "fixed" if is_loose is None else "floating",
        }
        joint_origin = ET.SubElement(joint, "origin")
        joint_origin_xyz = bbox_center.tolist()
        joint_origin_rpy = p.getEulerFromQuaternion(canonical_orientation)
        joint_origin.attrib = {
            "xyz": " ".join([str(item) for item in joint_origin_xyz]),
            "rpy": " ".join([str(item) for item in joint_origin_rpy]),
        }
        joint_parent = ET.SubElement(joint, "parent")
        joint_parent.attrib = {"link": "world"}
        joint_child = ET.SubElement(joint, "child")
        joint_child.attrib = {"link": obj_link_name}

    scene_urdf_file = os.path.join(scene_urdf_dir, "{}_best.urdf".format(scene_name))
    xmlstr = minidom.parseString(ET.tostring(scene_tree_root)).toprettyxml(indent="   ")
    with open(scene_urdf_file, "w") as f:
        f.write(xmlstr)
    tree = ET.parse(scene_urdf_file)
    print(scene_urdf_file)
    tree.write(scene_urdf_file, xml_declaration=True)

    for broken_obj_folder in broken_obj_folders:
        shutil.rmtree(broken_obj_folder)


if __name__ == "__main__":
    main()