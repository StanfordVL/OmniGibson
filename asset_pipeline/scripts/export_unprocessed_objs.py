import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from xml.dom import minidom

import numpy as np
import pybullet as p
import trimesh
from scipy.spatial.transform import Rotation as R

from . import mesh_tree

PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?$")

# A sentinel value for use as the object tree root.
ROOT = object()

SCALE_COEFFICIENT = 0.001

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
            parent_to_children[obj_inst][ROOT] = [(obj, link_name, None)]
        else:
            if parent_link_name not in parent_to_children[obj_inst]:
                parent_to_children[obj_inst][parent_link_name] = []
            parent_to_children[obj_inst][parent_link_name].append((obj, link_name, joint_type))

    # Return in sorted instance ID order.
    return OrderedDict(sorted(parent_to_children.items(), key=lambda x: (x[0][0], x[0][1], int(x[0][2]))))


def get_frame_change_transformation():
    # Correct coordinate system change in pybullet
    coordinate_matrix = np.eye(4)
    coordinate_matrix[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()

    return coordinate_matrix


def scale_rotate_mesh(mesh, translation, rotation):
    coordinate_matrix = get_frame_change_transformation()
    mesh.apply_transform(coordinate_matrix)
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
            for meta_link in meta_links[meta_link_type][link_name].values():
                meta_link["position"] = np.array(meta_link["position"])
                meta_link["position"] *= SCALE_COEFFICIENT
                meta_link["position"] -= translation
                meta_link["position"] = np.dot(rotation_inv.as_matrix(), meta_link["position"])
                meta_link["position"] += translation
                meta_link["orientation"] = (rotation_inv * R.from_quat(meta_link["orientation"])).as_quat()

                if "length" in meta_link:
                    meta_link["length"] *= SCALE_COEFFICIENT

                if "width" in meta_link:
                    meta_link["width"] *= SCALE_COEFFICIENT

                if "size" in meta_link:
                    meta_link["size"] *= SCALE_COEFFICIENT


def normalize_meta_links(meta_links, link_name, offset):
    for meta_link_type in meta_links:
        if link_name in meta_links[meta_link_type]:
            for meta_link in meta_links[meta_link_type][link_name].values():
                meta_link["position"] += offset


def get_base_link_center(mesh):
    mesh_copy = mesh.copy()
    coordinate_matrix = get_frame_change_transformation()
    mesh_copy.apply_transform(coordinate_matrix)
    return get_mesh_center(mesh_copy)


def get_mesh_center(mesh):
    if mesh.is_watertight:
        return mesh.center_mass
    else:
        return mesh.centroid


def main():
    p.connect(p.DIRECT)

    target = sys.argv[1]
    mesh_root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/meshes")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/unprocessed_objects")

    G = mesh_tree.build_mesh_tree()

    # Go through each object.
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
    for root_node in roots:
        obj_cat, obj_model, obj_inst_id, _ = root_node

        # Only save the model for the 0th instance.
        should_save_model = int(obj_inst_id) == 0 and not G.nodes[root_node]["is_broken"]
        if not should_save_model:
            continue

        print("\nprocessing", root_node)
        obj_name = "-".join([obj_cat, obj_model, obj_inst_id])
        obj_output_dir = os.path.join(output_dir, obj_cat, obj_model)

        # Extract base link orientation and position
        metadata = G.nodes[root_node]["metadata"]
        canonical_orientation = np.array(metadata["orientation"])
        meta_links = metadata["meta_links"]

        mesh = G.nodes[root_node]["lower_mesh"]
        base_link_center = get_base_link_center(mesh)

        # Start creating the URDF file here.
        os.makedirs(obj_output_dir, exist_ok=True)
        tree_root = ET.Element("robot")

        # Iterate over each link.
        parent_sets = [ROOT]
        parent_centers = {}
        while len(parent_sets) > 0:
            next_parent_sets = []
            for parent_link_name in parent_sets:
                # Leaf nodes are skipped
                if parent_link_name not in obj_parent_to_children:
                    continue

                for obj_name, link_name, joint_type in obj_parent_to_children[parent_link_name]:
                    next_parent_sets.append(link_name)
                    obj_dir = os.path.join(mesh_root_dir, obj_name)
                    obj_file = os.path.join(obj_dir, "{}.obj".format(obj_name))

                    # Load the mesh.
                    mesh = trimesh.load(obj_file, process=False, force="mesh")

                    scale_rotate_mesh(mesh, base_link_center, canonical_orientation)
                    scale_rotate_meta_links(meta_links, link_name, base_link_center, canonical_orientation)

                    center = get_mesh_center(mesh)

                    parent_centers[link_name] = center

                    # Cache "lower" mesh before translation
                    lower_mesh = mesh.copy()

                    # Make the mesh centered at its CoM
                    mesh.apply_translation(-center)
                    normalize_meta_links(meta_links, link_name, -center)

                    # Somehow we need to manually write the vertex normals to cache
                    mesh._cache.cache["vertex_normals"] = mesh.vertex_normals

                    # Save the mesh
                    obj_link_folder = os.path.join(obj_output_dir, obj_name)
                    os.makedirs(obj_link_folder, exist_ok=True)
                    obj_relative_path = "{}.obj".format(obj_name)
                    obj_path = os.path.join(obj_link_folder, obj_relative_path)
                    mesh.export(obj_path, file_type="obj")

                    # Move the mesh to the correct path
                    obj_link_mesh_folder = os.path.join(obj_output_dir, "shape")
                    os.makedirs(obj_link_mesh_folder, exist_ok=True)
                    obj_link_visual_mesh_folder = os.path.join(obj_link_mesh_folder, "visual")
                    os.makedirs(obj_link_visual_mesh_folder, exist_ok=True)
                    obj_link_collision_mesh_folder = os.path.join(obj_link_mesh_folder, "collision")
                    os.makedirs(obj_link_collision_mesh_folder, exist_ok=True)
                    obj_link_material_folder = os.path.join(obj_output_dir, "material")
                    os.makedirs(obj_link_material_folder, exist_ok=True)

                    # Fix texture file paths if necessary.
                    original_material_folder = os.path.join(obj_dir, "material")
                    if False: # os.path.exists(original_material_folder):
                        for fname in os.listdir(original_material_folder):
                            # fname is in the same format as room_light-0-0_VRayAOMap.png
                            vray_name = fname[fname.index("VRay") : -4]
                            if vray_name in vray_mapping:
                                dst_fname = vray_mapping[vray_name]
                            else:
                                raise ValueError("Unknown texture map: {}".format(fname))

                            src_texture_file = os.path.join(original_material_folder, fname)
                            dst_texture_file = os.path.join(
                                obj_link_material_folder, "{}_{}_{}.png".format(obj_name, link_name, dst_fname)
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
                    src_mtl_file = os.path.join(obj_link_folder, "material_0.mtl")
                    if os.path.exists(src_mtl_file):
                        dst_mtl_file = os.path.join(obj_link_visual_mesh_folder, "{}.mtl".format(obj_name))
                        with open(dst_mtl_file, "r") as f:
                            new_lines = []
                            for line in f.readlines():
                                # TODO: bake multi-channel PBR texture
                                if "map_Kd material_0.png" in line:
                                    line = ""
                                    for key in mtl_mapping:
                                        line += "{} ../../material/{}_{}_{}.png\n".format(
                                            key, obj_name, link_name, mtl_mapping[key]
                                        )
                                new_lines.append(line)

                        with open(dst_mtl_file, "w") as f:
                            for line in new_lines:
                                f.write(line)

                    # Create the link in URDF
                    tree_root.attrib = {"name": obj_model}
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
                        upper_mesh = trimesh.load(upper_mesh_file, process=False, force="mesh")
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

                            normalize_meta_links(meta_links, link_name, mesh_offset)

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

        urdf_path = os.path.join(obj_output_dir, "object.urdf")

        xmlstr = minidom.parseString(ET.tostring(tree_root)).toprettyxml(indent="   ")
        with open(urdf_path, "w") as f:
            f.write(xmlstr)
        tree = ET.parse(urdf_path)
        tree.write(urdf_path, xml_declaration=True)
        print("saving", obj_output_dir)

        # Save medadata json
        body_id = p.loadURDF(urdf_path)  # TODO: Do we need this?
        lower, upper = p.getAABB(body_id)  # TODO: Use the trimesh aabb
        base_link_offset = ((np.array(lower) + np.array(upper)) / 2.0).tolist()
        bbox_size = (np.array(upper) - np.array(lower)).tolist()
        metadata = {
            "base_link_offset": base_link_offset,
            "bbox_size": bbox_size,
            "meta_links": meta_links,
        }
        obj_misc_folder = os.path.join(obj_output_dir, "misc")
        os.makedirs(obj_misc_folder, exist_ok=True)
        metadata_file = os.path.join(obj_misc_folder, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, cls=NumpyEncoder)

    success_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/export_unprocessed_objs.success")
    with open(success_file, "w"):
        pass

if __name__ == "__main__":
    main()