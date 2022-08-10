import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from xml.dom import minidom

import networkx as nx
import numpy as np
import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R

from . import mesh_tree


VRAY_MAPPING = {
    "VRayRawDiffuseFilterMap": "albedo",
    "VRayNormalsMap": "normal",
    "VRayMtlReflectGlossinessBake": "roughness",
    "VRayMetalnessMap": "metalness",
    "VRayRawRefractionFilterMap": "opacity",
    "VRaySelfIlluminationMap": "emission",
    "VRayAOMap": "ao",
}

MTL_MAPPING = {
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


def transform_mesh(orig_mesh, translation, rotation):
    mesh = orig_mesh.copy()

    # Rotate the object around the CoM of the base link frame by the transpose of its canonical orientation
    # so that the object has identity orientation in the end
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R.from_quat(rotation).inv().as_matrix()
    mesh.apply_transform(trimesh.transformations.translation_matrix(-translation))
    mesh.apply_transform(rotation_matrix)
    # mesh.apply_transform(trimesh.transformations.translation_matrix(translation))

    return mesh


def transform_meta_links(orig_meta_links, translation, rotation):
    meta_links = copy.deepcopy(orig_meta_links)
    rotation_inv = R.from_quat(rotation).inv()
    for meta_link_type in meta_links:
        for meta_link in meta_links[meta_link_type].values():
            meta_link["position"] = meta_link["position"]
            meta_link["position"] -= translation
            meta_link["position"] = np.dot(rotation_inv.as_matrix(), meta_link["position"])
            # meta_link["position"] += translation
            meta_link["orientation"] = (rotation_inv * R.from_quat(meta_link["orientation"])).as_quat()

    return meta_links


def normalize_meta_links(orig_meta_links, offset):
    meta_links = copy.deepcopy(orig_meta_links)
    for meta_link_type in meta_links:
        for meta_link in meta_links[meta_link_type].values():
            meta_link["position"] += offset

    return meta_links


def get_mesh_center(mesh):
    if mesh.is_watertight:
        return mesh.center_mass
    else:
        return mesh.centroid


def compute_bounding_box(root_node_data):
    combined_mesh = root_node_data["combined_mesh"]
    combined_mesh_center = get_mesh_center(combined_mesh)
    mesh_orientation = root_node_data["metadata"]["orientation"]
    canonical_combined_mesh = transform_mesh(combined_mesh, combined_mesh_center, mesh_orientation)
    base_link_offset = canonical_combined_mesh.bounding_box.centroid
    bbox_size = canonical_combined_mesh.bounding_box.extents

    # Compute the bbox world centroid
    bbox_world_rotation = R.from_quat(mesh_orientation)
    bbox_world_centroid = combined_mesh_center + bbox_world_rotation.apply(base_link_offset)

    return bbox_size, base_link_offset, bbox_world_centroid, bbox_world_rotation


def process_link(G, link_node, base_link_center, canonical_orientation, obj_name, obj_dir, tree_root):
    _, _, _, link_name = link_node
    raw_meta_links = G.nodes[link_node]["meta_links"]

    # Create a canonicalized copy of the lower and upper meshes.
    canonical_mesh = transform_mesh(G.nodes[link_node]["lower_mesh"], base_link_center, canonical_orientation)
    meta_links = transform_meta_links(raw_meta_links, base_link_center, canonical_orientation)

    # Somehow we need to manually write the vertex normals to cache
    canonical_mesh._cache.cache["vertex_normals"] = canonical_mesh.vertex_normals

    # Save the mesh
    with tempfile.TemporaryDirectory() as td:
        obj_relative_path = f"{obj_name}-{link_name}.obj"
        link_obj_path = os.path.join(td, obj_relative_path)
        canonical_mesh.export(link_obj_path, file_type="obj")

        # Move the mesh to the correct path
        obj_link_mesh_folder = os.path.join(obj_dir, "shape")
        os.makedirs(obj_link_mesh_folder, exist_ok=True)
        obj_link_visual_mesh_folder = os.path.join(obj_link_mesh_folder, "visual")
        os.makedirs(obj_link_visual_mesh_folder, exist_ok=True)
        obj_link_collision_mesh_folder = os.path.join(obj_link_mesh_folder, "collision")
        os.makedirs(obj_link_collision_mesh_folder, exist_ok=True)
        obj_link_material_folder = os.path.join(obj_dir, "material")
        os.makedirs(obj_link_material_folder, exist_ok=True)

        # Fix texture file paths if necessary.
        original_material_folder = os.path.join(td, "material")
        if os.path.exists(original_material_folder):
            for fname in os.listdir(original_material_folder):
                # fname is in the same format as room_light-0-0_VRayAOMap.png
                vray_name = fname[fname.index("VRay") : -4] if "VRay" in fname else None
                if vray_name in VRAY_MAPPING:
                    dst_fname = VRAY_MAPPING[vray_name]
                else:
                    raise ValueError(f"Unknown texture map: {fname}")

                src_texture_file = os.path.join(original_material_folder, fname)
                dst_texture_file = os.path.join(
                    obj_link_material_folder, f"{obj_name}-{link_name}-{dst_fname}.png"
                )
                shutil.copy(src_texture_file, dst_texture_file)

        visual_shape_file = os.path.join(obj_link_visual_mesh_folder, obj_relative_path)
        shutil.copy(link_obj_path, visual_shape_file)

        # Generate collision mesh
        collision_shape_file = os.path.join(obj_link_collision_mesh_folder, obj_relative_path)
        vhacd = os.path.join(os.path.dirname(__file__), "vhacd.exe")
        subprocess.call([vhacd, "--input", visual_shape_file, "--output", collision_shape_file, "--log", "NUL"], shell=False, stdout=subprocess.DEVNULL)

        # Modify MTL reference in OBJ file
        with open(visual_shape_file, "r") as f:
            new_lines = []
            for line in f.readlines():
                if "mtllib material_0.mtl" in line:
                    line = f"mtllib {obj_name}.mtl\n"
                new_lines.append(line)

        with open(visual_shape_file, "w") as f:
            for line in new_lines:
                f.write(line)

        # Modify texture reference in MTL file
        src_mtl_file = os.path.join(td, "material_0.mtl")
        if os.path.exists(src_mtl_file):
            dst_mtl_file = os.path.join(obj_link_visual_mesh_folder, f"{obj_name}.mtl")
            with open(src_mtl_file, "r") as f:
                new_lines = []
                for line in f.readlines():
                    # TODO: bake multi-channel PBR texture
                    if "map_Kd material_0.png" in line:
                        line = ""
                        for key in MTL_MAPPING:
                            line += f"{key} ../../material/{obj_name}-{link_name}-{MTL_MAPPING[key]}.png\n"
                    new_lines.append(line)

            with open(dst_mtl_file, "w") as f:
                for line in new_lines:
                    f.write(line)

    # Create the link in URDF
    
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

    in_edges = G.in_edges(link_node)
    assert len(in_edges) <= 1, f"Something's wrong: there's more than 1 in-edge to node {link_node}"

    # This object might be a base link and thus without an in-edge. Nothing to do then.
    if len(in_edges) == 1:
        # Grab the lone edge to the parent.
        edge, = in_edges
        parent_node, child_node = edge
        joint_type = G.edges[edge]["joint_type"]

        # Load the child node's meshes.
        lower_mesh = G.nodes[child_node]["lower_mesh"]
        upper_mesh = G.nodes[child_node]["upper_mesh"]

        # Load the centers.
        parent_center = get_mesh_center(G.nodes[parent_node]["lower_mesh"])
        child_center = get_mesh_center(lower_mesh)

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
            pt = child_center

            ap = pt - a
            ab = b - a
            joint_origin_xyz = a + np.dot(ap, ab) / np.dot(ab, ab) * ab

            # Assign visual and collision mesh origin based on the diff between CoM and joint origin
            mesh_offset = child_center - joint_origin_xyz
            visual_origin.attrib = {"xyz": " ".join([str(item) for item in mesh_offset])}
            collision_origin.attrib = {"xyz": " ".join([str(item) for item in mesh_offset])}

            meta_links = normalize_meta_links(meta_links, mesh_offset)

            # Assign the joint origin relative to the parent CoM
            joint_origin_xyz = joint_origin_xyz - parent_center
        else:
            # Prismatic joint
            diff = upper_mesh.centroid - lower_mesh.centroid

            # Find joint axis and joint limit
            upper_limit = np.linalg.norm(diff)
            joint_axis_xyz = diff / upper_limit

            # Assign the joint origin relative to the parent CoM
            joint_origin_xyz = child_center - parent_center

        # Create the joint in the URDF
        joint = ET.SubElement(tree_root, "joint")
        joint.attrib = {
            "name": f"j_{child_node[3]}",
            "type": "revolute" if joint_type == "R" else "prismatic",
        }
        joint_origin = ET.SubElement(joint, "origin")
        joint_origin.attrib = {"xyz": " ".join([str(item) for item in joint_origin_xyz])}
        joint_axis = ET.SubElement(joint, "axis")
        joint_axis.attrib = {"xyz": " ".join([str(item) for item in joint_axis_xyz])}
        joint_parent = ET.SubElement(joint, "parent")
        joint_parent.attrib = {"link": parent_node[3]}
        joint_child = ET.SubElement(joint, "child")
        joint_child.attrib = {"link": child_node[3]}
        joint_limit = ET.SubElement(joint, "limit")
        joint_limit.attrib = {"lower": str(0.0), "upper": str(upper_limit)}

    return meta_links


def process_object(G, root_node, output_dir):
    obj_cat, obj_model, obj_inst_id, _ = root_node
    obj_output_dir = os.path.join(output_dir, obj_cat, obj_model)
    os.makedirs(obj_output_dir, exist_ok=True)

    # Process the object
    obj_cat, obj_model, obj_inst_id, _ = root_node
    obj_name = "-".join([obj_cat, obj_model])

    # Prepare the URDF tree
    tree_root = ET.Element("robot")
    tree_root.attrib = {"name": obj_name}

    # Extract base link orientation and position
    base_link_metadata = G.nodes[root_node]["metadata"]
    canonical_orientation = np.array(base_link_metadata["orientation"])
    base_link_mesh = G.nodes[root_node]["lower_mesh"]
    base_link_center = get_mesh_center(base_link_mesh)

    meta_links = {}

    # Iterate over each link.
    for link_node in nx.dfs_preorder_nodes(G, root_node):
        _, _, _, link_name = link_node
        meta_links[link_name] = process_link(G, link_node, base_link_center, canonical_orientation, obj_name, obj_output_dir, tree_root)

    # Save the URDF file.
    urdf_path = os.path.join(obj_output_dir, f"{obj_model}.urdf")
    xmlstr = minidom.parseString(ET.tostring(tree_root)).toprettyxml(indent="   ")
    with open(urdf_path, "w") as f:
        f.write(xmlstr)
    tree = ET.parse(urdf_path)
    tree.write(urdf_path, xml_declaration=True)

    bbox_size, base_link_offset, _, _ = compute_bounding_box(G.nodes[root_node])

    # Save metadata json
    metadata = {
        "base_link_offset": base_link_offset.tolist(),
        "bbox_size": bbox_size.tolist(),
        "meta_links": meta_links,
    }
    obj_misc_folder = os.path.join(obj_output_dir, "misc")
    os.makedirs(obj_misc_folder, exist_ok=True)
    metadata_file = os.path.join(obj_misc_folder, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, cls=NumpyEncoder)

def main():
    target = sys.argv[1]
    object_list_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/object_list.json")
    mesh_root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/meshes")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/objects")

    # Load the mesh list from the object list json.
    with open(object_list_filename, "r") as f:
        mesh_list = json.load(f)["meshes"]

    # Build the mesh tree using our mesh tree library. The scene code also uses this system.
    G = mesh_tree.build_mesh_tree(mesh_list, mesh_root_dir)

    # Go through each object.
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]

    # Only save the 0th instance.
    saveable_roots = tqdm.tqdm([root_node for root_node in roots if int(root_node[2]) == 0 and not G.nodes[root_node]["is_broken"]])
    for root_node in saveable_roots:
        obj_cat, obj_model, obj_inst_id, _ = root_node
        saveable_roots.set_description(f"{obj_cat}-{obj_model}")

        # Start processing the object.
        process_object(G, root_node, output_dir)

    # If we got here, we were successful. Let's create the success file.
    success_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/export_objs.success")
    with open(success_file, "w"):
        pass

if __name__ == "__main__":
    main()