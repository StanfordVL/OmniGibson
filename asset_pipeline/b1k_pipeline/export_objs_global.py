import collections
from concurrent import futures
import copy
import io
import json
import logging
import os
import pathlib
import sys
import shutil
import subprocess
import tempfile
import threading
import traceback
import xml.etree.ElementTree as ET
from collections import OrderedDict
from xml.dom import minidom

from dask.distributed import Client

import networkx as nx
import numpy as np
import tqdm
import trimesh
import trimesh.voxel.creation
from scipy.spatial.transform import Rotation as R

from b1k_pipeline import mesh_tree
from b1k_pipeline.utils import parse_name, PIPELINE_ROOT, get_targets

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

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
    "map_Pm": "metalness",
    "map_Tf": "opacity",
    # "map_Ke": "emission",
    # "map_Ks": "ao",
}

ALLOWED_PART_TAGS = {
    "subpart",
    "extrapart",
    "connectedpart",
}

# TODO: Make this use a local version if necessary.
# from dask_kubernetes.operator import KubeCluster
# cluster = KubeCluster(name="b1k-pipeline-vhacd", image="docker.io/igibson/vhacd-worker")
# cluster = KubeCluster.from_yaml('worker-template.yaml')
# cluster.scale(1)  # add 20 workers

# client = Client(cluster)
# client = Client('capri32.stanford.edu:8786')
VHACD_EXECUTABLE = "/svl/u/gabrael/v-hacd/app/build/TestVHACD"
# VHACD_EXECUTABLE = "TestVHACD"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def call_vhacd(obj_file_path, dest_file_path, dask_client):
    # This is the function that sends VHACD requests to a worker. It needs to read the contents
    # of the source file into memory, transmit that to the worker, receive the contents of the
    # result file and save those at the destination path.
    with open(obj_file_path, 'rb') as f:
        file_bytes = f.read()
    # data_future = client.scatter(file_bytes)
    data_future = file_bytes
    vhacd_future = dask_client.submit(
        run_vhacd_search,
        data_future,
        key=obj_file_path,
        retries=1)
    result = vhacd_future.result()
    if not result:
        raise ValueError("vhacd failed on object " + str(obj_file_path))
    with open(dest_file_path, 'wb') as f:
        f.write(result)

def get_vhacd_mesh(file_bytes, hull_count):
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.obj")
        out_path = os.path.join(td, "decomp.obj")  # This is the path that VHACD outputs to.
        with open(in_path, 'wb') as f:
            f.write(file_bytes)
        vhacd_cmd = [str(VHACD_EXECUTABLE), in_path, "-r", "1000000", "-d", "20", "-v", "60", "-h", str(hull_count)]
        try:
            proc = subprocess.run(vhacd_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=td, check=True)
            return trimesh.load(out_path, file_type="obj", force="mesh", skip_textures=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"VHACD failed with exit code {e.returncode}. Output:\n{e.output}")

def binary_search(x, f, low, high):
    mid = 0
 
    while low <= high:
        mid = (high + low) // 2
        val = f(mid)
        if val < x:
            low = mid + 1
        elif val > x:
            high = mid - 1
        else:
            return mid
    return low

def voxelize(m):
    pitch = 0.01
    return trimesh.voxel.creation.local_voxelize(
        m, pitch=pitch, point=np.array([0, 0, 0]),
        radius=int(np.ceil(0.5/pitch))
    ).matrix

def run_vhacd_search(visual_content):
    # First, run VHACD on the full thing with 128 hulls as an upper bound
    max_mesh = get_vhacd_mesh(visual_content, 128)
    translation = -np.mean(max_mesh.bounds, axis=0)
    scale = np.min(1 / max_mesh.extents)
    matrix_1 = trimesh.transformations.scale_matrix(scale)
    matrix_2 = trimesh.transformations.translation_matrix(translation)
    matrix = matrix_1 @ matrix_2
    max_mesh.apply_transform(matrix)  
    v_max_mesh = voxelize(max_mesh)

    # Define a binary-searchable function to find the best entry
    with tempfile.TemporaryDirectory() as td:
        memory = {}
        def compute_iou(log_hull_count):
            hull_count = 2 ** log_hull_count
            hull_count_mesh = get_vhacd_mesh(visual_content, hull_count)
            out_fn = pathlib.Path(td) / f"{hull_count}.obj"
            hull_count_mesh.export(str(out_fn), file_type="obj")
            hull_count_mesh.apply_transform(matrix)
            v_hull_count_mesh = voxelize(hull_count_mesh)

            # Compute their intersection volume
            intersection = v_max_mesh & v_hull_count_mesh
            intersection_cnt = np.count_nonzero(intersection)
            union = v_max_mesh | v_hull_count_mesh
            union_cnt = np.count_nonzero(union)
            iou = intersection_cnt / union_cnt

            memory[log_hull_count] = (out_fn, iou)

            return iou
        
        # Then, start binary search on the hull count to find the lowest entry above 0.85
        lowest_acceptable_log_hull_count = min(binary_search(0.85, compute_iou, 0, 6), 6)
        
        # Return the contents of the lowest acceptable hull count file
        lowest_acceptable_hull_file = memory[lowest_acceptable_log_hull_count][0]
        with open(lowest_acceptable_hull_file, "rb") as f:
            return f.read()

def timeout(timelimit):
    def decorator(func):
        def decorated(*args, **kwargs):
            with futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timelimit)
                except futures.TimeoutError:
                    raise TimeoutError from None
                executor._threads.clear()
                futures.thread._threads_queues.clear()
                return result
        return decorated
    return decorator


def transform_mesh(orig_mesh, translation, rotation):
    mesh = orig_mesh.copy()

    # Rotate the object around the CoM of the base link frame by the transpose of its canonical orientation
    # so that the object has identity orientation in the end

    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(rotation).as_matrix()
    transform[:3, 3] = translation
    inv_transform = trimesh.transformations.inverse_matrix(transform)
    mesh.apply_transform(inv_transform)

    # rotation_matrix = np.eye(4)
    # rotation_matrix[:3, :3] = R.from_quat(rotation).inv().as_matrix()
    # mesh.apply_transform(trimesh.transformations.translation_matrix(-translation))
    # mesh.apply_transform(rotation_matrix)
    ## mesh.apply_transform(trimesh.transformations.translation_matrix(R.from_quat(rotation).inv().apply(translation)))

    return mesh


def transform_meta_links(orig_meta_links, translation, rotation):
    meta_links = copy.deepcopy(orig_meta_links)
    rotation_inv = R.from_quat(rotation).inv()
    for meta_link_id_to_subid in meta_links.values():
        for meta_link_subid_to_link in meta_link_id_to_subid.values():
            for meta_link in meta_link_subid_to_link:
                meta_link["position"] = meta_link["position"]
                meta_link["position"] -= translation
                meta_link["position"] = np.dot(rotation_inv.as_matrix(), meta_link["position"])
                # meta_link["position"] += translation
                meta_link["orientation"] = (rotation_inv * R.from_quat(meta_link["orientation"])).as_quat()

    return meta_links


def normalize_meta_links(orig_meta_links, offset):
    meta_links = copy.deepcopy(orig_meta_links)
    for meta_link_id_to_subid in meta_links.values():
        for meta_link_subid_to_link in meta_link_id_to_subid.values():
            for meta_link in meta_link_subid_to_link:
                meta_link["position"] += offset

    return meta_links


def get_mesh_center(mesh):
    if mesh.is_watertight:
        return mesh.center_mass
    else:
        return mesh.centroid


def get_part_nodes(G, root_node):
    root_node_metadata = G.nodes[root_node]["metadata"]
    part_nodes = []
    for part_name in root_node_metadata["parts"]:
        # Find the part node
        part_name_parsed = parse_name(part_name)
        part_cat = part_name_parsed.group("category")
        part_model = part_name_parsed.group("model_id")
        part_inst_id = part_name_parsed.group("instance_id")
        part_link_name = part_name_parsed.group("link_name")
        part_link_name = "base_link" if part_link_name is None else part_link_name
        if part_link_name != "base_link":
            continue
        part_node_key = (part_cat, part_model, part_inst_id, part_link_name)
        if part_node_key not in G.nodes:
            print(list(G.nodes))
        assert part_node_key in G.nodes, f"Could not find part node {part_node_key}"
        part_nodes.append(part_node_key)

    return part_nodes


@timeout(10)
def compute_mesh_stable_poses(mesh):
    return trimesh.poses.compute_stable_poses(mesh, n_samples=5, threshold=0.03)

def compute_stable_poses(G, root_node):
    # First assemble a complete collision mesh
    all_link_meshes = []
    for link_node in nx.dfs_preorder_nodes(G, root_node):
        collision_mesh = G.nodes[link_node]["collision_mesh"].copy()
        mesh_pose_in_base_frame = G.nodes[link_node]["link_frame_in_base"] +  G.nodes[link_node]["mesh_in_link_frame"]
        transform = trimesh.transformations.translation_matrix(mesh_pose_in_base_frame)
        collision_mesh.apply_transform(transform)
        all_link_meshes.append(collision_mesh)
    combined_collision_mesh = trimesh.util.concatenate(all_link_meshes)

    # Then run trimesh's stable pose computation.
    poses = []
    probs = []
    try:
        poses, probs = compute_mesh_stable_poses(combined_collision_mesh)
    except:
        print("Could not compute stable poses.")
        pass

    # Return the obtained poses.
    return list({"prob": prob, "rotation": trimesh.transformations.quaternion_from_matrix(pose)} for pose, prob in zip(poses, probs))

def get_bbox_data_for_mesh(mesh):
    axis_aligned_bbox = mesh.bounding_box
    axis_aligned_bbox_dict = {
        "extent": np.array(axis_aligned_bbox.primitive.extents).tolist(),
        "transform": np.array(axis_aligned_bbox.primitive.transform).tolist(),
    }

    oriented_bbox = mesh.bounding_box_oriented
    oriented_bbox_dict = {
        "extent": np.array(oriented_bbox.primitive.extents).tolist(),
        "transform": np.array(oriented_bbox.primitive.transform).tolist(),
    }

    return {"axis_aligned": axis_aligned_bbox_dict, "oriented": oriented_bbox_dict}

def compute_link_aligned_bounding_boxes(G, root_node):
    link_bounding_boxes = collections.defaultdict(dict)
    for link_node in nx.dfs_preorder_nodes(G, root_node):
        obj_cat, obj_model, obj_inst_id, link_name = link_node
       
        # Get the pose and transform it
        for key in ["collision", "visual"]:
            try:
                link_bounding_boxes[link_name][key] = get_bbox_data_for_mesh(
                    transform_mesh(G.nodes[link_node][key + "_mesh"], G.nodes[link_node]["mesh_in_link_frame"], [0, 0, 0, 1]))
            except Exception as e:
                print(f"Problem with {obj_cat}-{obj_model} link {link_name}: {str(e)}")

    return link_bounding_boxes

def compute_object_bounding_box(root_node_data):
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


def process_link(G, link_node, base_link_center, canonical_orientation, obj_name, obj_dir, tree_root, out_metadata, dask_client):
    _, _, _, link_name = link_node
    raw_meta_links = G.nodes[link_node]["meta_links"]

    # Create a canonicalized copy of the lower and upper meshes.
    mesh_center = get_mesh_center(G.nodes[link_node]["lower_mesh_ordered"])
    canonical_mesh = transform_mesh(G.nodes[link_node]["lower_mesh"], mesh_center, canonical_orientation)
    meta_links = transform_meta_links(raw_meta_links, mesh_center, canonical_orientation)

    # Somehow we need to manually write the vertex normals to cache
    canonical_mesh._cache.cache["vertex_normals"] = canonical_mesh.vertex_normals

    in_edges = list(G.in_edges(link_node))
    assert len(in_edges) <= 1, f"Something's wrong: there's more than 1 in-edge to node {link_node}"

    # Save the mesh
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        
        obj_relative_path = f"{obj_name}-{link_name}.obj"
        link_obj_path = td / obj_relative_path
        canonical_mesh.export(str(link_obj_path), file_type="obj")

        # Move the mesh to the correct path
        obj_link_mesh_folder = obj_dir / "shape"
        obj_link_mesh_folder.mkdir(parents=True, exist_ok=True)
        obj_link_visual_mesh_folder = obj_link_mesh_folder / "visual"
        obj_link_visual_mesh_folder.mkdir(parents=True, exist_ok=True)
        obj_link_collision_mesh_folder = obj_link_mesh_folder / "collision"
        obj_link_collision_mesh_folder.mkdir(parents=True, exist_ok=True)
        obj_link_material_folder = obj_dir / "material"
        obj_link_material_folder.mkdir(parents=True, exist_ok=True)

        # Fix texture file paths if necessary.
        original_material_folder = pathlib.Path(G.nodes[link_node]["material_dir"])
        if original_material_folder.exists():
            for src_texture_file in original_material_folder.iterdir():
                fname = src_texture_file.name
                # fname is in the same format as room_light-0-0_VRayAOMap.png
                vray_name = fname[fname.index("VRay") : -4] if "VRay" in fname else None
                if vray_name in VRAY_MAPPING:
                    dst_fname = VRAY_MAPPING[vray_name]
                else:
                    raise ValueError(f"Unknown texture map: {fname}")

                dst_texture_file = obj_link_material_folder / f"{obj_name}-{link_name}-{dst_fname}.png"
                try:
                    shutil.copy(src_texture_file, dst_texture_file)
                except shutil.SameFileError:
                    pass

        visual_shape_file = obj_link_visual_mesh_folder / obj_relative_path
        try:
            shutil.copy(link_obj_path, visual_shape_file)
        except shutil.SameFileError:
            pass
        
        # Generate collision mesh
        collision_shape_file = obj_link_collision_mesh_folder / obj_relative_path
        call_vhacd(str(visual_shape_file.absolute()), str(collision_shape_file.absolute()), dask_client)
        # vhacd = PIPELINE_ROOT / "b1k_pipeline" / "vhacd.exe"
        # vhacd_cmd = [str(vhacd), "--input", str(visual_shape_file.absolute()), "--output", str(collision_shape_file.absolute()), "--log", "NUL", "--resolution", "10000000", "--depth 15"]
        # print("Running vhacd:", " ".join(vhacd_cmd))
        # assert subprocess.call(vhacd_cmd, shell=False, stdout=subprocess.DEVNULL) == 0
        # Store the final meshes
        G.nodes[link_node]["visual_mesh"] = canonical_mesh.copy()
        G.nodes[link_node]["collision_mesh"] = trimesh.load(str(collision_shape_file), process=False, force="mesh", skip_materials=True, maintain_order=True)

        # Modify MTL reference in OBJ file
        mtl_name = f"{obj_name}-{link_name}.mtl"
        with open(visual_shape_file, "r") as f:
            new_lines = []
            for line in f.readlines():
                if "mtllib material_0.mtl" in line:
                    line = f"mtllib {mtl_name}\n"
                new_lines.append(line)

        with open(visual_shape_file, "w") as f:
            for line in new_lines:
                f.write(line)

        # Modify texture reference in MTL file
        src_mtl_file = td / "material_0.mtl"
        if src_mtl_file.exists():
            dst_mtl_file = obj_link_visual_mesh_folder / mtl_name
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
    link_xml = ET.SubElement(tree_root, "link")
    link_xml.attrib = {"name": link_name}
    visual_xml = ET.SubElement(link_xml, "visual")
    visual_origin_xml = ET.SubElement(visual_xml, "origin")
    visual_origin_xml.attrib = {"xyz": " ".join([str(item) for item in [0.0] * 3])}
    visual_geometry_xml = ET.SubElement(visual_xml, "geometry")
    visual_mesh_xml = ET.SubElement(visual_geometry_xml, "mesh")
    visual_mesh_xml.attrib = {"filename": os.path.join("shape", "visual", obj_relative_path).replace("\\", "/")}

    collision_xml = ET.SubElement(link_xml, "collision")
    collision_origin_xml = ET.SubElement(collision_xml, "origin")
    collision_origin_xml.attrib = {"xyz": " ".join([str(item) for item in [0.0] * 3])}
    collision_geometry_xml = ET.SubElement(collision_xml, "geometry")
    collision_mesh_xml = ET.SubElement(collision_geometry_xml, "mesh")
    collision_mesh_xml.attrib = {"filename": os.path.join("shape", "collision", obj_relative_path).replace("\\", "/")}

    # This object might be a base link and thus without an in-edge. Nothing to do then.
    if len(in_edges) == 0:
        G.nodes[link_node]["link_frame_in_base"] = np.zeros(3)
        G.nodes[link_node]["mesh_in_link_frame"] = np.zeros(3)
    else:
        # Grab the lone edge to the parent.
        edge, = in_edges
        parent_node, child_node = edge
        joint_type = G.edges[edge]["joint_type"]
        parent_frame = G.nodes[parent_node]["link_frame_in_base"]

        # Load the meshes.
        lower_canonical_mesh = transform_mesh(G.nodes[child_node]["lower_mesh_ordered"], base_link_center + parent_frame, canonical_orientation)

        # Load the centers.
        child_center = get_mesh_center(lower_canonical_mesh)

        # Create the joint in the URDF
        joint_xml = ET.SubElement(tree_root, "joint")
        joint_xml.attrib = {
            "name": f"j_{child_node[3]}",
            "type": {"P": "prismatic", "R": "revolute", "F": "fixed"}[joint_type]
        }

        joint_parent_xml = ET.SubElement(joint_xml, "parent")
        joint_parent_xml.attrib = {"link": parent_node[3]}
        joint_child_xml = ET.SubElement(joint_xml, "child")
        joint_child_xml.attrib = {"link": child_node[3]}

        mesh_offset = np.zeros(3)
        if joint_type in ("P", "R"):
            upper_canonical_mesh = transform_mesh(G.nodes[child_node]["upper_mesh"], base_link_center + parent_frame, canonical_orientation)
            
            if joint_type == "R":
                # Revolute joint
                num_v_lower = lower_canonical_mesh.vertices.shape[0]
                num_v_upper = upper_canonical_mesh.vertices.shape[0]
                assert num_v_lower == num_v_upper, f"{child_node} lower mesh has {num_v_lower} vertices while upper has {num_v_upper}. These should match."
                num_v = num_v_lower
                random_index = np.random.choice(num_v, min(num_v, 20), replace=False)
                from_vertices = lower_canonical_mesh.vertices[random_index]
                to_vertices = upper_canonical_mesh.vertices[random_index]

                # Find joint axis and joint limit
                r = R.align_vectors(
                    to_vertices - np.mean(to_vertices, axis=0),
                    from_vertices - np.mean(from_vertices, axis=0),
                )[0]
                upper_limit = r.magnitude()
                assert upper_limit < np.deg2rad(
                    180
                ), "upper limit of revolute joint should be <180 degrees"
                joint_axis = r.as_rotvec() / r.magnitude()

                # Let X = from_vertices_mean, Y = to_vertices_mean, R is rotation, T is translation
                # R * (X - T) + T = Y
                # => (I - R) T = Y - R * X
                # Find the translation part of the joint origin
                r_mat = r.as_matrix()
                from_vertices_mean = from_vertices.mean(axis=0)
                to_vertices_mean = to_vertices.mean(axis=0)
                left_mat = np.eye(3) - r_mat
                arbitrary_point_on_joint_axis = np.linalg.lstsq(
                    left_mat, (to_vertices_mean - np.dot(r_mat, from_vertices_mean)), rcond=None
                )[0]

                # The joint origin has infinite number of solutions along the joint axis
                # Find the translation part of the joint origin that is closest to the CoM of the link
                # by projecting the CoM onto the joint axis
                arbitrary_point_to_center = child_center - arbitrary_point_on_joint_axis
                joint_origin = arbitrary_point_on_joint_axis  + (
                     np.dot(arbitrary_point_to_center, joint_axis) * joint_axis)

                # Assign visual and collision mesh origin so that the offset from the joint origin is removed.
                mesh_offset = child_center - joint_origin
                visual_origin_xml.attrib = {"xyz": " ".join([str(item) for item in mesh_offset])}
                collision_origin_xml.attrib = {"xyz": " ".join([str(item) for item in mesh_offset])}

                meta_links = normalize_meta_links(meta_links, mesh_offset)
            elif joint_type == "P":
                # TODO: Assert that the two are facing the same direction.

                # Prismatic joint
                diff = get_mesh_center(upper_canonical_mesh) - get_mesh_center(lower_canonical_mesh)

                # Find joint axis and joint limit
                if not np.allclose(diff, 0):
                    upper_limit = np.linalg.norm(diff)
                    joint_axis = diff / upper_limit
                else:
                    upper_limit = 0
                    joint_axis = np.array([0, 0, 1])

                # Assign the joint origin relative to the parent CoM
                joint_origin = child_center

            # Save these joints' data
            joint_origin_xml = ET.SubElement(joint_xml, "origin")
            joint_origin_xml.attrib = {"xyz": " ".join([str(item) for item in joint_origin])}
            joint_axis_xml = ET.SubElement(joint_xml, "axis")
            joint_axis_xml.attrib = {"xyz": " ".join([str(item) for item in joint_axis])}
            joint_limit_xml = ET.SubElement(joint_xml, "limit")
            joint_limit_xml.attrib = {"lower": str(0.0), "upper": str(upper_limit)}
        else:
            # Fixed joints are quite simple.
            joint_origin = child_center

            if joint_type == "F":
                joint_origin_xml = ET.SubElement(joint_xml, "origin")
                joint_origin_xml.attrib = {"xyz": " ".join([str(item) for item in joint_origin])}
            else:
                raise ValueError("Unexpected joint type: " + str(joint_type))
        
        G.nodes[link_node]["link_frame_in_base"] = parent_frame + joint_origin
        G.nodes[link_node]["mesh_in_link_frame"] = mesh_offset

    out_metadata["meta_links"][link_name] = meta_links
    out_metadata["link_tags"][link_name] = G.nodes[link_node]["tags"]


def process_object(G, root_node, output_dir, dask_client):
    obj_cat, obj_model, obj_inst_id, _ = root_node
    obj_output_dir = output_dir / obj_cat / obj_model
    obj_output_dir.mkdir(parents=True, exist_ok=True)

    # Process the object
    obj_cat, obj_model, obj_inst_id, _ = root_node
    obj_name = "-".join([obj_cat, obj_model])

    # Prepare the URDF tree
    tree_root = ET.Element("robot")
    tree_root.attrib = {"name": obj_model}

    # Extract base link orientation and position
    base_link_metadata = G.nodes[root_node]["metadata"]
    canonical_orientation = np.array(base_link_metadata["orientation"])
    base_link_mesh = G.nodes[root_node]["lower_mesh"]
    base_link_center = get_mesh_center(base_link_mesh)

    out_metadata = {
        "meta_links": {},
        "link_tags": {},
        "object_parts": [],
    }

    # Iterate over each link.
    for link_node in nx.dfs_preorder_nodes(G, root_node):
        process_link(G, link_node, base_link_center, canonical_orientation, obj_name, obj_output_dir, tree_root, out_metadata, dask_client)

    # Save the URDF file.
    urdf_path = obj_output_dir / f"{obj_model}.urdf"
    xmlstr = minidom.parseString(ET.tostring(tree_root)).toprettyxml(indent="   ")
    with open(urdf_path, "w") as f:
        f.write(xmlstr)
    tree = ET.parse(urdf_path)
    tree.write(urdf_path, xml_declaration=True)

    bbox_size, base_link_offset, _, _ = compute_object_bounding_box(G.nodes[root_node])

    # Compute part information
    for part_node_key in get_part_nodes(G, root_node):
        # Get the part node bounding box
        part_bb_size, _, part_bb_in_world_pos, part_bb_in_world_rot = compute_object_bounding_box(G.nodes[part_node_key])

        # Convert into our base link frame
        our_transform = np.eye(4)
        our_transform[:3, 3] = base_link_center
        our_transform[:3, :3] = R.from_quat(canonical_orientation).as_matrix()
        bb_transform = np.eye(4)
        bb_transform[:3, 3] = part_bb_in_world_pos
        bb_transform[:3, :3] = part_bb_in_world_rot.as_matrix()
        bb_transform_in_our = np.linalg.inv(our_transform) @ bb_transform
        bb_pos_in_our = bb_transform_in_our[:3, 3]
        bb_quat_in_our = R.from_matrix(bb_transform_in_our[:3, :3]).as_quat()

        # Get the part type
        part_tags = set(G.nodes[part_node_key]["tags"]) & ALLOWED_PART_TAGS
        assert(len(part_tags) == 1), f"Part node {part_node_key} has multiple part tags: {part_tags}"
        part_type, = part_tags

        # Add the metadata
        out_metadata["object_parts"].append({
            "category": part_node_key[0],
            "model": part_node_key[1],
            "type": part_type,
            "bb_pos": bb_pos_in_our,
            "bb_orn": bb_quat_in_our,
            "bb_size": part_bb_size,
        })

    # Save metadata json
    out_metadata.update({
        "base_link_offset": base_link_offset.tolist(),
        "bbox_size": bbox_size.tolist(),
        "orientations": compute_stable_poses(G, root_node),
        "link_bounding_boxes": compute_link_aligned_bounding_boxes(G, root_node),
    })
    obj_misc_folder = obj_output_dir / "misc"
    obj_misc_folder.mkdir(parents=True, exist_ok=True)
    metadata_file = obj_misc_folder / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(out_metadata, f, cls=NumpyEncoder)

def process_target(target, output_dir, executor, dask_client):
    object_list_filename = PIPELINE_ROOT / "cad" / target / "artifacts/object_list.json"
    mesh_root_dir = PIPELINE_ROOT / "cad" / target / "artifacts/meshes"

    with open(object_list_filename, "r") as f:
        mesh_list = json.load(f)["meshes"]

    # Build the mesh tree using our mesh tree library. The scene code also uses this system.
    G = mesh_tree.build_mesh_tree(mesh_list, str(mesh_root_dir))

    # Go through each object.
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]

    # Only save the 0th instance.
    saveable_roots = [root_node for root_node in roots if int(root_node[2]) == 0 and not G.nodes[root_node]["is_broken"]]
    # object_futures = {}
    for root_node in saveable_roots:
        # Start processing the object. We start by creating an object-specific
        # copy of the mesh tree (also including info about any parts)
        relevant_nodes = set(nx.dfs_tree(G, root_node).nodes())
        relevant_nodes |= {
            node
            for part_root_node in get_part_nodes(G, root_node)  # Get every part root node
            for node in nx.dfs_tree(G, part_root_node).nodes()}  # Get the subtree of each part
        Gprime = G.subgraph(relevant_nodes).copy()

        process_object(Gprime, root_node, output_dir, dask_client)
        # object_future = executor.submit(process_object, Gprime, root_node, output_dir, dask_client)
        # object_futures[object_future] = str(root_node)

    # return object_futures

def main():
    output_dir = PIPELINE_ROOT / "artifacts/aggregate/objects"
    json_file = PIPELINE_ROOT / "artifacts/pipeline/export_objs.json"

    # Load the mesh list from the object list json.
    errors = {}
    all_futures = {}

    dask_client = Client('svl17.stanford.edu:35423')
    
    with futures.ThreadPoolExecutor(max_workers=100) as executor:
        for target in tqdm.tqdm(get_targets("combined")):
            all_futures[executor.submit(process_target, target, output_dir, executor, dask_client)] = target
            # all_futures.update(process_target(target, output_dir, executor, dask_client))
                
        with tqdm.tqdm(total=len(all_futures)) as object_pbar:
            for future in futures.as_completed(all_futures.keys()):
                try:
                    result = future.result()
                except:
                    name = all_futures[future]
                    errors[name] = traceback.format_exc()

                object_pbar.update(1)

    with open(json_file, "w") as f:
        json.dump({"success": not errors, "errors": errors}, f)

if __name__ == "__main__":
    main()