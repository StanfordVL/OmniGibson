import copy
import json
import os
import re

import networkx as nx
import numpy as np
import tqdm
import trimesh


PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?$")
SCALE_FACTOR = 0.001
SCALE_MATRIX = trimesh.transformations.scale_matrix(SCALE_FACTOR)

def build_mesh_tree(mesh_list, mesh_root):
    G = nx.DiGraph()

    print("Building mesh tree.")
    pbar = tqdm.tqdm(mesh_list)
    for mesh_name in pbar:
        pbar.set_description(mesh_name)
        groups = PATTERN.match(mesh_name).groups()
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

        link_name = "base_link" if link_name is None else link_name

        node_key = (obj_cat, obj_model, obj_inst_id, link_name)
        if (obj_cat, obj_model, obj_inst_id, link_name) not in G.nodes:
            G.add_node((obj_cat, obj_model, obj_inst_id, link_name), is_broken=is_broken, is_loose=is_loose, is_randomization_fixed=is_randomization_fixed)
        
        # Get the path for the mesh
        mesh_dir = os.path.join(mesh_root, mesh_name)
        mesh_path = os.path.join(mesh_dir, "{}.obj".format(mesh_name))
        assert os.path.exists(mesh_path), f"Expected mesh {mesh_name} does not exist in directory."
        json_path = os.path.join(mesh_dir, "{}.json".format(mesh_name))
        assert os.path.isfile(json_path), f"Expected metadata for {mesh_name} could not be found."
        with open(json_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
            meta_links = metadata["meta_links"]

            # Delete meta links from metadata to avoid confusion
            del metadata["meta_links"]

            # Apply the scaling factor.
            for meta_link_type in meta_links:
                for meta_link in meta_links[meta_link_type].values():
                    meta_link["position"] = np.array(meta_link["position"]) * SCALE_FACTOR
                    if "length" in meta_link:
                        meta_link["length"] *= SCALE_FACTOR
                    if "width" in meta_link:
                        meta_link["width"] *= SCALE_FACTOR
                    if "size" in meta_link:
                        meta_link["size"] *= SCALE_FACTOR

        # Add the data for the position onto the node.
        if joint_limit == "upper":
            assert "upper_filename" not in G.nodes[node_key], f"Found two upper meshes for {node_key}"
            G.nodes[node_key]["upper_filename"] = mesh_path
            upper_mesh = trimesh.load(mesh_path, process=False, force="mesh", skip_materials=True, maintain_order=True)
            upper_mesh.apply_transform(SCALE_MATRIX)
            G.nodes[node_key]["upper_mesh"] = upper_mesh
        else:
            assert "lower_filename" not in G.nodes[node_key]
            G.nodes[node_key]["lower_filename"] = mesh_path, f"Found two lower meshes for {node_key}"
            lower_mesh = trimesh.load(mesh_path, process=False, force="mesh")
            lower_mesh.apply_transform(SCALE_MATRIX)
            G.nodes[node_key]["lower_mesh"] = lower_mesh

            lower_mesh_ordered = trimesh.load(mesh_path, process=False, force="mesh", skip_materials=True, maintain_order=True)
            lower_mesh_ordered.apply_transform(SCALE_MATRIX)
            G.nodes[node_key]["lower_mesh_ordered"] = lower_mesh_ordered

            G.nodes[node_key]["metadata"] = metadata
            G.nodes[node_key]["meta_links"] = meta_links
            G.nodes[node_key]["material_dir"] = os.path.join(mesh_dir, "material")

        # Add the edge in from the parent
        if link_name != "base_link":
            assert parent_link_name, f"Non-base_link {node_key} should have a parent link name."
            parent_key = (obj_cat, obj_model, obj_inst_id, parent_link_name)
            G.add_edge(parent_key, node_key, joint_type=joint_type)

    # Quick validation.
    for node, data in G.nodes(data=True):
        assert node[-1] == "base_link" or "upper_filename" in data, f"{node} does not have upper filename."
        assert node[-1] == "base_link" or "upper_mesh" in data, f"{node} does not have upper mesh."
        assert "lower_filename" in data, f"{node} does not have lower filename."
        assert "lower_mesh" in data, f"{node} does not have lower mesh."
        assert "lower_mesh_ordered" in data, f"{node} does not have lower mesh."
        assert "metadata" in data, f"{node} does not have metadata."

        if "upper_mesh" in data:
            lower_vertices = len(data["lower_mesh_ordered"].vertices)
            upper_vertices = len(data["upper_mesh"].vertices)
            assert lower_vertices == upper_vertices, f"{node} lower mesh has {lower_vertices} while upper mesh has {upper_vertices}. They should be equal."

        if node[3] == "base_link":
            assert G.in_degree(node) == 0, f"Base_link node {node} should have in degree 0."
        else:
            assert G.in_degree(node) != 0, f"Non-base_link node {node} should not have in degree 0."

    # Create combined mesh for each root node and add some data.
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
    for root in roots:
        nodes = nx.dfs_preorder_nodes(G, root)
        meshes = [G.nodes[node]["lower_mesh"] for node in nodes]
        combined_mesh = trimesh.util.concatenate(meshes)
        G.nodes[root]["combined_mesh"] = combined_mesh

    return G