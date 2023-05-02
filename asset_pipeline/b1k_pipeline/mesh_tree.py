import sys
sys.path.append(r"D:\ig_pipeline")

import json
import os
import re

import networkx as nx
import numpy as np
import tqdm
import trimesh

import b1k_pipeline.utils


def build_mesh_tree(mesh_list, mesh_fs, load_upper=True, show_progress=False, scale_factor=1):
    G = nx.DiGraph()

    scale_matrix = trimesh.transformations.scale_matrix(scale_factor)

    pbar = tqdm.tqdm(mesh_list) if show_progress else mesh_list
    for mesh_name in pbar:
        if show_progress:
            pbar.set_description(mesh_name)
        match = b1k_pipeline.utils.parse_name(mesh_name)
        is_broken = match.group("bad")
        is_randomization_fixed = match.group("randomization_disabled")
        is_loose = match.group("loose")
        obj_cat = match.group("category")
        obj_model = match.group("model_id")
        obj_inst_id = match.group("instance_id")
        link_name = match.group("link_name")
        parent_link_name = match.group("parent_link_name")
        joint_type = match.group("joint_type")
        joint_side = match.group("joint_side")
        tags_str = match.group("tag")

        if joint_side == "upper" and not load_upper:
            continue

        link_name = "base_link" if link_name is None else link_name

        tags = []
        if tags_str:
            tags = sorted([x[1:] for x in tags_str.split("-") if x])

        node_key = (obj_cat, obj_model, obj_inst_id, link_name)
        if node_key not in G.nodes:
            G.add_node(node_key)
        G.nodes[node_key]["is_broken"] = is_broken
        G.nodes[node_key]["is_loose"] = is_loose
        G.nodes[node_key]["is_randomization_fixed"] = is_randomization_fixed
        G.nodes[node_key]["tags"] = tags
        
        # Get the path for the mesh
        mesh_dir = mesh_fs.opendir(mesh_name)
        with mesh_dir.open("{mesh_name}.obj", "rb") as mesh_file, \
             mesh_dir.open("{mesh_name}.json", "r") as metadata_file:
            metadata = json.load(metadata_file)
            meta_links = metadata["meta_links"]

            # Delete meta links from metadata to avoid confusion
            del metadata["meta_links"]

            # Apply the scaling factor.
            for meta_link_id_to_subid in meta_links.values():
                for meta_link_subid_to_link in meta_link_id_to_subid.values():
                    for meta_link in meta_link_subid_to_link:
                        meta_link["position"] = np.array(meta_link["position"]) * scale_factor
                        if "length" in meta_link:
                            meta_link["length"] *= scale_factor
                        if "width" in meta_link:
                            meta_link["width"] *= scale_factor
                        if "size" in meta_link:
                            meta_link["size"] = (np.asarray(meta_link["size"]) * scale_factor).tolist()

        # Add the data for the position onto the node.
        if joint_side == "upper":
            assert "upper_mesh" not in G.nodes[node_key], f"Found two upper meshes for {node_key}"
            upper_mesh = trimesh.load(mesh_file, format="obj", process=False, force="mesh", skip_materials=True, maintain_order=True)
            upper_mesh.apply_transform(scale_matrix)
            G.nodes[node_key]["upper_mesh"] = upper_mesh
        else:
            assert "lower_mesh" not in G.nodes[node_key], f"Found two lower meshes for {node_key}"
            lower_mesh = trimesh.load(mesh_file, format="obj", process=False, force="mesh")
            lower_mesh.apply_transform(scale_matrix)
            G.nodes[node_key]["lower_mesh"] = lower_mesh

            lower_mesh_ordered = trimesh.load(mesh_file, format="obj", process=False, force="mesh", skip_materials=True, maintain_order=True)
            lower_mesh_ordered.apply_transform(scale_matrix)
            G.nodes[node_key]["lower_mesh_ordered"] = lower_mesh_ordered

            G.nodes[node_key]["metadata"] = metadata
            G.nodes[node_key]["meta_links"] = meta_links
            G.nodes[node_key]["material_dir"] = mesh_dir.opendir("material")

        # Add the edge in from the parent
        if link_name != "base_link":
            assert parent_link_name, f"Non-base_link {node_key} should have a parent link name."
            parent_key = (obj_cat, obj_model, obj_inst_id, parent_link_name)
            G.add_edge(parent_key, node_key, joint_type=joint_type)

    # Quick validation.
    for node, data in G.nodes(data=True):
        needs_upper = False
        if node[-1] != "base_link":
            (_, _, d), = G.in_edges(node, data=True)
            joint_type = d["joint_type"]
            needs_upper = load_upper and not data["is_broken"] and joint_type != "F"
        assert not needs_upper or "upper_mesh" in data, f"{node} does not have upper mesh."
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