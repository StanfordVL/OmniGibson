import json
import os
import re

import networkx as nx
import trimesh


PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?$")


def build_mesh_tree(mesh_list, mesh_root):
    G = nx.DiGraph()

    for mesh_name in mesh_list:
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

        # Add the data for the position onto the node.
        if joint_limit == "upper":
            assert "upper_filename" not in G.nodes[node_key], f"Found two upper meshes for {node_key}"
            G.nodes[node_key]["upper_filename"] = mesh_name
            G.nodes[node_key]["upper_mesh"] = trimesh.load(mesh_name, process=False, force="mesh")
        elif joint_limit == "lower":
            assert "lower_filename" not in G.nodes[node_key]
            G.nodes[node_key]["lower_filename"] = mesh_name, f"Found two lower meshes for {node_key}"
            G.nodes[node_key]["lower_mesh"] = trimesh.load(mesh_name, process=False, force="mesh")
            G.nodes[node_key]["metadata"] = metadata

        # Add the edge in from the parent
        if link_name != "base_link":
            assert parent_link_name, f"Non-base_link {node_key} should have a parent link name."
            parent_key = (obj_cat, obj_model, obj_inst_id, parent_link_name)
            G.add_edge(parent_key, node_key, joint_type=joint_type)

    # Quick validation.
    for node, data in G.nodes(data=True):
        assert "upper_filename" in data, f"{node} does not have upper filename."
        assert "upper_mesh" in data, f"{node} does not have upper mesh."
        assert "lower_filename" in data, f"{node} does not have upper filename."
        assert "lower_mesh" in data, f"{node} does not have upper mesh."
        assert "metadata" in data, f"{node} does not have metadata."

        if node[3] == "base_link":
            assert G.in_degree(node) == 0, f"Base_link node {node} should have in degree 0."
        else:
            assert G.in_degree(node) != 0, f"Non-base_link node {node} should not have in degree 0."

    # Create combined mesh for each root node.
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
    for root in roots:
        nodes = nx.dfs_preorder_nodes(G, root)
        meshes = [G.nodes[node]["lower_mesh"] for node in nodes]
        combined_mesh = trimesh.util.concatenate(meshes)
        G.nodes[root]["combined_mesh"] = combined_mesh

    return G