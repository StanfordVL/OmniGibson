import csv
import json

import networkx as nx
import tqdm
import trimesh
from fs.zipfs import ZipFS

from b1k_pipeline.utils import parse_name, load_mesh, PipelineFS


def build_mesh_tree(mesh_list, target_output_fs, load_upper=True, load_meshes=True, filter_nodes=None, show_progress=False):
    G = nx.DiGraph()

    # Load the rename file
    renames = {}
    with PipelineFS().open("metadata/object_renames.csv") as f:
        for row in csv.DictReader(f):
            key = (row["Original category (auto)"], row["ID (auto)"])
            renames[key] = row["New Category"]

    def maybe_rename_category(cat, model):
        if (cat, model) in renames:
            return renames[(cat, model)]
        return cat

    # Load the collision selections
    collision_selections = {}
    if target_output_fs.exists("collision_selection.json"):
        with target_output_fs.open("collision_selection.json", "r") as f:
            mesh_to_collision = json.load(f)
            match_to_collision = {parse_name(k): v for k, v in mesh_to_collision.items()}
            collision_selections = {(k.group("model_id"), k.group("link_name") if k.group("link_name") else "base_link"): v for k, v in match_to_collision.items() if k is not None}
    else:
        print("Warning: No collision selection file found. Collision meshes will not be loaded.")

    # Open the mesh filesystems
    mesh_fs = ZipFS(target_output_fs.open("meshes.zip", "rb"))
    collision_mesh_fs = ZipFS(target_output_fs.open("collision_meshes.zip", "rb"))

    pbar = tqdm.tqdm(mesh_list) if show_progress else mesh_list
    for mesh_name in pbar:
        if show_progress:
            pbar.set_description(mesh_name)
        match = parse_name(mesh_name)
        is_broken = match.group("bad")
        is_randomization_fixed = match.group("randomization_disabled")
        is_loose = match.group("loose")
        obj_model = match.group("model_id")
        obj_cat = maybe_rename_category(match.group("category"), obj_model)
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

        if filter_nodes is not None and node_key not in filter_nodes:
            continue

        if node_key not in G.nodes:
            G.add_node(node_key)
        G.nodes[node_key]["is_broken"] = is_broken
        G.nodes[node_key]["is_loose"] = is_loose
        G.nodes[node_key]["is_randomization_fixed"] = is_randomization_fixed
        G.nodes[node_key]["tags"] = tags
        
        # Get the path for the mesh
        mesh_dir = mesh_fs.opendir(mesh_name)
        mesh_fn = f"{mesh_name}.obj"
        with mesh_dir.open(f"{mesh_name}.json", "r") as metadata_file:
            metadata = json.load(metadata_file)

        meta_links = metadata["meta_links"]

        # Delete meta links from metadata to avoid confusion
        del metadata["meta_links"]

        # Add the data for the position onto the node.
        if joint_side == "upper":
            assert "upper_mesh" not in G.nodes[node_key], f"Found two upper meshes for {node_key}"
            if load_meshes:
                upper_mesh = load_mesh(mesh_dir, mesh_fn, process=False, force="mesh", skip_materials=True, maintain_order=True)
                G.nodes[node_key]["upper_mesh"] = upper_mesh
        else:
            G.nodes[node_key]["metadata"] = metadata
            G.nodes[node_key]["meta_links"] = meta_links
            G.nodes[node_key]["material_dir"] = mesh_dir.opendir("material") if mesh_dir.exists("material") else None

            if load_meshes:
                assert "lower_mesh" not in G.nodes[node_key], f"Found two lower meshes for {node_key}"
                lower_mesh = load_mesh(mesh_dir, mesh_fn, process=False, force="mesh")
                G.nodes[node_key]["lower_mesh"] = lower_mesh

                lower_mesh_ordered = load_mesh(mesh_dir, mesh_fn, process=False, force="mesh", skip_materials=True, maintain_order=True)
                G.nodes[node_key]["lower_mesh_ordered"] = lower_mesh_ordered

                # Attempt to load the collision mesh
                # First check if a collision mesh exist in the same directory
                collision_filenames = [x for x in mesh_dir.listdir("/") if "Mcollision" in x and x.endswith(".obj")]
                assert len(collision_filenames) <= 1, f"Found multiple collision meshes for {node_key}"
                if collision_filenames:
                    collision_filename, = collision_filenames
                    collision_mesh = load_mesh(mesh_dir, collision_filename, process=False, force="mesh", skip_materials=True)
                    G.nodes[node_key]["collision_mesh"] = collision_mesh
                elif mesh_name:
                    # Try to load a collision mesh selection
                    collision_key = (obj_model, link_name)
                    if collision_key in collision_selections:
                        collision_selection = collision_selections[collision_key]
                        try:
                            collision_mesh = load_mesh(collision_mesh_fs.opendir(mesh_name), collision_selection + ".obj", process=False, force="mesh", skip_materials=True)
                            G.nodes[node_key]["collision_mesh"] = collision_mesh
                        except:
                            pass
                            # raise

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
        assert not load_meshes or not needs_upper or "upper_mesh" in data, f"{node} does not have upper mesh."
        assert not load_meshes or "lower_mesh" in data, f"{node} does not have lower mesh."
        assert not load_meshes or "lower_mesh_ordered" in data, f"{node} does not have lower mesh."
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
    if load_meshes:
        roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
        for root in roots:
            nodes = nx.dfs_preorder_nodes(G, root)
            meshes = [G.nodes[node]["lower_mesh"] for node in nodes]
            combined_mesh = trimesh.util.concatenate(meshes)
            G.nodes[root]["combined_mesh"] = combined_mesh

            if all("collision_mesh" in G.nodes[node] for node in nodes):
                collision_meshes = [G.nodes[node]["collision_mesh"] for node in nodes]
                combined_collision_mesh = trimesh.util.concatenate(collision_meshes)
                G.nodes[root]["combined_collision_mesh"] = combined_collision_mesh

    return G
