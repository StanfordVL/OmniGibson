import io
import json
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from fs.zipfs import ZipFS

from b1k_pipeline.mesh_tree import build_mesh_tree
from b1k_pipeline.export_objs_global import compute_object_bounding_box
import b1k_pipeline.utils

SKIP_CATEGORIES = {}

def main():
    target = sys.argv[1]
    scene_name = os.path.split(target)[-1]
    pipeline_fs = b1k_pipeline.utils.PipelineFS()
    target_output_fs = pipeline_fs.target_output(target)

    # Load the mesh list from the object list json.
    with target_output_fs.open("object_list.json", "r") as f:
        mesh_list = json.load(f)["meshes"]

    # Build the mesh tree using our mesh tree library.
    # We don't need the upper side joints since we will only use these objects for bboxes.
    G = build_mesh_tree(mesh_list, target_output_fs, load_upper=False)

    # Go through each object.
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]

    scene_tree_root = ET.Element("robot")
    scene_tree_root.attrib = {"name": "igibson_scene"}
    world_link = ET.SubElement(scene_tree_root, "link")
    world_link.attrib = {"name": "world"}

    for root_node in roots:
        obj_cat, obj_model, obj_inst_id, _ = root_node
        if obj_cat in SKIP_CATEGORIES:
            continue

        obj_name_in_scene = "-".join([obj_cat, obj_model, obj_inst_id])
        obj_rooms = G.nodes[root_node]["metadata"]["layer_name"]
        # TODO: Verify rooms.
        if obj_rooms == "0":
            obj_rooms = ""

        # Get the relevant bbox info.
        bbox_size, _, bbox_world_center, bbox_world_rot = compute_object_bounding_box(G.nodes[root_node])

        # Save pose to scene URDF
        scene_link = ET.SubElement(scene_tree_root, "link")
        scene_link.attrib = {
            "category": obj_cat,
            "model": obj_model,
            "name": obj_name_in_scene,
            "rooms": obj_rooms,
        }
        scene_link.attrib["bounding_box"] = " ".join(["%.4f" % item for item in bbox_size])
        joint = ET.SubElement(scene_tree_root, "joint")
        joint.attrib = {
            "name": f"j_{obj_name_in_scene}",
            "type": "fixed" if G.nodes[root_node]["is_loose"] is None else "floating",
        }
        joint_origin = ET.SubElement(joint, "origin")
        joint_origin_xyz = bbox_world_center.tolist()
        joint_origin_rpy = bbox_world_rot.as_euler("xyz")
        joint_origin.attrib = {
            "xyz": " ".join([str(item) for item in joint_origin_xyz]),
            "rpy": " ".join([str(item) for item in joint_origin_rpy]),
        }
        joint_parent = ET.SubElement(joint, "parent")
        joint_parent.attrib = {"link": "world"}
        joint_child = ET.SubElement(joint, "child")
        joint_child.attrib = {"link": obj_name_in_scene}

    # Write, reparse, and write with header, using the XML library,
    xmlstr = minidom.parseString(ET.tostring(scene_tree_root)).toprettyxml(indent="   ")
    xmlio = io.StringIO(xmlstr)
    tree = ET.parse(xmlio)
    with target_output_fs.open("scene.urdf", "wb") as f:
        tree.write(f, xml_declaration=True)

    # If we got here, we were successful. Let's create the success file.
    with target_output_fs.open("export_scene.success", "w"):
        pass


if __name__ == "__main__":
    main()