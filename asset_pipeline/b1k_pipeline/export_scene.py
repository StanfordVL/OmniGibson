import json
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

from b1k_pipeline.mesh_tree import build_mesh_tree
from b1k_pipeline.export_objs import compute_bounding_box


def main():
    target = sys.argv[1]
    scene_name = os.path.split(target)[-1]
    object_list_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/object_list.json")
    mesh_root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/meshes")
    objects_root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/aggregate/objects")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/scene/urdf")

    # Load the mesh list from the object list json.
    with open(object_list_filename, "r") as f:
        mesh_list = json.load(f)["meshes"]

    # Build the mesh tree using our mesh tree library. The scene code also uses this system.
    G = build_mesh_tree(mesh_list, mesh_root_dir)

    # Go through each object.
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]

    scene_tree_root = ET.Element("robot")
    scene_tree_root.attrib = {"name": "igibson_scene"}
    world_link = ET.SubElement(scene_tree_root, "link")
    world_link.attrib = {"name": "world"}

    for root_node in roots:
        obj_cat, obj_model, obj_inst_id, _ = root_node
        obj_name = "-".join([obj_cat, obj_model])
        obj_name_in_scene = "-".join([obj_cat, obj_model, obj_inst_id])

        # Assert that this object is available in the objects folder.
        assert os.path.exists(os.path.join(objects_root_dir, obj_name.replace("-", "/"))), f"Could not find object {obj_name} in objects directory."

        # Get the relevant bbox info.
        bbox_size, _, bbox_world_center, bbox_world_rot = compute_bounding_box(G.nodes[root_node])

        # Save pose to scene URDF
        scene_link = ET.SubElement(scene_tree_root, "link")
        scene_link.attrib = {
            "category": obj_cat,
            "model": obj_model,
            "name": obj_name_in_scene,
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

    scene_urdf_file = os.path.join(output_dir, f"{scene_name}_best.urdf")
    os.makedirs(output_dir, exist_ok=True)
    xmlstr = minidom.parseString(ET.tostring(scene_tree_root)).toprettyxml(indent="   ")
    with open(scene_urdf_file, "w") as f:
        f.write(xmlstr)
    tree = ET.parse(scene_urdf_file)
    print(scene_urdf_file)
    tree.write(scene_urdf_file, xml_declaration=True)

    # If we got here, we were successful. Let's create the success file.
    success_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts/export_scene.success")
    with open(success_file, "w"):
        pass


if __name__ == "__main__":
    main()