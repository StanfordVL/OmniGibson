import json
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

from b1k_pipeline.mesh_tree import build_mesh_tree
from b1k_pipeline.export_objs_global import compute_object_bounding_box
from b1k_pipeline.utils import PIPELINE_ROOT

OFFSETS = {
    "restaurant_hotel": np.array([-13697.9,-14270.8,-3387.5]) / 1000.0,
    "office_vendor_machine": np.array([-427.945,5878.52,0]) / 1000.0,
}

def main():
    target = sys.argv[1]
    scene_name = os.path.split(target)[-1]
    object_list_filename = PIPELINE_ROOT / "cad" / target / "artifacts/object_list.json"
    mesh_root_dir = PIPELINE_ROOT / "cad" / target / "artifacts/meshes"
    objects_root_dir = PIPELINE_ROOT / "artifacts/aggregate/objects"
    output_dir = PIPELINE_ROOT / "cad" / target / "artifacts/scene/urdf"

    # Load the mesh list from the object list json.
    with open(object_list_filename, "r") as f:
        mesh_list = json.load(f)["meshes"]

    # Build the mesh tree using our mesh tree library.
    # We don't need the upper side joints since we will only use these objects for bboxes.
    G = build_mesh_tree(mesh_list, str(mesh_root_dir), load_upper=False)

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
        obj_rooms = G.nodes[root_node]["metadata"]["layer_name"]
        # TODO: Verify rooms.
        if obj_rooms == "0":
            obj_rooms = ""

        # Assert that this object is available in the objects folder.
        obj_path = objects_root_dir / obj_name.replace("-", "/")
        assert obj_path.exists(), f"Could not find object {obj_name} in objects directory."

        # Get the relevant bbox info.
        bbox_size, _, bbox_world_center, bbox_world_rot = compute_object_bounding_box(G.nodes[root_node])

        if scene_name in OFFSETS:
            bbox_world_center = bbox_world_center - OFFSETS[scene_name]

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

    scene_urdf_file = output_dir / f"{scene_name}_best.urdf"
    scene_urdf_file.parent.mkdir(parents=True, exist_ok=True)
    xmlstr = minidom.parseString(ET.tostring(scene_tree_root)).toprettyxml(indent="   ")
    with open(scene_urdf_file, "w") as f:
        f.write(xmlstr)
    tree = ET.parse(scene_urdf_file)
    print(scene_urdf_file)
    tree.write(scene_urdf_file, xml_declaration=True)

    # If we got here, we were successful. Let's create the success file.
    success_file = PIPELINE_ROOT / "cad" / target / "artifacts/export_scene.success"
    with open(success_file, "w"):
        pass


if __name__ == "__main__":
    main()