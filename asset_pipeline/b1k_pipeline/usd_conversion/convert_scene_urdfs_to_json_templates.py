import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects import DatasetObject
from omnigibson.scenes import Scene
from omnigibson.utils.config_utils import NumpyEncoder


def convert_scene_urdf_to_json(urdf, json_path):
    # First, load the requested objects from the URDF into OG
    load_scene_from_urdf(urdf=urdf)

    # Play the simulator, then save
    og.sim.play()
    Path(os.path.dirname(json_path)).mkdir(parents=True, exist_ok=True)
    og.sim.save(json_path=json_path)

    # Load the json, remove the init_info because we don't need it, then save it again
    with open(json_path, "r") as f:
        scene_info = json.load(f)

    scene_info.pop("init_info")

    with open(json_path, "w+") as f:
        json.dump(scene_info, f, cls=NumpyEncoder, indent=4)


def load_scene_from_urdf(urdf):
    # First, grab object info from the urdf
    objs_info = get_objects_config_from_scene_urdf(urdf=urdf)

    # Load all the objects manually into a scene
    scene = Scene(use_floor_plane=False)
    og.sim.import_scene(scene)

    for obj_name, obj_info in objs_info.items():
        try:
            if not os.path.exists(DatasetObject.get_usd_path(obj_info['cfg']['category'], obj_info['cfg']['model']).replace(".usd", ".encrypted.usd")):
                print("Missing object", obj_name)
                continue
            obj = DatasetObject(
                prim_path=f"/World/{obj_name}",
                name=obj_name,
                **obj_info["cfg"],
            )
            og.sim.import_object(obj)
            obj.set_bbox_center_position_orientation(
                position=obj_info["bbox_pos"], orientation=obj_info["bbox_quat"]
            )
        except Exception as e:
            raise ValueError(f"Failed to load object {obj_name}") from e

    # Take a sim step
    og.sim.step()


def string_to_array(string):
    """
    Converts an array string in mujoco xml to np.array.
    Examples:
        "0 1 2" => [0, 1, 2]
    Args:
        string (str): String to convert to an array
    Returns:
        np.array: Numerical array equivalent of @string
    """
    return np.array([float(x) for x in string.split(" ")])


def get_objects_config_from_scene_urdf(urdf):
    tree = ET.parse(urdf)
    root = tree.getroot()
    objects_cfg = dict()
    get_objects_config_from_element(root, model_pose_info=objects_cfg)
    return objects_cfg


def get_objects_config_from_element(element, model_pose_info):
    # First pass through, populate the joint pose info
    for ele in element:
        if ele.tag == "joint":
            name, pos, quat, fixed_jnt = get_joint_info(ele)
            name = name.replace("-", "_")
            model_pose_info[name] = {
                "bbox_pos": pos,
                "bbox_quat": quat,
                "cfg": {
                    "fixed_base": fixed_jnt,
                },
            }

    # Second pass through, import object models
    for ele in element:
        if ele.tag == "link":
            # This is a valid object, import the model
            name = ele.get("name").replace("-", "_")
            if name == "world":
                # Skip this
                pass
            else:
                print(name)
                assert (
                    name in model_pose_info
                ), f"Did not find {name} in current model pose info!"
                model_pose_info[name]["cfg"]["category"] = ele.get("category")
                model_pose_info[name]["cfg"]["model"] = ele.get("model")
                model_pose_info[name]["cfg"]["bounding_box"] = (
                    string_to_array(ele.get("bounding_box"))
                    if "bounding_box" in ele.keys()
                    else None
                )
                in_rooms = ele.get("rooms", "")
                if in_rooms:
                    in_rooms = in_rooms.split(",")
                model_pose_info[name]["cfg"]["in_rooms"] = in_rooms
                model_pose_info[name]["cfg"]["scale"] = (
                    string_to_array(ele.get("scale")) if "scale" in ele.keys() else None
                )
                model_pose_info[name]["cfg"]["bddl_object_scope"] = ele.get(
                    "object_scope", None
                )

        # If there's children nodes, we iterate over those
        for child in ele:
            get_objects_config_from_element(child, model_pose_info=model_pose_info)


def get_joint_info(joint_element):
    child, pos, quat, fixed_jnt = None, None, None, None
    fixed_jnt = joint_element.get("type") == "fixed"
    for ele in joint_element:
        if ele.tag == "origin":
            quat = T.mat2quat(T.euler2mat(string_to_array(ele.get("rpy"))))
            pos = string_to_array(ele.get("xyz"))
        elif ele.tag == "child":
            child = ele.get("link")
    return child, pos, quat, fixed_jnt
