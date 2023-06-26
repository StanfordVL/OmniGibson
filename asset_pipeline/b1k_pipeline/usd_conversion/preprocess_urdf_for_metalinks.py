import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import omnigibson.utils.transform_utils as T

META_LINK_RENAME_MAPPING = {
    "fillable": "container",
    "fluidsink": "particlesink",
    "fluidsource": "particlesource",
}

ALLOWED_META_TYPES = {
    'particlesource': "dimensionless",
    'togglebutton': "primitive",
    'attachment': "dimensionless",
    'heatsource': "dimensionless",
    'particleapplier': "primitive",
    'particleremover': "primitive",
    'particlesink': "primitive",
    'slicer': "primitive",
    'container': "primitive",
    'collision': "convexmesh",
    'lights': "light",
}

def pretty_print_xml(current, parent=None, index=-1, depth=0, use_tabs=False):
    space = "\t" if use_tabs else " " * 4
    for i, node in enumerate(current):
        pretty_print_xml(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = "\n" + (space * depth)
        else:
            parent[index - 1].tail = "\n" + (space * depth)
        if index == len(parent) - 1:
            current.tail = "\n" + (space * (depth - 1))


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.
    Examples:
        [0, 1, 2] => "0 1 2"
    Args:
        array (n-array): Array to convert to a string
    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array])


def convert_to_string(inp):
    """
    Converts any type of {bool, int, float, list, tuple, array, string, np.str_} into an mujoco-xml compatible string.
        Note that an input string / np.str_ results in a no-op action.
    Args:
        inp: Input to convert to string
    Returns:
        str: String equivalent of @inp
    """
    if type(inp) in {list, tuple, np.ndarray}:
        return array_to_string(inp)
    elif type(inp) in {int, float, bool, np.float32, np.float64, np.int32, np.int64}:
        return str(inp).lower()
    elif type(inp) in {str, np.str_}:
        return inp
    else:
        raise ValueError("Unsupported type received: got {}".format(type(inp)))


def create_joint(
    name,
    parent,
    child,
    pos=(0, 0, 0),
    rpy=(0, 0, 0),
    joint_type="fixed",
    axis=None,
    damping=None,
    friction=None,
    limits=None,
):
    """
    Generates XML joint
    Args:
        name (str): Name of this joint
        parent (str or ET.Element): Name of parent link or parent link element itself for this joint
        child (str or ET.Element): Name of child link or child link itself for this joint
        pos (list or tuple or array): (x,y,z) offset pos values when creating the collision body
        rpy (list or tuple or array): (r,p,y) offset rot values when creating the joint
        joint_type (str): What type of joint to create. Must be one of {fixed, revolute, prismatic}
        axis (None or 3-tuple): If specified, should be (x,y,z) axis corresponding to DOF
        damping (None or float): If specified, should be damping value to apply to joint
        friction (None or float): If specified, should be friction value to apply to joint
        limits (None or 2-tuple): If specified, should be min / max limits to the applied joint
    Returns:
        ET.Element: Generated joint element
    """
    # Create the initial joint
    jnt = ET.Element("joint", name=name, type=joint_type)
    # Create origin subtag
    origin = ET.SubElement(
        jnt,
        "origin",
        attrib={"rpy": convert_to_string(rpy), "xyz": convert_to_string(pos)},
    )
    # Make sure parent and child are both names (str) -- if they're not str already, we assume it's the element ref
    if not isinstance(parent, str):
        parent = parent.get("name")
    if not isinstance(child, str):
        child = child.get("name")
    # Create parent and child subtags
    parent = ET.SubElement(jnt, "parent", link=parent)
    child = ET.SubElement(jnt, "child", link=child)
    # Add additional parameters if specified
    if axis is not None:
        ax = ET.SubElement(jnt, "axis", xyz=convert_to_string(axis))
    dynamic_params = {}
    if damping is not None:
        dynamic_params["damping"] = convert_to_string(damping)
    if friction is not None:
        dynamic_params["friction"] = convert_to_string(friction)
    if dynamic_params:
        dp = ET.SubElement(jnt, "dynamics", **dynamic_params)
    if limits is not None:
        lim = ET.SubElement(jnt, "limit", lower=limits[0], upper=limits[1])

    # Return this element
    return jnt


def create_link(name, subelements=None, mass=None, inertia=None):
    """
    Generates XML link element
    Args:
        name (str): Name of this link
        subelements (None or list): If specified, specifies all nested elements that should belong to this link
            (e.g.: visual, collision body elements)
        mass (None or float): If specified, will add an inertial tag with specified mass value
        inertia (None or 6-array): If specified, will add an inertial tag with specified inertia value
            Value should be (ixx, iyy, izz, ixy, ixz, iyz)
    Returns:
        ET.Element: Generated link
    """
    # Create the initial link
    link = ET.Element("link", name=name)
    # Add all subelements if specified
    if subelements is not None:
        for ele in subelements:
            link.append(ele)
    # Add mass subelement if requested
    if mass is not None or inertia is not None:
        inertial = ET.SubElement(link, "inertial")
    if mass is not None:
        ET.SubElement(inertial, "mass", value=convert_to_string(mass))
    if inertia is not None:
        axes = ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]
        inertia_vals = {ax: str(i) for ax, i in zip(axes, inertia)}
        ET.SubElement(inertial, "inertia", **inertia_vals)

    # Return this element
    return link


def create_metalink(
    root_element,
    metalink_name,
    parent_link_name="base_link",
    pos=(0, 0, 0),
    rpy=(0, 0, 0),
):
    # Create joint
    jnt = create_joint(
        name=f"{metalink_name}_joint",
        parent=parent_link_name,
        child=f"{metalink_name}_link",
        pos=pos,
        rpy=rpy,
        joint_type="fixed",
    )
    # Create child link
    link = create_link(
        name=f"{metalink_name}_link",
        mass=0.0001,
        inertia=[0.00001, 0.00001, 0.00001, 0, 0, 0],
    )

    # Add to root element
    root_element.append(jnt)
    root_element.append(link)


def generate_urdf_from_xmltree(root_element, name, dirpath, unique_urdf=False):
    """
    Generates a URDF file corresponding to @xmltree at @dirpath with name @name.urdf.
    Args:
        root_element (ET.Element): Element tree that compose the URDF
        name (str): Name of this file (name assigned to robot tag)
        dirpath (str): Absolute path to the location / filename for the generated URDF
        unique_urdf (bool): Whether to use a unique identifier when naming urdf (uses current datetime)
    Returns:
        str: Path to newly created urdf (fpath/<name>.urdf)
    """
    # Write to fpath, making sure the directory exists (if not, create it)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    # Get file
    date = (
        datetime.now()
        .isoformat(timespec="microseconds")
        .replace(".", "_")
        .replace(":", "_")
        .replace("-", "_")
    )
    fname = f"{name}_{date}.urdf" if unique_urdf else f"{name}.urdf"
    fpath = os.path.join(dirpath, fname)
    with open(fpath, "w") as f:
        # Write top level header line first
        f.write('<?xml version="1.0" ?>\n')
        # Convert xml to string form and write to file
        pretty_print_xml(current=root_element)
        xml_str = ET.tostring(root_element, encoding="unicode")
        f.write(xml_str)

    # Return path to file
    return fpath


def update_obj_urdf_with_metalinks(obj_category, obj_model, dataset_root):
    # Check if filepath exists
    model_root_path = f"{dataset_root}/objects/{obj_category}/{obj_model}"
    urdf_path = f"{model_root_path}/{obj_model}.urdf"

    # Load urdf
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Load metadata
    metadata_fpath = f"{model_root_path}/misc/metadata.json"
    with open(metadata_fpath, "r") as f:
        metadata = json.load(f)

    # Pop meta links
    assert not (
        "links" in metadata and "meta_links" in metadata
    ), "Only expected one of links and meta_links to be found in metadata, but found both!"

    if "meta_links" in metadata:
        # Rename meta links, e.g. from "fillable" to "container"
        for link, meta_link in metadata["meta_links"].items():
            for meta_link_name, meta_link_attrs in meta_link.items():
                if meta_link_name in META_LINK_RENAME_MAPPING:
                    metadata["meta_links"][link][META_LINK_RENAME_MAPPING[meta_link_name]] = meta_link_attrs
                    del metadata["meta_links"][link][meta_link_name]

        with open(metadata_fpath, "w") as f:
            json.dump(metadata, f)

        meta_links = metadata.pop("meta_links")
        print("meta_links:", meta_links)
        for parent_link_name, child_link_attrs in meta_links.items():
            for meta_link_name, ml_attrs in child_link_attrs.items():
                assert meta_link_name in ALLOWED_META_TYPES, f"meta_link_name {meta_link_name} not in {ALLOWED_META_TYPES}"

                for ml_id, attrs_list in ml_attrs.items():
                    if len(attrs_list) > 0:
                        if ALLOWED_META_TYPES[meta_link_name] != "dimensionless":
                            # If not dimensionless, we create one meta link for a list of meshes below it
                            attrs_list = [attrs_list[0]]
                        else:
                            # Otherwise, we create one meta link for each frame
                            # For non-attachment meta links, we expect only one instance per type
                            # E.g. heatsource_leftstove_0, heatsource_rightstove_0, but not heatsource_leftstove_1
                            if meta_link_name != "attachment":
                                assert len(attrs_list) == 1, f"Expected only one instance for meta_link {meta_link_name}_{ml_id}, but found {len(attrs_list)}"

                        for i, attrs in enumerate(attrs_list):
                            pos = attrs["position"]
                            quat = attrs["orientation"]

                            # For particle applier only, the orientation of the meta link matters (particle should shoot towards the negative z-axis)
                            # If the meta link is created based on the orientation of the first mesh that is a cone, we need to rotate it by 180 degrees
                            # because the cone is pointing in the wrong direction.
                            if meta_link_name == "particleapplier" and attrs["type"] == "cone":
                                assert len(attrs_list) == 1, f"Expected only one instance for meta_link {meta_link_name}_{ml_id}, but found {len(attrs_list)}"
                                quat = T.quat_multiply(quat, T.axisangle2quat([np.pi, 0.0, 0.0]))

                            # Create metalink
                            create_metalink(
                                root_element=root,
                                metalink_name=f"{meta_link_name}_{ml_id}_{i}",
                                parent_link_name=parent_link_name,
                                pos=pos,
                                rpy=T.quat2euler(quat),
                            )


    # Export this URDF
    return generate_urdf_from_xmltree(
        root_element=root,
        name=f"{obj_model}_with_metalinks",
        dirpath=model_root_path,
        unique_urdf=False,
    )
