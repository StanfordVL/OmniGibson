import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import OrderedDict
from copy import deepcopy
from os.path import exists
from pathlib import Path

import numpy as np
import omni
import omnigibson.utils.transform_utils as T
from omnigibson.utils.render_utils import make_glass
import pxr.Vt
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import close_stage, get_current_stage, open_stage
from omni.usd import create_material_input, get_shader_from_material
from omnigibson.macros import gm
from omnigibson.utils.usd_utils import BoundingBoxAPI, create_primitive_mesh
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from pxr.Sdf import ValueTypeNames as VT
from pxr.UsdGeom import Tokens

from b1k_pipeline.usd_conversion.preprocess_urdf_for_metalinks import ALLOWED_META_TYPES

LIGHT_MAPPING = {
    0: "Rect",
    2: "Sphere",
    4: "Disk",
}

OBJECT_STATE_TEXTURES = {
    "burnt",
    "cooked",
    "frozen",
    "soaked",
    "toggledon",
}

def set_mtl_albedo(mtl_prim, texture):
    mtl = "diffuse_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def set_mtl_normal(mtl_prim, texture):
    mtl = "normalmap_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def set_mtl_ao(mtl_prim, texture):
    mtl = "ao_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def set_mtl_roughness(mtl_prim, texture):
    mtl = "reflectionroughness_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    create_material_input(
        mtl_prim, "reflection_roughness_texture_influence", 1.0, VT.Float
    )
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def set_mtl_metalness(mtl_prim, texture):
    mtl = "metallic_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    create_material_input(mtl_prim, "metallic_texture_influence", 1.0, VT.Float)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def set_mtl_opacity(mtl_prim, texture):
    return
    mtl = "opacity_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    create_material_input(mtl_prim, "enable_opacity", True, VT.Bool)
    create_material_input(mtl_prim, "enable_opacity_texture", True, VT.Bool)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


def set_mtl_emission(mtl_prim, texture):
    mtl = "emissive_color_texture"
    create_material_input(mtl_prim, mtl, texture, VT.Asset)
    create_material_input(mtl_prim, "enable_emission", True, VT.Bool)
    # Verify it was set
    shade = get_shader_from_material(mtl_prim)
    print(f"mtl {mtl}: {shade.GetInput(mtl).Get()}")


RENDERING_CHANNEL_MAPPINGS = {
    "diffuse": set_mtl_albedo,
    "albedo": set_mtl_albedo,
    "normal": set_mtl_normal,
    "ao": set_mtl_ao,
    "roughness": set_mtl_roughness,
    "metalness": set_mtl_metalness,
    "opacity": set_mtl_opacity,
    "emission": set_mtl_emission,
}

MTL_MAP_TYPE_MAPPINGS = {
    "map_kd": "albedo",
    "map_bump": "normal",
    "map_pr": "roughness",
    "map_pm": "metalness",
    "map_tf": "opacity",
    "map_ke": "emission",
    "map_ks": "ao",
    "map_": "metalness",
}


def string_to_array(string):
    """
    Converts a array string in mujoco xml to np.array.
    Examples:
        "0 1 2" => [0, 1, 2]
    Args:
        string (str): String to convert to an array
    Returns:
        np.array: Numerical array equivalent of @string
    """
    return np.array([float(x) for x in string.split(" ")])


def get_joint_info(joint_element):
    child, pos, quat = None, None, None
    for ele in joint_element:
        if ele.tag == "origin":
            quat = T.convert_quat(
                T.mat2quat(T.euler2mat(string_to_array(ele.get("rpy")))), "wxyz"
            )
            pos = string_to_array(ele.get("xyz"))
        elif ele.tag == "child":
            child = ele.get("link")
    return child, pos, quat


def rename_prim(prim, name):
    path_from = prim.GetPrimPath().pathString
    path_to = f"{'/'.join(path_from.split('/')[:-1])}/{name}"
    omni.kit.commands.execute("MovePrim", path_from=path_from, path_to=path_to)
    return get_prim_at_path(path_to)


def get_visual_objs_from_urdf(urdf_path):
    # Will return a dictionary mapping link name (e.g.: base_link) to dictionary of owned visual meshes mapping mesh
    # name to visual obj file for that mesh
    visual_objs = OrderedDict()
    # Parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for ele in root:
        if ele.tag == "link":
            name = ele.get("name").replace("-", "_")
            visual_objs[name] = OrderedDict()
            for sub_ele in ele:
                if sub_ele.tag == "visual":
                    visual_mesh_name = sub_ele.get("name", "visuals").replace("-", "_")
                    obj_file = (
                        None
                        if sub_ele.find(".//mesh") is None
                        else sub_ele.find(".//mesh").get("filename")
                    )
                    if obj_file is None:
                        print(
                            f"Warning: No obj file found associated with {name}/{visual_mesh_name}!"
                        )
                    visual_objs[name][visual_mesh_name] = obj_file

    return visual_objs


def copy_object_state_textures(obj_category, obj_model, dataset_root):
    obj_root_dir = f"{dataset_root}/objects/{obj_category}/{obj_model}"
    old_mat_fpath = f"{obj_root_dir}/material"
    new_mat_fpath = f"{obj_root_dir}/usd/materials"
    for mat_file in os.listdir(old_mat_fpath):
        should_copy = False
        for object_state in OBJECT_STATE_TEXTURES:
            if object_state in mat_file.lower():
                should_copy = True
                break
        if should_copy:
            shutil.copy(f"{old_mat_fpath}/{mat_file}", new_mat_fpath)


def import_rendering_channels(
    obj_prim, obj_category, obj_model, model_root_path, usd_path, dataset_root
):
    usd_dir = os.path.dirname(usd_path)
    # # mat_dir = f"{model_root_path}/material/{obj_category}" if \
    # #     obj_category in {"ceilings", "walls", "floors"} else f"{model_root_path}/material"
    # mat_dir = f"{model_root_path}/material"
    # # Compile all material files we have
    # mat_files = set(os.listdir(mat_dir))

    # Remove the material prims as we will create them explictly later.
    # TODO: Be a bit more smart about this. a material procedurally generated will lose its material without it having
    # be regenerated
    stage = omni.usd.get_context().get_stage()
    for prim in obj_prim.GetChildren():
        looks_prim = None
        if prim.GetName() == "Looks":
            looks_prim = prim
        elif prim.GetPrimTypeInfo().GetTypeName() == "Xform":
            looks_prim_path = f"{str(prim.GetPrimPath())}/Looks"
            looks_prim = get_prim_at_path(looks_prim_path)
        if not looks_prim:
            continue
        for subprim in looks_prim.GetChildren():
            if subprim.GetPrimTypeInfo().GetTypeName() != "Material":
                continue
            print(
                f"Removed material prim {subprim.GetPath()}:",
                stage.RemovePrim(subprim.GetPath()),
            )

    # # Create new default material for this object.
    # mtl_created_list = []
    # omni.kit.commands.execute(
    #     "CreateAndBindMdlMaterialFromLibrary",
    #     mdl_name="OmniPBR.mdl",
    #     mtl_name="OmniPBR",
    #     mtl_created_list=mtl_created_list,
    # )
    # default_mat = get_prim_at_path(mtl_created_list[0])
    # default_mat = rename_prim(prim=default_mat, name=f"default_material")
    # print("Created default material:", default_mat.GetPath())
    #
    # # We may delete this default material if it's never used
    # default_mat_is_used = False

    # Grab all visual objs for this object
    urdf_path = f"{dataset_root}/objects/{obj_category}/{obj_model}/{obj_model}_with_metalinks.urdf"
    visual_objs = get_visual_objs_from_urdf(urdf_path)

    # Extract absolute paths to mtl files for each link
    link_mtl_files = (
        OrderedDict()
    )  # maps link name to dictionary mapping mesh name to mtl file
    mtl_infos = (
        OrderedDict()
    )  # maps mtl name to dictionary mapping material channel name to png file
    mat_files = (
        OrderedDict()
    )  # maps mtl name to corresponding list of material filenames
    mtl_old_dirs = (
        OrderedDict()
    )  # maps mtl name to corresponding directory where the mtl file exists
    mat_old_paths = (
        OrderedDict()
    )  # maps mtl name to corresponding list of relative mat paths from mtl directory
    for link_name, link_meshes in visual_objs.items():
        link_mtl_files[link_name] = OrderedDict()
        for mesh_name, obj_file in link_meshes.items():
            # Get absolute path and open the obj file if it exists:
            if obj_file is not None:
                obj_path = (
                    f"{dataset_root}/objects/{obj_category}/{obj_model}/{obj_file}"
                )
                with open(obj_path, "r") as f:
                    mtls = []
                    for line in f.readlines():
                        if "mtllib" in line and line[0] != "#":
                            mtls.append(line.split("mtllib ")[-1].split("\n")[0])
                    assert (
                        len(mtls) == 1
                    ), f"Only one mtl is supported per obj file in omniverse -- found {len(mtls)}!"
                mtl = mtls[0]
                # TODO: Make name unique
                mtl_name = (
                    ".".join(os.path.basename(mtl).split(".")[:-1])
                    .replace("-", "_")
                    .replace(".", "_")
                )
                mtl_old_dir = f"{'/'.join(obj_path.split('/')[:-1])}"
                link_mtl_files[link_name][mesh_name] = mtl_name
                mtl_infos[mtl_name] = OrderedDict()
                mtl_old_dirs[mtl_name] = mtl_old_dir
                mat_files[mtl_name] = []
                mat_old_paths[mtl_name] = []
                # Open the mtl file
                mtl_path = f"{mtl_old_dir}/{mtl}"
                with open(mtl_path, "r") as f:
                    # Read any lines beginning with map that aren't commented out
                    for line in f.readlines():
                        if line[:4] == "map_":
                            map_type, map_file = line.split(" ")
                            map_file = map_file.split("\n")[0]
                            map_filename = os.path.basename(map_file)
                            mat_files[mtl_name].append(map_filename)
                            mat_old_paths[mtl_name].append(map_file)
                            mtl_infos[mtl_name][
                                MTL_MAP_TYPE_MAPPINGS[map_type.lower()]
                            ] = map_filename

    # Next, for each material information, we create a new material and port the material files to the USD directory
    mat_new_fpath = os.path.join(usd_dir, "materials")
    Path(mat_new_fpath).mkdir(parents=True, exist_ok=True)
    shaders = OrderedDict()  # maps mtl name to shader prim
    for mtl_name, mtl_info in mtl_infos.items():
        for mat_old_path in mat_old_paths[mtl_name]:
            shutil.copy(
                os.path.join(mtl_old_dirs[mtl_name], mat_old_path), mat_new_fpath
            )

        # Create the new material
        mtl_created_list = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_created_list=mtl_created_list,
        )
        mat = get_prim_at_path(mtl_created_list[0])

        # Apply all rendering channels for this material
        for mat_type, mat_file in mtl_info.items():
            render_channel_fcn = RENDERING_CHANNEL_MAPPINGS.get(mat_type, None)
            if render_channel_fcn is not None:
                render_channel_fcn(mat, os.path.join("materials", mat_file))
            else:
                # Warn user that we didn't find the correct rendering channel
                print(
                    f"Warning: could not find rendering channel function for material: {mat_type}, skipping"
                )

        # Rename material
        mat = rename_prim(prim=mat, name=mtl_name)
        shade = UsdShade.Material(mat)
        shaders[mtl_name] = shade
        print(f"Created material {mtl_name}:", mtl_created_list[0])

    # Bind each (visual) mesh to its appropriate material in the object
    # We'll loop over each link, create a list of 2-tuples each consisting of (mesh_prim_path, mtl_name) to be bound
    root_prim_path = obj_prim.GetPrimPath().pathString
    for link_name, mesh_mtl_names in link_mtl_files.items():
        # Special case -- omni always calls the visuals "visuals" by default if there's only a single visual mesh for the
        # given
        if len(mesh_mtl_names) == 1:
            mesh_mtl_infos = [
                (
                    f"{root_prim_path}/{link_name}/visuals",
                    list(mesh_mtl_names.values())[0],
                )
            ]
        else:
            mesh_mtl_infos = []
            for mesh_name, mtl_name in mesh_mtl_names.items():
                # Omni only accepts a-z, A-Z as valid start characters for prim names
                # So we check if there is an invalid character, and modify it as we know Omni does
                if not ord("a") <= ord(mesh_name[0]) <= ord("z") and not ord(
                    "A"
                ) <= ord(mesh_name[0]) <= ord("Z"):
                    mesh_name = "a_" + mesh_name[1:]
                mesh_mtl_infos.append(
                    (f"{root_prim_path}/{link_name}/visuals/{mesh_name}", mtl_name)
                )
        for mesh_prim_path, mtl_name in mesh_mtl_infos:
            visual_prim = get_prim_at_path(mesh_prim_path)
            assert (
                visual_prim
            ), f"Error: Did not find valid visual prim at {mesh_prim_path}!"
            # Bind the created link material to the visual prim
            print(
                f"Binding material {mtl_name}, shader {shaders[mtl_name]}, to prim {mesh_prim_path}..."
            )
            UsdShade.MaterialBindingAPI(visual_prim).Bind(
                shaders[mtl_name], UsdShade.Tokens.strongerThanDescendants
            )

    # Lastly, we copy object_state texture maps that are state-conditioned; e.g.: cooked, soaked, etc.
    copy_object_state_textures(obj_category=obj_category, obj_model=obj_model, dataset_root=dataset_root)

    # ###################################
    #
    # # Iterate over all children of the object prim, if /<obj_name>/<link_name>/visual exists, then we
    # # know <link_name> is a valid link, and we check explicitly for these material files in our set
    # # Note: we assume that the link name is included as a string within the mat_file!
    # for prim in obj_prim.GetChildren():
    #     if prim.GetPrimTypeInfo().GetTypeName() == "Xform":
    #         # This could be a link, check if it owns a visual subprim
    #         link_name = prim.GetName()
    #         visual_prim = get_prim_at_path(f"{prim.GetPrimPath().pathString}/visuals")
    #         print(f"path: {prim.GetPrimPath().pathString}/visuals")
    #         print(f"visual prim: {visual_prim}")
    #
    #         if visual_prim:
    #             # Aggregate all material files for this prim
    #             link_mat_files = []
    #             for mat_file in deepcopy(mat_files):
    #                 if link_name in mat_file:
    #                     # Add this mat file and remove it from the set
    #                     link_mat_files.append(mat_file)
    #                     mat_files.remove(mat_file)
    #             # Potentially write material files for this prim if we have any valid materials
    #             print("link_mat_files:", link_mat_files)
    #             if not link_mat_files:
    #                 # Bind default material to the visual prim
    #                 shade = UsdShade.Material(default_mat)
    #                 UsdShade.MaterialBindingAPI(visual_prim).Bind(shade, UsdShade.Tokens.strongerThanDescendants)
    #                 default_mat_is_used = True
    #             else:
    #                 # Create new material for this link
    #                 mtl_created_list = []
    #                 omni.kit.commands.execute(
    #                     "CreateAndBindMdlMaterialFromLibrary",
    #                     mdl_name="OmniPBR.mdl",
    #                     mtl_name="OmniPBR",
    #                     mtl_created_list=mtl_created_list,
    #                 )
    #                 print(f"Created material for link {link_name}:", mtl_created_list[0])
    #                 mat = get_prim_at_path(mtl_created_list[0])
    #
    #                 shade = UsdShade.Material(mat)
    #                 # Bind the created link material to the visual prim
    #                 UsdShade.MaterialBindingAPI(visual_prim).Bind(shade, UsdShade.Tokens.strongerThanDescendants)
    #
    #                 # Iterate over all material channels and write them to the material
    #                 for link_mat_file in link_mat_files:
    #                     # Copy this file into the materials folder
    #                     mat_fpath = os.path.join(usd_dir, "materials")
    #                     shutil.copy(os.path.join(mat_dir, link_mat_file), mat_fpath)
    #                     # Check if any valid rendering channel
    #                     mat_type = link_mat_file.split("_")[-1].split(".")[0].lower()
    #                     # Apply the material if it exists
    #                     render_channel_fcn = RENDERING_CHANNEL_MAPPINGS.get(mat_type, None)
    #                     if render_channel_fcn is not None:
    #                         render_channel_fcn(mat, os.path.join("materials", link_mat_file))
    #                     else:
    #                         # Warn user that we didn't find the correct rendering channel
    #                         print(f"Warning: could not find rendering channel function for material: {mat_type}, skipping")
    #
    #                 # Rename material
    #                 mat = rename_prim(prim=mat, name=f"material_{link_name}")
    #
    # # For any remaining materials, we write them to the default material
    # # default_mat = get_prim_at_path(f"{obj_prim.GetPrimPath().pathString}/Looks/material_material_0")
    # # default_mat = get_prim_at_path(f"{obj_prim.GetPrimPath().pathString}/Looks/material_default")
    # print(f"default mat: {default_mat}, obj: {obj_category}, {prim.GetPrimPath().pathString}")
    # for mat_file in mat_files:
    #     # Copy this file into the materials folder
    #     mat_fpath = os.path.join(usd_dir, "materials")
    #     shutil.copy(os.path.join(mat_dir, mat_file), mat_fpath)
    #     # Check if any valid rendering channel
    #     mat_type = mat_file.split("_")[-1].split(".")[0].lower()
    #     # Apply the material if it exists
    #     render_channel_fcn = RENDERING_CHANNEL_MAPPINGS.get(mat_type, None)
    #     if render_channel_fcn is not None:
    #         render_channel_fcn(default_mat, os.path.join("materials", mat_file))
    #         default_mat_is_used = True
    #     else:
    #         # Warn user that we didn't find the correct rendering channel
    #         print(f"Warning: could not find rendering channel function for material: {mat_type}")
    #
    # # Possibly delete the default material prim if it was never used
    # if not default_mat_is_used:
    #     stage.RemovePrim(default_mat.GetPrimPath())


def add_xform_properties(prim):
    properties_to_remove = [
        "xformOp:rotateX",
        "xformOp:rotateXZY",
        "xformOp:rotateY",
        "xformOp:rotateYXZ",
        "xformOp:rotateYZX",
        "xformOp:rotateZ",
        "xformOp:rotateZYX",
        "xformOp:rotateZXY",
        "xformOp:rotateXYZ",
        "xformOp:transform",
    ]
    prop_names = prim.GetPropertyNames()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    # TODO: wont be able to delete props for non root links on articulated objects
    for prop_name in prop_names:
        if prop_name in properties_to_remove:
            prim.RemoveProperty(prop_name)
    if "xformOp:scale" not in prop_names:
        xform_op_scale = xformable.AddXformOp(
            UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, ""
        )
        xform_op_scale.Set(Gf.Vec3d([1.0, 1.0, 1.0]))
    else:
        xform_op_scale = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))

    if "xformOp:translate" not in prop_names:
        xform_op_translate = xformable.AddXformOp(
            UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, ""
        )
    else:
        xform_op_translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))

    if "xformOp:orient" not in prop_names:
        xform_op_rot = xformable.AddXformOp(
            UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, ""
        )
    else:
        xform_op_rot = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
    xformable.SetXformOpOrder([xform_op_translate, xform_op_rot, xform_op_scale])

def process_meta_link(stage, obj_model, meta_link_type, meta_link_infos):
    """
    Process a meta link by creating visual meshes or lights below it
    """
    assert meta_link_type in ALLOWED_META_TYPES
    if ALLOWED_META_TYPES[meta_link_type] not in ["primitive", "light"] and meta_link_type != "particlesource":
        return

    is_light = ALLOWED_META_TYPES[meta_link_type] == "light"

    for link_id, mesh_info_list in meta_link_infos.items():
        if len(mesh_info_list) == 0:
            continue

        if meta_link_type in ["togglebutton", "particleapplier", "particleremover", "particlesink", "particlesource"]:
            assert len(mesh_info_list) == 1, f"Invalid number of meshes for {meta_link_type}"

        meta_link_in_parent_link_pos, meta_link_in_parent_link_orn = mesh_info_list[0]["position"], mesh_info_list[0]["orientation"]

        # For particle applier only, the orientation of the meta link matters (particle should shoot towards the negative z-axis)
        # If the meta link is created based on the orientation of the first mesh that is a cone, we need to rotate it by 180 degrees
        # because the cone is pointing in the wrong direction. This is already done in update_obj_urdf_with_metalinks;
        # we just need to make sure meta_link_in_parent_link_orn is updated correctly.
        if meta_link_type == "particleapplier" and mesh_info_list[0]["type"] == "cone":
            meta_link_in_parent_link_orn = T.quat_multiply(meta_link_in_parent_link_orn, T.axisangle2quat([np.pi, 0.0, 0.0]))

        for i, mesh_info in enumerate(mesh_info_list):
            is_mesh = False
            if is_light:
                # Create a light
                light_type = LIGHT_MAPPING[mesh_info["type"]]
                prim_path = f"/{obj_model}/lights_{link_id}_0_link/light_{i}"
                prim = (
                    UsdLux.__dict__[f"{light_type}Light"]
                    .Define(stage, prim_path)
                    .GetPrim()
                )
                UsdLux.ShapingAPI.Apply(prim).GetShapingConeAngleAttr().Set(180.0)
            else:
                if meta_link_type == "particlesource":
                    mesh_type = "Cylinder"
                else:
                    # Create a primitive shape
                    mesh_type = mesh_info["type"].capitalize() if mesh_info["type"] != "box" else "Cube"
                prim_path = f"/{obj_model}/{meta_link_type}_{link_id}_0_link/mesh_{i}"
                assert mesh_type in UsdGeom.__dict__
                # togglebutton has to be a sphere
                if meta_link_type in ["togglebutton"]:
                    assert mesh_type in ["Sphere"], f"Invalid mesh type for togglebutton: {mesh_type}"
                    is_mesh = True
                # particle applier has to be a cone or cylinder because of the visualization of the particle flow
                elif meta_link_type in ["particleapplier"]:
                    assert mesh_type in ["Cone", "Cylinder"], f"Invalid mesh type for particleapplier: {mesh_type}"
                prim = create_primitive_mesh(prim_path, mesh_type, stage=stage).GetPrim() if is_mesh else UsdGeom.__dict__[mesh_type].Define(stage, prim_path).GetPrim()

            add_xform_properties(prim=prim)
            # Make sure mesh_prim has XForm properties
            xform_prim = XFormPrim(prim_path=prim_path)

            # Get the mesh/light pose in the parent link frame
            mesh_in_parent_link_pos, mesh_in_parent_link_orn = np.array(mesh_info["position"]), np.array(
                mesh_info["orientation"])

            # Get the mesh/light pose in the meta link frame
            mesh_in_meta_link_pos, mesh_in_meta_link_orn = \
                T.relative_pose_transform(mesh_in_parent_link_pos, mesh_in_parent_link_orn,
                                          meta_link_in_parent_link_pos, meta_link_in_parent_link_orn)

            if is_light:
                xform_prim.prim.GetAttribute("color").Set(
                    Gf.Vec3f(*np.array(mesh_info["color"]) / 255.0)
                )
                xform_prim.prim.GetAttribute("intensity").Set(mesh_info["intensity"])
                if light_type == "Rect":
                    xform_prim.prim.GetAttribute("height").Set(mesh_info["length"])
                    xform_prim.prim.GetAttribute("width").Set(mesh_info["width"])
                elif light_type == "Disk":
                    xform_prim.prim.GetAttribute("radius").Set(mesh_info["length"])
                elif light_type == "Sphere":
                    xform_prim.prim.GetAttribute("radius").Set(mesh_info["length"])
                else:
                    raise ValueError(f"Invalid light type: {light_type}")
            else:
                if mesh_type == "Cylinder":
                    if not is_mesh:
                        xform_prim.prim.GetAttribute("radius").Set(0.5)
                        xform_prim.prim.GetAttribute("height").Set(1.0)
                    if meta_link_type == "particlesource":
                        desired_radius = 0.0125
                        desired_height = 0.05
                        height_offset = -desired_height / 2.0
                    else:
                        desired_radius = mesh_info["size"][0]
                        desired_height = mesh_info["size"][2]
                        height_offset = desired_height / 2.0
                    xform_prim.prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(desired_radius * 2, desired_radius * 2, desired_height))
                    # Offset the position by half the height because in 3dsmax the origin of the cylinder is at the center of the base
                    mesh_in_meta_link_pos += T.quat2mat(mesh_in_meta_link_orn) @ np.array(
                        [0.0, 0.0, height_offset])
                elif mesh_type == "Cone":
                    if not is_mesh:
                        xform_prim.prim.GetAttribute("radius").Set(0.5)
                        xform_prim.prim.GetAttribute("height").Set(1.0)
                    desired_radius = mesh_info["size"][0]
                    desired_height = mesh_info["size"][2]
                    height_offset = -desired_height / 2.0
                    xform_prim.prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(desired_radius * 2, desired_radius * 2, desired_height))
                    # Flip the orientation of the z-axis because in 3dsmax the cone is pointing in the opposite direction
                    mesh_in_meta_link_orn = T.quat_multiply(mesh_in_meta_link_orn, T.axisangle2quat([np.pi, 0.0, 0.0]))
                    # Offset the position by half the height because in 3dsmax the origin of the cone is at the center of the base
                    mesh_in_meta_link_pos += T.quat2mat(mesh_in_meta_link_orn) @ np.array(
                        [0.0, 0.0, height_offset])
                elif mesh_type == "Cube":
                    if not is_mesh:
                        xform_prim.prim.GetAttribute("size").Set(1.0)
                    xform_prim.prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*mesh_info["size"]))
                elif mesh_type == "Sphere":
                    if not is_mesh:
                        xform_prim.prim.GetAttribute("radius").Set(0.5)
                    desired_radius = mesh_info["size"][0]
                    xform_prim.prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(desired_radius * 2, desired_radius * 2, desired_radius * 2))
                else:
                    raise ValueError(f"Invalid mesh type: {mesh_type}")

            xform_prim.set_local_pose(
                translation=mesh_in_meta_link_pos,
                orientation=T.convert_quat(mesh_in_meta_link_orn, to="wxyz")
            )

def process_glass_link(prim):
    # Update any glass parts to use the glass material instead
    glass_prim_paths = []
    for gchild in prim.GetChildren():
        if gchild.GetTypeName() == "Mesh":
            # check if has col api, if not, this is visual
            if not gchild.HasAPI(UsdPhysics.CollisionAPI):
                glass_prim_paths.append(gchild.GetPath().pathString)
        elif gchild.GetTypeName() == "Scope":
            # contains multiple additional prims, check those
            for ggchild in gchild.GetChildren():
                if ggchild.GetTypeName() == "Mesh":
                    # check if has col api, if not, this is visual
                    if not ggchild.HasAPI(UsdPhysics.CollisionAPI):
                        glass_prim_paths.append(ggchild.GetPath().pathString)

    assert glass_prim_paths

    stage = get_current_stage()
    root_path = stage.GetDefaultPrim().GetPath().pathString
    glass_mtl_prim_path = f"{root_path}/Looks/OmniGlass"
    if not get_prim_at_path(glass_mtl_prim_path):
        mtl_created = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniGlass.mdl",
            mtl_name="OmniGlass",
            mtl_created_list=mtl_created,
        )

    for glass_prim_path in glass_prim_paths:
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=glass_prim_path,
            material_path=glass_mtl_prim_path,
            strength=None,
        )

# TODO: Handle metalinks
# TODO: Import heights per link folder into USD folder
def import_obj_metadata(obj_category, obj_model, dataset_root, import_render_channels=False):
    # Check if filepath exists
    model_root_path = f"{dataset_root}/objects/{obj_category}/{obj_model}"
    usd_path = f"{model_root_path}/usd/{obj_model}.usd"

    # Load model
    open_stage(usd_path)
    stage = get_current_stage()
    prim = stage.GetDefaultPrim()

    data = dict()
    for data_group in {"metadata", "mvbb_meta", "material_groups", "heights_per_link"}:
        data_path = f"{model_root_path}/misc/{data_group}.json"
        if exists(data_path):
            # Load data
            with open(data_path, "r") as f:
                data[data_group] = json.load(f)

    # If certain metadata doesn't exist, populate with some core info
    if "base_link_offset" not in data["metadata"]:
        data["metadata"]["base_link_offset"] = [0, 0, 0]
    if "bbox_size" not in data["metadata"]:
        low_bb, high_bb = BoundingBoxAPI.compute_aabb(prim.GetPrimPath().pathString)
        data["metadata"]["bbox_size"] = (high_bb - low_bb).tolist()

    # Pop bb and base link offset and meta links info
    base_link_offset = data["metadata"].pop("base_link_offset")
    default_bb = data["metadata"].pop("bbox_size")

    # Manually modify material groups info
    if "material_groups" in data:
        data["material_groups"] = {
            "groups": data["material_groups"][0],
            "links": data["material_groups"][1],
        }

    # Manually modify metadata
    if "openable_joint_ids" in data["metadata"]:
        data["metadata"]["openable_joint_ids"] = {
            str(pair[0]): pair[1] for pair in data["metadata"]["openable_joint_ids"]
        }

    # Grab light info if any
    meta_links = data["metadata"].get("meta_links", dict())

    # TODO: Use parent link name
    for link_name, link_metadata in meta_links.items():
        for meta_link_type, meta_link_infos in link_metadata.items():
            process_meta_link(stage, obj_model, meta_link_type, meta_link_infos)

    # # Update metalink info
    # if "meta_links" in data["metadata"]:
    #     meta_links = data["metadata"].pop("meta_links")
    #     print("meta_links:", meta_links)
    #     # TODO: Use parent link name
    #     for parent_link_name, child_link_attrs in meta_links.items():
    #         for meta_link_name, ml_attrs in child_link_attrs.items():
    #             for ml_id, attrs_list in ml_attrs.items():
    #                 for i, attrs in enumerate(attrs_list):
    #                     # # Create new Xform prim that will contain info
    #                     ml_prim_path = (
    #                         f"{prim.GetPath()}/{meta_link_name}_{ml_id}_{i}_link"
    #                     )
    #                     link_prim = get_prim_at_path(ml_prim_path)
    #                     assert (
    #                         link_prim
    #                     ), f"Should have found valid metalink prim at prim path: {ml_prim_path}"
    #
    #                     link_prim.CreateAttribute("ig:is_metalink", VT.Bool)
    #                     link_prim.GetAttribute("ig:is_metalink").Set(True)
    #
    #                     # TODO! Validate that this works
    #                     # test on water sink 02: water sink location is 0.1, 0.048, 0.32
    #                     # water source location is -0.03724, 0.008, 0.43223
    #                     add_xform_properties(prim=link_prim)
    #                     link_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*atrr["xyz"]))
    #                     if atrr["rpy"] is not None:
    #                         link_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(*(T.euler2quat(atrr["rpy"])[[3, 0, 1, 2]])))
    #
    #                     link_prim.CreateAttribute("ig:orientation", VT.Quatf)
    #                     link_prim.GetAttribute("ig:orientation").Set(Gf.Quatf(*atrr["rpy"]))

    # Iterate over dict and replace any lists of dicts as dicts of dicts (with each dict being indexed by an integer)
    data = recursively_replace_list_of_dict(data)

    # Create attributes for bb, offset, category, model and store values
    prim.CreateAttribute("ig:nativeBB", VT.Vector3f)
    prim.CreateAttribute("ig:offsetBaseLink", VT.Vector3f)
    prim.CreateAttribute("ig:category", VT.String)
    prim.CreateAttribute("ig:model", VT.String)
    prim.GetAttribute("ig:nativeBB").Set(Gf.Vec3f(*default_bb))
    prim.GetAttribute("ig:offsetBaseLink").Set(Gf.Vec3f(*base_link_offset))
    prim.GetAttribute("ig:category").Set(obj_category)
    prim.GetAttribute("ig:model").Set(obj_model)

    print(f"data: {data}")

    # Store remaining data as metadata
    prim.SetCustomData(data)
    # for k, v in data.items():
    #     print(f"setting custom data {k}")
    #     print(v)
    #     print()
    #     prim.SetCustomDataByKey(k, v)
    #     # if k == "metadata":
    #     #     print(v)
    #     #     print()
    #     #     prim.SetCustomDataByKey(k, v["link_bounding_boxes"]["base_link"]["collision"]["axis_aligned"]["transform"])
    #     #     input("succeeded!")
    #     # else:
    #     #     prim.SetCustomDataByKey(k, v)

    # Add material channels
    # print(f"prim children: {prim.GetChildren()}")
    # looks_prim_path = f"{str(prim.GetPrimPath())}/Looks"
    # looks_prim = prim.GetChildren()[0] #get_prim_at_path(looks_prim_path)
    # mat_prim_path = f"{str(prim.GetPrimPath())}/Looks/material_material_0"
    # mat_prim = looks_prim.GetChildren()[0] #get_prim_at_path(mat_prim_path)
    # print(f"looks children: {looks_prim.GetChildren()}")
    # print(f"mat prim: {mat_prim}")
    if import_render_channels:
        import_rendering_channels(
            obj_prim=prim,
            obj_category=obj_category,
            obj_model=obj_model,
            model_root_path=model_root_path,
            usd_path=usd_path,
            dataset_root=dataset_root,
        )

    for link, link_tags in data["metadata"]["link_tags"].items():
        if "glass" in link_tags:
            process_glass_link(prim.GetChild(link))

    # Save stage
    stage.Save()

    # Delete stage reference and clear the sim stage variable, opening the dummy stage along the way
    del stage


def recursively_replace_list_of_dict(dic):
    for k, v in dic.items():
        print(f"k: {k}")
        if v is None:
            # Replace None
            dic[k] = Tokens.none
        elif isinstance(v, list) or isinstance(v, tuple):
            if len(v) == 0:
                dic[k] = pxr.Vt.Vec3fArray()
            elif isinstance(v[0], dict):
                # Replace with dict in place
                v = {str(i): vv for i, vv in enumerate(v)}
                dic[k] = v
            elif isinstance(v[0], list) or isinstance(v[0], tuple):
                # # Flatten the lists
                # dic[k] = []
                # for vv in v:
                #     dic[k] += vv
                print("v0: ", v[0])
                if len(v[0]) == 1:
                    # Do nothing
                    pass
                if len(v[0]) == 2:
                    dic[k] = pxr.Vt.Vec2fArray(v)
                elif len(v[0]) == 3:
                    dic[k] = pxr.Vt.Vec3fArray(v)
                elif len(v[0]) == 4:
                    dic[k] = pxr.Vt.Vec4fArray(v)
                else:
                    raise ValueError(
                        f"No support for storing matrices of length {len(v[0])}!"
                    )
            elif isinstance(v[0], int):
                # if len(v) == 1:
                #     # Do nothing
                #     pass
                # elif len(v) == 2:
                #     dic[k] = Gf.Vec2i(v)
                # elif len(v) == 3:
                #     dic[k] = Gf.Vec3i(v)
                # elif len(v) == 4:
                #     dic[k] = Gf.Vec4i(v)
                # else:
                dic[k] = pxr.Vt.IntArray(v)
                # raise ValueError(f"No support for storing numeric arrays of length {len(v)}! Array: {v}")
            elif isinstance(v[0], float):
                # if len(v) == 1:
                #     # Do nothing
                #     pass
                # elif len(v) == 2:
                #     dic[k] = Gf.Vec2f(v)
                # elif len(v) == 3:
                #     dic[k] = Gf.Vec3f(v)
                # elif len(v) == 4:
                #     dic[k] = Gf.Vec4f(v)
                # else:
                dic[k] = pxr.Vt.FloatArray(v)
                # raise ValueError(f"No support for storing numeric arrays of length {len(v)}! Array: {v}")
            else:
                # Replace any Nones
                for i, ele in enumerate(v):
                    if ele is None:
                        v[i] = Tokens.none
        if isinstance(v, dict):
            # Iterate through nested dictionaries
            dic[k] = recursively_replace_list_of_dict(v)

    return dic
