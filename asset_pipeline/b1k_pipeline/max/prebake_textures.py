import pathlib
import sys
import traceback

sys.path.append(r"D:\ig_pipeline")

import os
import tempfile
import traceback
import cv2
import numpy as np
from pymxs import runtime as rt
import json
import time
import re
from collections import defaultdict
import b1k_pipeline.utils

import trimesh
from fs.zipfs import ZipFS
from fs.osfs import OSFS
from fs.tempfs import TempFS
import fs.copy

btt = rt.BakeToTexture

EXPORT_TEXTURES = False
USE_UNWRELLA = True
USE_NATIVE_EXPORT = False
IMG_SIZE = 1024
HQ_IMG_SIZE = 4096
HQ_IMG_CATEGORIES = {"floors"}
NEW_UV_CHANNEL = 99

# PBRMetalRough
# CHANNEL_MAPPING = {
#     "VRayRawDiffuseFilterMap": "Base Color Map",
#     "VRayBumpNormalsMap": "Normal Map",
#     # "VRayMtlReflectGlossinessBake": "Roughness Map",
# }

# PhysicalMaterial
CHANNEL_MAPPING = {
    "VRayRawDiffuseFilterMap": "Base Color Map",  # or VRayDiffuseFilter
    "VRayNormalsMap": "Bump Map",  # or VRayBumpNormals
    "VRayMtlReflectGlossinessBake": "Roughness Map",  # iGibson/Omniverse renderer expects we flip the glossiness map
    # "VRayAOMap": "Refl Color Map",  # Physical Material doesn't have a dedicated AO map
    # "VRaySelfIlluminationMap": "Emission Color Map",
    "VRayRawRefractionFilterMap": "Transparency Color Map",  # or VRayRefractionFilterMap: Transparency Map
    "VRayMetalnessMap": "Metalness Map",  # requires V-ray 5, update 2.3
}
# CHANNEL_MAPPING = {
#     "Color": "Base Color Map",
#     "Normal": "Bump Map",
# }

RENDER_PRESET_FILENAME = str(
    (b1k_pipeline.utils.PIPELINE_ROOT / "render_presets" / "objrender.rps").absolute()
)

allow_list = []
black_list = []


def unlight_mats_recursively(mat):
    if mat:
        if hasattr(mat, "selfIllumination_multiplier"):
            mat.selfIllumination_multiplier = 0.0
        for i in range(mat.numsubs):
            submat = mat[i]
            unlight_mats_recursively(submat)


def unlight_all_mats():
    materials = {x.material for x in rt.objects}
    for mat in materials:
        unlight_mats_recursively(mat)


class TextureBaker:
    def __init__(self, bakery):
        self.unwrapped_objs = set()
        self.MAP_NAME_TO_IDS = self.get_map_name_to_ids()

        self.unwrap_times = {}
        self.baking_times = {}

        self.bakery = bakery
        os.makedirs(bakery, exist_ok=True)

    @staticmethod
    def get_map_name_to_ids():
        map_name_to_ids = {}
        for item in btt.getCompatibleMapTypes():
            map_name = item.split(" : ")[0]
            class_id_str = item[item.rindex("(") + 1 : item.rindex(")")]
            id_tuples = tuple([int(item) for item in class_id_str.split(", ")])
            map_name_to_ids[map_name] = id_tuples
        return map_name_to_ids

    def get_process_objs(self):
        objs = []
        for obj in rt.objects:
            if rt.classOf(obj) != rt.Editable_Poly:
                continue
            if allow_list and all(
                re.fullmatch(p, obj.name) is None for p in allow_list
            ):
                continue
            if black_list and any(
                re.fullmatch(p, obj.name) is not None for p in black_list
            ):
                continue
            parsed_name = b1k_pipeline.utils.parse_name(obj.name)
            if parsed_name.group("meta_type"):
                continue
            if parsed_name.group("instance_id") != "0":
                continue
            if parsed_name.group("bad"):
                continue
            if parsed_name.group("joint_side") == "upper":
                continue

            objs.append(obj)

        return objs

    def uv_unwrapping_unwrella(self, obj):
        u = rt.Unwrella()
        rt.addmodifier(obj, u)
        u.stretch = 0.15
        u.rescale = True
        u.prerotate = True
        u.packmode = 1  # Efficient
        u.map_channel = NEW_UV_CHANNEL
        u.keep_seams = False
        u.unwrap_mode = 0

        u.padding = 2
        u.width = 1024
        u.height = 1024
        u.preview = False

        assert u.unwrap(), f"Unwrapping error w/ {obj.name}: {u.error}"

    def uv_unwrapping_native(self, obj):
        rt.select(obj)

        # Select all faces in preparation for uv unwrap
        rt.polyop.setFaceSelection(obj, rt.name("all"))

        modifier = rt.unwrap_uvw()
        rt.addmodifier(obj, modifier)

        modifier.unwrap2.setApplyToWholeObject(True)

        # In case the object is already uv unwrapped, we will need to assign to a new UV channel
        modifier.setMapChannel(NEW_UV_CHANNEL)
        modifier.unwrap2.flattenMapNoParams()

    def uv_unwrapping(self, obj):
        # Within the same object (e.g. window-0-0), there might be instances (e.g. window-0-0-leaf1-base_link-R-upper and window-0-0-leaf1-base_link-R-lower). We only want to unwrap once.
        if obj.baseObject in self.unwrapped_objs:
            return

        start_time = time.time()

        print("uv_unwrapping", obj.name)
        self.unwrapped_objs.add(obj.baseObject)

        if USE_UNWRELLA:
            self.uv_unwrapping_unwrella(obj)
        else:
            self.uv_unwrapping_native(obj)

        # TODO: Record metadata about the hash at the time of the unwrapping.
        # TODO: If it's assigned to a baked material, switch back to the unbaked material.

        rt.update(obj)
        rt.forceCompleteRedraw()
        rt.windows.processPostedMessages()

        end_time = time.time()
        self.unwrap_times[obj.name] = end_time - start_time

    def texture_baking(self, obj):
        # vray = rt.renderers.current
        # vray.camera_autoExposure = False
        # vray.options_lights = False
        # vray.options_hiddenLights = False
        rt.sceneexposurecontrol.exposurecontrol = None

        start_time = time.time()

        print("prepare_texture_baking", obj.name)

        # Clear the baking list first.
        btt.deleteAllMaps()

        for i, map_name in enumerate(CHANNEL_MAPPING.keys()):
            texture_map = btt.addMapByClassId(obj, self.MAP_NAME_TO_IDS[map_name])

            # Only the first channel requires creating the new PhysicalMaterial
            if i == 0:
                btt.setOutputTo(
                    obj, "CreateNewMaterial", material=rt.PhysicalMaterial()
                )
                # btt.setOutputTo(obj, "CreateNewMaterial", material=rt.PBRMetalRough())

            # Make sure the new UV channel is selected
            texture_map.uvChannel = NEW_UV_CHANNEL
            img_size = IMG_SIZE
            if (
                b1k_pipeline.utils.parse_name(obj.name).group("category")
                in HQ_IMG_CATEGORIES
            ):
                img_size = HQ_IMG_SIZE
            texture_map.imageWidth = img_size
            texture_map.imageHeight = img_size
            texture_map.edgePadding = 4
            texture_map.fileType = "png"

            # Set the apply color mapping option
            if texture_map.getOptionsCount() > 0:
                assert (
                    texture_map.getOptionName(1) == "Apply color mapping"
                ), "Apply color mapping option not found"
                texture_map.setOptionValue(1, False)

            # Mapping from the original channel (of VRay, Corona, etc) to the new channel of PhysicalMaterial
            texture_map.setTargetMapSlot(CHANNEL_MAPPING[map_name])

        btt.outputPath = self.bakery
        btt.autoCloseProgressDialog = True
        btt.showFrameBuffer = False
        btt.alwaysOverwriteExistingFiles = True

        print("start baking")
        assert btt.bake(), "baking failed"
        print("finish baking")
        end_time = time.time()
        self.baking_times[obj.name] = end_time - start_time

        # This will clear all the shell material and assign the newly created baked materials to the objects (in place of the original material)
        # TODO: Don't do this! We want to keep the original material.
        btt.clearShellKeepBaked()

        rt.update(obj)
        rt.forceCompleteRedraw()
        rt.windows.processPostedMessages()

    def run(self):
        # assert rt.classOf(rt.renderers.current) == rt.V_Ray_5__update_2_3, f"Renderer should be set to V-Ray 5.2.3 CPU instead of {rt.classOf(rt.renderers.current)}"
        assert rt.execute("max modify mode")

        # Remove lights from all materials
        unlight_all_mats()

        objs = self.get_process_objs()
        failures = {}
        for i, obj in enumerate(objs):
            try:
                print(f"{i+1} / {len(objs)} total")

                rt.select([obj])
                rt.IsolateSelection.EnterIsolateSelectionMode()

                obj.isHidden = False
                for child in obj.children:
                    child.isHidden = False

                # TODO: Can we bake all at once?
                self.uv_unwrapping(obj)
                self.texture_baking(obj)
            except Exception as e:
                failures[obj.name] = traceback.format_exc()

        failure_msg = "\n".join(f"{obj}: {err}" for obj, err in failures.items())
        assert len(failures) == 0, f"Some objects could not be exported:\n{failure_msg}"


def process_file(filename):
    preset_categories = rt.renderpresets.LoadCategories(RENDER_PRESET_FILENAME)
    assert rt.renderpresets.Load(0, RENDER_PRESET_FILENAME, preset_categories)

    rt.setvraysilentmode(True)

    # TODO: Assert that the bakery path is valid
    bakery_path = pathlib.Path(filename).parent / "bakery"
    exp = TextureBaker(str(bakery_path))
    exp.run()


if __name__ == "__main__":
    main()
