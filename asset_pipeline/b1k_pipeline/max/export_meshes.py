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

from fs.zipfs import ZipFS
from fs.osfs import OSFS
from fs.tempfs import TempFS
import fs.copy

btt = rt.BakeToTexture

USE_UNWRELLA = True
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
    "VRayMtlReflectGlossinessBake": "Roughness Map",   # iGibson/Omniverse renderer expects we flip the glossiness map
    # "VRayAOMap": "Refl Color Map",  # Physical Material doesn't have a dedicated AO map
    # "VRaySelfIlluminationMap": "Emission Color Map",
    "VRayRawRefractionFilterMap": "Transparency Color Map", # or VRayRefractionFilterMap: Transparency Map
    "VRayMetalnessMap": "Metalness Map",  # requires V-ray 5, update 2.3
}
# CHANNEL_MAPPING = {
#     "Color": "Base Color Map",
#     "Normal": "Bump Map",
# }

RENDER_PRESET_FILENAME = str((b1k_pipeline.utils.PIPELINE_ROOT / "render_presets" / "objrender.rps").absolute())

allow_list = [
]
black_list = [
]

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

class ObjectExporter:
    def __init__(self, bakery, obj_out_dir, export_textures=True):
        self.unwrapped_objs = set()
        self.MAP_NAME_TO_IDS = self.get_map_name_to_ids()

        self.unwrap_times = {}
        self.baking_times = {}
        self.export_times = {}

        self.bakery = bakery
        os.makedirs(bakery, exist_ok=True)
        self.obj_out_dir = obj_out_dir
        assert os.path.exists(obj_out_dir)

        self.export_textures = export_textures

    @staticmethod
    def get_map_name_to_ids():
        map_name_to_ids = {}
        for item in btt.getCompatibleMapTypes():
            map_name = item.split(' : ')[0]
            class_id_str = item[item.rindex("(") + 1:item.rindex(")")]
            id_tuples = tuple([int(item) for item in class_id_str.split(', ')])
            map_name_to_ids[map_name] = id_tuples
        return map_name_to_ids

    def should_bake_texture(self, obj):
        if not self.export_textures:
            return False

        result = b1k_pipeline.utils.parse_name(obj.name)
        # only bake texture for the first instance of a non-broken model
        return result and result.group("instance_id") == "0" and not result.group("bad") and not result.group("joint_side") == "upper"
       
    def get_process_objs(self):
        objs = []
        wrong_objs = []
        for obj in rt.objects:
            if rt.classOf(obj) != rt.Editable_Poly:
                continue
            if allow_list and all(re.fullmatch(p, obj.name) is None for p in allow_list):
                continue
            if black_list and any(re.fullmatch(p, obj.name) is not None for p in black_list):
                continue
            if b1k_pipeline.utils.parse_name(obj.name) is None:
                wrong_objs.append((obj.name, rt.ClassOf(obj)))
                continue

            obj_dir = os.path.join(self.obj_out_dir, obj.name)
            obj_file = os.path.join(obj_dir, obj.name + ".obj")
            json_file = os.path.join(obj_dir, obj.name + ".json")
            if os.path.exists(obj_file) and os.path.exists(json_file):
               continue
            objs.append(obj)
        
        for light in rt.lights:
            if b1k_pipeline.utils.parse_name(light.name) is None:
                wrong_objs.append((light.name, rt.ClassOf(light)))
        
        if len(wrong_objs) != 0:
            for obj_name, obj_type in wrong_objs:
                print(obj_name, obj_type)
            assert False, wrong_objs

        objs.sort(key=lambda obj: int(b1k_pipeline.utils.parse_name(obj.name).group("instance_id")))

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
        rt.polyop.setFaceSelection(obj, rt.name('all'))

        modifier = rt.unwrap_uvw()
        rt.addmodifier(obj, modifier)

        modifier.unwrap2.setApplyToWholeObject(True)

        # In case the object is already uv unwrapped, we will need to assign to a new UV channel
        modifier.setMapChannel(NEW_UV_CHANNEL)
        modifier.unwrap2.flattenMapNoParams()


    def uv_unwrapping(self, obj):
        if not self.should_bake_texture(obj):
            return
        
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

        if not self.should_bake_texture(obj):
            return

        start_time = time.time()

        print("prepare_texture_baking", obj.name)

        # Clear the baking list first.
        btt.deleteAllMaps()

        for i, map_name in enumerate(CHANNEL_MAPPING.keys()):
            texture_map = btt.addMapByClassId(obj, self.MAP_NAME_TO_IDS[map_name])

            # Only the first channel requires creating the new PhysicalMaterial
            if i == 0:
                btt.setOutputTo(obj, "CreateNewMaterial", material=rt.PhysicalMaterial())
                # btt.setOutputTo(obj, "CreateNewMaterial", material=rt.PBRMetalRough())

            # Make sure the new UV channel is selected
            texture_map.uvChannel = NEW_UV_CHANNEL
            img_size = IMG_SIZE
            if b1k_pipeline.utils.parse_name(obj.name).group("category") in HQ_IMG_CATEGORIES:
                img_size = HQ_IMG_SIZE
            texture_map.imageWidth = img_size
            texture_map.imageHeight = img_size
            texture_map.edgePadding = 4
            texture_map.fileType = "png"

            # Set the apply color mapping option
            if texture_map.getOptionsCount() > 0:
                assert texture_map.getOptionName(1) == 'Apply color mapping', "Apply color mapping option not found"
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
        btt.clearShellKeepBaked()

        rt.update(obj)
        rt.forceCompleteRedraw()
        rt.windows.processPostedMessages()

    def export_obj(self, obj):
        print("export_meshes", obj.name)
        start_time = time.time()

        rt.select(obj)
        obj_dir = os.path.join(self.obj_out_dir, obj.name)
        os.makedirs(obj_dir, exist_ok=True)

        # WARNING: we don't know how to set fine-grained setting of OBJ export. It always inherits the setting of the last export. 
        obj_path = os.path.join(obj_dir, obj.name + ".obj")
        # rt.ObjExp.setIniName(os.path.join(os.path.parent(__file__), "gw_objexp.ini"))
        assert rt.getIniSetting(rt.ObjExp.getIniName(), "Material", "UseMapPath") == "1", "Map path not used."
        assert rt.getIniSetting(rt.ObjExp.getIniName(), "Material", "MapPath") == "./material/", "Wrong material path."
        assert rt.getIniSetting(rt.ObjExp.getIniName(), "Geometry", "FlipZyAxis") == "0", "Should not flip axes when exporting."
        assert rt.units.systemScale == 1, "System scale not set to 1mm."
        assert rt.units.systemType == rt.Name("millimeters"), "System scale not set to 1mm."

        rt.exportFile(obj_path, rt.Name("noPrompt"), selectedOnly=True, using=rt.ObjExp)
        assert os.path.exists(obj_path), f"Could not export object {obj.name}"
        if self.should_bake_texture(obj):
            assert os.path.exists(os.path.join(obj_dir, "material")), f"Could not export materials for object {obj.name}"
        self.export_obj_metadata(obj, obj_dir)

        end_time = time.time()
        self.export_times[obj.name] = end_time - start_time

    def export_obj_metadata(self, obj, obj_dir):
        print("export_meshes_metadata", obj.name)

        metadata = {}
        # Export the canonical orientation
        metadata["orientation"] = [obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w]
        metadata["meta_links"] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        metadata["parts"] = []
        metadata["layer_name"] = obj.layer.name
        obj_name_result = b1k_pipeline.utils.parse_name(obj.name)
        assert obj_name_result, f"Unparseable object name {obj.name}"
        for light in rt.lights:
            light_name_result = b1k_pipeline.utils.parse_name(light.name)
            assert light_name_result, f"Unparseable light name {light.name}"
            if obj_name_result.group("category") == light_name_result.group("category") and obj_name_result.group("model_id") == light_name_result.group("model_id") and obj_name_result.group("instance_id") == light_name_result.group("instance_id"):
                # assert light.normalizeColor == 1, "The light's unit is NOT lm."
                assert light_name_result.group("light_id"), "The light does not have an ID."
                light_id = str(int(light_name_result.group("light_id")))
                metadata["meta_links"]["lights"][light_id]["0"] = {
                    "type": light.type,
                    "length": light.sizeLength,
                    "width": light.sizeWidth,
                    "color": [light.color.r, light.color.g, light.color.b],
                    "intensity": light.multiplier,
                    "position": [light.objecttransform.position.x, light.objecttransform.position.y, light.objecttransform.position.z],
                    "orientation": [light.objecttransform.rotation.x, light.objecttransform.rotation.y, light.objecttransform.rotation.z, light.objecttransform.rotation.w],
                }

        for child in obj.children:
            # Take care of exporting object parts.
            if rt.classOf(child) in (rt.Editable_Poly, rt.PolyMeshObject):
                metadata["parts"].append(child.name)
                continue

            is_valid_meta = rt.classOf(child) in {rt.Point, rt.Box, rt.Cylinder, rt.Sphere, rt.Cone}
            assert is_valid_meta, f"Meta link {child.name} has unexpected type {rt.classOf(child)}"

            child_name_result = b1k_pipeline.utils.parse_name(child.name)
            if not child_name_result.group("meta_type"):
                continue

            meta_info = child_name_result.group("meta_info")
            meta_type = child_name_result.group("meta_type")
            meta_id_str = child_name_result.group("meta_id")
            meta_id = "0" if meta_id_str is None else meta_id_str
            meta_subid_str = child_name_result.group("meta_subid")
            meta_subid = "0" if meta_subid_str is None else meta_subid_str
            assert meta_subid not in metadata["meta_links"][meta_type][meta_id], f"Meta subID {meta_info} is repeated in object {obj.name}"
            object_transform = child.objecttransform  # This is a 4x3 array
            position = object_transform.position
            rotation = object_transform.rotation
            scale = np.array(list(object_transform.scale))

            metadata["meta_links"][meta_type][meta_id][meta_subid] = {
                "position": list(position),
                "orientation": [rotation.x, rotation.y, rotation.z, rotation.w],
            }

            if rt.classOf(child) == rt.Sphere:
                size = np.array([child.radius, child.radius, child.radius]) * scale
                metadata["meta_links"][meta_type][meta_id][meta_subid]["type"] = "sphere"
                metadata["meta_links"][meta_type][meta_id][meta_subid]["size"] = list(size)
            elif rt.classOf(child) == rt.Box:
                size = np.array([child.width, child.length, child.height]) * scale
                metadata["meta_links"][meta_type][meta_id][meta_subid]["type"] = "box"
                metadata["meta_links"][meta_type][meta_id][meta_subid]["size"] = list(size)
            elif rt.classOf(child) == rt.Cylinder:
                size = np.array([child.radius, child.radius, child.height]) * scale
                metadata["meta_links"][meta_type][meta_id][meta_subid]["type"] = "cylinder"
                metadata["meta_links"][meta_type][meta_id][meta_subid]["size"] = list(size)
            elif rt.classOf(child) == rt.Cone:
                assert np.isclose(child.radius1, 0), f"Cone radius1 should be 0 for {child.name}"
                size = np.array([child.radius2, child.radius2, child.height]) * scale
                metadata["meta_links"][meta_type][meta_id][meta_subid]["type"] = "cone"
                metadata["meta_links"][meta_type][meta_id][meta_subid]["size"] = list(size)

        # Convert the subID to a list
        for keyed_by_id in metadata["meta_links"].values():
            for id, keyed_by_subid in keyed_by_id.items():
                found_subids = set(keyed_by_subid.keys())
                expected_subids = {str(x) for x in range(len(found_subids))}
                assert found_subids == expected_subids, f"{obj.name} has non-continuous subids {sorted(found_subids)}"
                int_keyed = {int(subid): meshes for subid, meshes in keyed_by_subid.items()}
                keyed_by_id[id] = [meshes for _, meshes in sorted(int_keyed.items())]

        json_file = os.path.join(obj_dir, obj.name + ".json")
        with open(json_file, "w") as f:
            json.dump(metadata, f)

    def fix_mtl_files(self, obj):
        obj_dir = os.path.join(self.obj_out_dir, obj.name)
        for file in os.listdir(obj_dir):
            if file.endswith(".mtl"):
                mtl_file = os.path.join(obj_dir, file)
                new_lines = []
                with open(mtl_file, "r") as f:
                    for line in f.readlines():
                        if "map" in line:
                            line = line.replace("material\\", "material/")
                            map_path = os.path.join(obj_dir, line.split(" ")[1].strip())
                            # For some reason, pybullet won't load the texture files unless we save it again with OpenCV
                            img = cv2.imread(map_path)
                            # These two maps need to be flipped
                            # glossiness -> roughness
                            # translucency -> opacity
                            if "VRayMtlReflectGlossinessBake" in map_path or "VRayRawRefractionFilterMap" in map_path:
                                print(f"Flipping {os.path.basename(map_path)}")
                                img = 1 - img.astype(float)/255

                                # Apply a sqrt here to glossiness to make it harder for things to be very shiny.
                                if "VRayMtlReflectGlossinessBake" in map_path: 
                                    img = np.sqrt(img)
                                    
                                img = (img * 255).astype(np.uint8)

                            cv2.imwrite(map_path, img)

                        new_lines.append(line)

                with open(mtl_file, "w") as f:
                    for line in new_lines:
                        f.write(line)

    def run(self):
        # assert rt.classOf(rt.renderers.current) == rt.V_Ray_5__update_2_3, f"Renderer should be set to V-Ray 5.2.3 CPU instead of {rt.classOf(rt.renderers.current)}"
        assert rt.execute('max modify mode')

        # Remove lights from all materials
        unlight_all_mats()

        objs = self.get_process_objs()
        should_bake_count = sum(1 for x in objs if self.should_bake_texture(x))
        should_bake_current = 0
        failures = {}
        for i, obj in enumerate(objs):
            try:
                print(f"{i+1} / {len(objs)} total")

                if self.should_bake_texture(obj):
                    should_bake_current += 1

                print(f"{should_bake_current} / {should_bake_count} baking")

                rt.select([obj])
                rt.IsolateSelection.EnterIsolateSelectionMode()

                # Make sure it's triangle mesh
                rt.polyop.setVertSelection(obj, rt.name('all'))
                obj.connectVertices()
                rt.polyop.setVertSelection(obj, rt.name('none'))

                obj.isHidden = False
                self.uv_unwrapping(obj)
                self.texture_baking(obj)
                self.export_obj(obj)
                self.fix_mtl_files(obj)
            except Exception as e:
                failures[obj.name] = traceback.format_exc()

        failure_msg = "\n".join(f"{obj}: {err}" for obj, err in failures.items())
        assert len(failures) == 0, f"Some objects could not be exported:\n{failure_msg}"

def main():
    preset_categories = rt.renderpresets.LoadCategories(RENDER_PRESET_FILENAME)
    assert rt.renderpresets.Load(0, RENDER_PRESET_FILENAME, preset_categories)

    rt.setvraysilentmode(True)
    out_dir = os.path.join(rt.maxFilePath, "artifacts")

    export_textures = True
    property_idx = rt.fileProperties.findProperty(rt.Name("custom"), "disableTextures")
    if property_idx != 0:
        export_textures = not rt.fileProperties.getPropertyValue(rt.Name("custom"), property_idx)

    success = True
    error_msg = ""
    unwrap_times = {}
    export_times = {}
    baking_times = {}
    try:
        with TempFS(temp_dir=r"D:\tmp") as bakery_fs, \
             TempFS(temp_dir=r"D:\tmp") as obj_out_fs, \
             ZipFS(os.path.join(out_dir, "meshes.zip"), write=True) as zip_fs:
            exp = ObjectExporter(bakery_fs.getsyspath("/"), obj_out_fs.getsyspath("/"), export_textures=export_textures)
            exp.run()

            # Copy the temp_fs to the zip_fs
            print("Move files to archive.")
            fs.copy.copy_fs(obj_out_fs, zip_fs)
            print("Finished copying.")

            unwrap_times = exp.unwrap_times
            export_times = exp.export_times
            baking_times = exp.baking_times
    except:
        success = False
        error_msg = traceback.format_exc()

    json_file = os.path.join(out_dir, "export_meshes.json")
    with open(json_file, "w") as f:
        json.dump({"success": success, "error_msg": error_msg, "unwrap_times": unwrap_times, "export_times": export_times, "baking_times": baking_times}, f)

    if success:
        print("Export successful!")
    else:
        print("Export failed.")
        print(error_msg)


if __name__ == "__main__":
    main()