import os
import tempfile
import traceback
from pymxs import runtime as rt
import sys
import json
import time
import re
from collections import defaultdict

btt = rt.BakeToTexture

IMG_SIZE = 512
NEW_UV_CHANNEL = 99

PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?$")
def parse_name(name):
    return PATTERN.fullmatch(name)

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
    "VRayAOMap": "Refl Color Map",  # Physical Material doesn't have a dedicated AO map
    "VRaySelfIlluminationMap": "Emission Color Map",
    "VRayRawRefractionFilterMap": "Transparency Color Map", # or VRayRefractionFilterMap: Transparency Map
    "VRayMetalnessMap": "Metalness Map",  # requires V-ray 5, update 2.3
}
# CHANNEL_MAPPING = {
#     "Color": "Base Color Map",
#     "Normal": "Bump Map",
# }

allow_list = [
    # "L-armchair-cehzwd-0"
    # "L-table_lamp-bbentu-0",
    # "floor_lamp-cosjfl-0",
    # "desk-zlyqcq-0",
    # "room_light-0-0-L0",
]
black_list = []

class ObjectExporter:
    def __init__(self, bakery, out_dir):
        self.unwrapped_objs = set()
        self.MAP_NAME_TO_IDS = self.get_map_name_to_ids()
        self.lights = self.get_all_lights()

        self.unwrap_times = {}
        self.baking_times = {}
        self.export_times = {}

        self.bakery = bakery
        os.makedirs(bakery, exist_ok=True)
        self.out_dir = out_dir
        self.obj_out_dir = os.path.join(self.out_dir, "objects")
        os.makedirs(self.obj_out_dir, exist_ok=True)

    @staticmethod
    def get_map_name_to_ids():
        map_name_to_ids = {}
        for item in btt.getCompatibleMapTypes():
            map_name = item.split(' : ')[0]
            class_id_str = item[item.rindex("(") + 1:item.rindex(")")]
            id_tuples = tuple([int(item) for item in class_id_str.split(', ')])
            map_name_to_ids[map_name] = id_tuples
        return map_name_to_ids

    @staticmethod
    def should_bake_texture(obj):
        result = parse_name(obj.name)
        # only bake texture for the first instance of a non-broken model
        return result and result.group("instance_id") == "0" and not result.group("bad")
        
    @staticmethod
    def get_process_objs():
        objs = []
        wrong_objs = []
        for obj in rt.objects:
            if rt.classOf(obj) != rt.Editable_Poly:
                if rt.classOf(obj) not in [rt.VRayLight, rt.VRayPhysicalCamera, rt.Targetobject]:
                    wrong_objs.append((obj.name, rt.ClassOf(obj)))
                continue
            if allow_list and obj.name not in allow_list:
                continue
            if black_list and obj.name in black_list:
                continue
            if parse_name(obj.name) is None:
                wrong_objs.append((obj.name, rt.ClassOf(obj)))
                continue
            objs.append(obj)
        
        if len(wrong_objs) != 0:
            for obj_name, obj_type in wrong_objs:
                print(obj_name, obj_type)
            assert False, wrong_objs

        return objs

    @staticmethod
    def get_all_lights():
        lights = []
        for light in rt.lights:
            assert rt.classOf(light) == rt.VRayLight, "this light ({}) is not a VRayLight".format(light.name)
            lights.append(light)
        return lights

    def uv_unwrapping(self, obj):
        if not self.should_bake_texture(obj):
            return
        
        # Within the same object (e.g. window-0-0), there might be instances (e.g. window-0-0-leaf1-base_link-R-upper and window-0-0-leaf1-base_link-R-lower). We only want to unwrap once.
        if obj.baseObject in self.unwrapped_objs:
            return

        start_time = time.time()

        print("uv_unwrapping", obj.name)
        self.unwrapped_objs.add(obj.baseObject)

        rt.select(obj)
        
        # Make sure it's triangle mesh
        rt.polyop.setVertSelection(obj, rt.name('all'))
        obj.connectVertices()
        rt.polyop.setVertSelection(obj, rt.name('none'))

        # Select all faces in preparation for uv unwrap
        rt.polyop.setFaceSelection(obj, rt.name('all'))

        modifier = rt.unwrap_uvw()
        rt.addmodifier(obj, modifier)

        modifier.unwrap2.setApplyToWholeObject(True)

        # In case the object is already uv unwrapped, we will need to assign to a new UV channel
        modifier.setMapChannel(NEW_UV_CHANNEL)
        modifier.unwrap2.flattenMapNoParams()

        rt.update(obj)
        rt.forceCompleteRedraw()
        rt.windows.processPostedMessages()

        end_time = time.time()
        self.unwrap_times[obj.name] = end_time - start_time

    def texture_baking(self, obj):
        vray = rt.renderers.current
        vray.camera_autoExposure = True
        vray.options_lights = False
        vray.options_hiddenLights = False

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
            texture_map.imageWidth = IMG_SIZE
            texture_map.imageHeight = IMG_SIZE
            texture_map.edgePadding = 4
            texture_map.fileType = "png"

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
        print("export_objs", obj.name)
        start_time = time.time()

        rt.select(obj)
        obj_dir = os.path.join(self.obj_out_dir, obj.name)
        os.makedirs(obj_dir, exist_ok=True)

        # WARNING: we don't know how to set fine-grained setting of OBJ export. It always inherits the setting of the last export. 
        obj_path = os.path.join(obj_dir, obj.name + ".obj")
        # rt.ObjExp.setIniName(os.path.join(os.path.parent(__file__), "gw_objexp.ini"))
        assert rt.getIniSetting(rt.ObjExp.getIniName(), "Material", "UseMapPath") == "1", "Map path not used."
        assert rt.getIniSetting(rt.ObjExp.getIniName(), "Material", "MapPath") == "./material/", "Wrong material path."
        rt.exportFile(obj_path, pymxs.runtime.Name("noPrompt"), selectedOnly=True, using=rt.ObjExp)
        assert os.path.exists(obj_path), f"Could not export object {obj.name}"
        if self.should_bake_texture(obj):
            assert os.path.exists(os.path.join(obj_dir, "material")), f"Could not export materials for object {obj.name}"
        self.export_obj_metadata(obj, obj_dir)

        end_time = time.time()
        self.export_times[obj.name] = end_time - start_time

    def export_obj_metadata(self, obj, obj_dir):
        print("export_objs_metadata", obj.name)

        metadata = {}
        # Export the canonical orientation
        metadata["orientation"] = [obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w]
        metadata["meta_links"] = {key: defaultdict(list) for key in ["lights"]}
        obj_name_result = parse_name(obj.name)
        for light in self.lights:
            light_name_result = parse_name(light.name)
            if obj_name_result.group("category") == light_name_result.group("category") and obj_name_result.group("model_id") == light_name_result.group("model_id") and obj_name_result.group("instance_id") == light_name_result.group("instance_id"):
                assert light.normalizeColor == 1, "The light's unit is NOT lm."
                # TODO: maybe always use the same intensity?
                # Reset pivot for lights (we don't trust the original pivots)
                # TODO: Don't do this. It's included in SC now. This causes issues w/ scale.
                rt.resetPivot(light)
                link_name = "base_link" if obj_name_result.group("link_name") is None else obj_name_result.group("link_name")
                metadata["meta_links"]["lights"][link_name
                ].append({
                    "type": light.type,
                    "length": light.sizeLength,
                    "width": light.sizeWidth,
                    "color": [light.color.r, light.color.g, light.color.b],
                    "intensity": light.multiplier,
                    "position": [light.position.x, light.position.y, light.position.z],
                    "orientation": [light.rotation.x, light.rotation.y, light.rotation.z, light.rotation.w],
                })

        json_file = os.path.join(obj_dir, obj.name + ".json")
        with open(json_file, "w") as f:
            json.dump(metadata, f)

    def run(self):
        assert rt.classOf(rt.renderers.current) == rt.V_Ray_5__update_2_3, f"Renderer should be set to V-Ray 5.2.3 CPU instead of {rt.classOf(rt.renderers.current)}"

        objs = self.get_process_objs()
        for obj in objs:
            self.uv_unwrapping(obj)
            self.texture_baking(obj)
            self.export_obj(obj)

def batch_main():
    rt.setvraysilentmode(True)
    out_dir = rt.maxops.mxsCmdLineArgs[rt.name('dir')]

    success = True
    error_msg = ""
    unwrap_times = {}
    export_times = {}
    baking_times = {}
    try:
        with tempfile.TemporaryDirectory() as bakery_dir:
            exp = ObjectExporter(bakery_dir, out_dir)
            exp.run()
            unwrap_times = exp.unwrap_times
            export_times = exp.export_times
            baking_times = exp.baking_times
    except:
        success = False
        error_msg = traceback.format_exc()

    json_file = os.path.join(out_dir, "export_objs.json")
    with open(json_file, "w") as f:
        json.dump({"success": success, "error_msg": error_msg, "channel_map": exp.MAP_NAME_TO_IDS, "unwrap_times": unwrap_times, "export_times": export_times, "baking_times": baking_times}, f)

    if success:
        with open(os.path.join(out_dir, "export_objs.success"), "w"):
            pass

def nonbatch_main():
    out_dir = r"D:\ig_pipeline\cad\scenes\gates_bedroom\artifacts"
    with tempfile.TemporaryDirectory() as bakery_dir:
            # bakery_dir = r"C:\Users\Cem\Downloads\bakery"
        exp = ObjectExporter(bakery_dir, out_dir)
        exp.run()

    print("Export successful!")

if __name__ == "__main__":
    # if "batch" in sys.executable:
    batch_main()
    # else:
    # nonbatch_main()