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

IMG_SIZE = 64
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
    # "room_light-0-0",
    # "room_light-0-0-L0",
]
black_list = []

def get_map_name_to_ids():
    map_name_to_ids = {}
    for item in btt.getCompatibleMapTypes():
        map_name = item.split(' : ')[0]
        class_id_str = item[item.rindex("(") + 1:item.rindex(")")]
        id_tuples = tuple([int(item) for item in class_id_str.split(', ')])
        map_name_to_ids[map_name] = id_tuples
    return map_name_to_ids

MAP_NAME_TO_IDS = get_map_name_to_ids()

def should_bake_texture(obj):
    result = parse_name(obj.name)
    # only bake texture for the first instance of a non-broken model
    return result and result.group("instance_id") == "0" and not result.group("bad")
    
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

def get_all_lights():
    lights = []
    for light in rt.lights:
        assert rt.classOf(light) == rt.VRayLight, "this light ({}) is not a VRayLight".format(light.name)
        lights.append(light)
    return lights

def uv_unwrapping(objs):
    times = {}

    unwrapped_objs = set()
    for obj in objs:
        if not should_bake_texture(obj):
            continue
        
        # Within the same object (e.g. window-0-0), there might be instances (e.g. window-0-0-leaf1-base_link-R-upper and window-0-0-leaf1-base_link-R-lower). We only want to unwrap once.
        if obj.baseObject in unwrapped_objs:
            continue

        start_time = time.time()

        print("uv_unwrapping", obj.name)
        unwrapped_objs.add(obj.baseObject)

        rt.select(obj)
        
        # Make sure it's triangle mesh
        rt.polyop.setVertSelection(obj, rt.name('all'))
        obj.connectVertices()

        # Select all faces in preparation for uv unwrap
        rt.polyop.setFaceSelection(obj, rt.name('all'))

        modifier = rt.unwrap_uvw()
        # In case the object is already uv unwrapped, we will need to assign to a new UV channel
        modifier.setMapChannel(NEW_UV_CHANNEL)

        rt.addmodifier(obj, modifier)
        modifier.flattenMapNoParams()

        end_time = time.time()
        times[obj.name] = end_time - start_time

    return times


def prepare_texture_baking(objs):
    for obj in objs:
        if not should_bake_texture(obj):
            continue

        print("prepare_texture_baking", obj.name)

        for i, map_name in enumerate(CHANNEL_MAPPING.keys()):
            texture_map = btt.addMapByClassId(obj, MAP_NAME_TO_IDS[map_name])

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

def texture_baking(bakery):
    start_time = time.time()

    btt.outputPath = bakery
    btt.autoCloseProgressDialog = True
    btt.showFrameBuffer = False
    btt.alwaysOverwriteExistingFiles = True
    print("start baking")
    assert btt.bake(), "baking failed"
    print("finish baking")

    # This will clear all the shell material and assign the newly created baked materials to the objects (in place of the original material)
    btt.clearShellKeepBaked()
    print("cleared shell material")

    end_time = time.time()
    return end_time - start_time

def export_objs(objs, base_dir):
    times = {}

    lights = get_all_lights()
    for obj in objs:
        print("export_objs", obj.name)
        start_time = time.time()

        rt.select(obj)
        obj_dir = os.path.join(base_dir, obj.name)
        os.makedirs(obj_dir, exist_ok=True)

        # WARNING: we don't know how to set fine-grained setting of OBJ export. It always inherits the setting of the last export. 
        obj_path = os.path.join(obj_dir, obj.name + ".obj")
        rt.exportFile(obj_path, pymxs.runtime.Name("noPrompt"), selectedOnly=True, using=pymxs.runtime.ObjExp)
        export_obj_metadata(obj, obj_dir, lights)

        end_time = time.time()
        times[obj.name] = end_time - start_time

    return times

def export_obj_metadata(obj, obj_dir, lights):
    print("export_objs_metadata", obj.name)

    metadata = {}
    # Export the canonical orientation
    metadata["orientation"] = [obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w]
    metadata["meta_links"] = {key: defaultdict(list) for key in ["lights"]}
    obj_name_result = parse_name(obj.name)
    for light in lights:
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


def main():
    out_dir = rt.maxops.mxsCmdLineArgs[rt.name('dir')]

    obj_out_dir = os.path.join(out_dir, "objects")
    os.makedirs(obj_out_dir, exist_ok=True)

    success = True
    error_msg = ""
    unwrap_times = {}
    export_times = {}
    baking_time = {}
    try:
        with tempfile.TemporaryDirectory() as bakery_dir:
            objs = get_process_objs()
            unwrap_times = uv_unwrapping(objs)
            prepare_texture_baking(objs)
            baking_time = texture_baking(os.path.abspath(bakery_dir))
            export_times = export_objs(objs, os.path.abspath(obj_out_dir))
    except Exception as e:
        success = False
        error_msg = traceback.format_exc()

    json_file = os.path.join(out_dir, "export_objs.json")
    with open(json_file, "w") as f:
        json.dump({"success": success, "error_msg": error_msg, "channel_map": MAP_NAME_TO_IDS, "unwrap_times": unwrap_times, "export_times": export_times, "baking_time": baking_time}, f)

    if success:
        with open(os.path.join(out_dir, "export_objs.success"), "w"):
            pass

if __name__ == "__main__":
    main()