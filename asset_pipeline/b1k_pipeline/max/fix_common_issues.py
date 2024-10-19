import sys

sys.path.append(r"D:\ig_pipeline")

import glob
import pathlib
import random
import re
import string
from collections import defaultdict

import pymxs
import tqdm

import b1k_pipeline.utils
import b1k_pipeline.max.prebake_textures

rt = pymxs.runtime
RENDER_PRESET_FILENAME = str(
    (b1k_pipeline.utils.PIPELINE_ROOT / "render_presets" / "objrender.rps").absolute()
)


def processed_fn(orig_fn: pathlib.Path):
    return orig_fn
    # return orig_fn.with_name(orig_fn.stem + '_autofix' + orig_fn.suffix)


def processFile(filename: pathlib.Path):
    # Load file, fixing the units
    print(f"\n\nProcessing {filename}")
    assert rt.loadMaxFile(str(filename), useFileUnits=False, quiet=True)
    # assert rt.units.systemScale == 1, "System scale not set to 1mm."
    # assert rt.units.systemType == rt.Name("millimeters"), "System scale not set to 1mm."

    # Switch to Vray
    # preset_categories = rt.renderpresets.LoadCategories(RENDER_PRESET_FILENAME)
    # assert rt.renderpresets.Load(0, RENDER_PRESET_FILENAME, preset_categories)

    # Fix any bad materials
    # rt.select(rt.objects)
    # rt.convertToVRay(True)

    # # Fix layers called hallway
    # existing_layer_names = {rt.LayerManager.getLayer(x).name for x in range(rt.LayerManager.count)}
    # for layer_id in range(rt.LayerManager.count):
    #     layer = rt.LayerManager.getLayer(layer_id)
    #     if "hallway_" in layer.name:
    #         to_name = layer.name.replace("hallway_", "corridor_")
    #         assert to_name not in existing_layer_names, f"Layer {to_name} already exists"
    #         layer.setName(to_name)

    # Fix any old names
    # objs_by_model = defaultdict(list)
    # for obj in rt.objects:
    #     result = parse_name(obj.name)
    #     if result is None:
    #         print("{} does not match naming convention".format(obj.name))
    #         continue

    #     if re.fullmatch("[a-z]{6}", result.group("model_id")) is None:
    #         objs_by_model[(result.group("category"), result.group("model_id"), result.group("bad"))].append(obj)

    # for category, model_id, bad in objs_by_model:
    #     if bad:
    #         random_str = "todo" + random.choices(string.ascii_lowercase, k=2)
    #     else:
    #         random_str = "".join(
    #             random.choice(string.ascii_lowercase) for _ in range(6)
    #         )
    #     for obj in objs_by_model[(category, model_id, bad)]:
    #         old_str = "-".join([category, model_id])
    #         new_str = "-".join([category, random_str])
    #         obj.name = obj.name.replace(old_str, new_str)

    # Get all editable polies
    # for obj in tqdm.tqdm(list(rt.objects)):
    #     if rt.classOf(obj) != rt.Editable_Poly:
    #         continue

    #     # Check all faces are triangular
    #     faces_maxscript = [rt.polyop.getFaceVerts(obj, i + 1) for i in range(rt.polyop.GetNumFaces(obj))]
    #     faces = [[int(v) - 1 for v in f] for f in faces_maxscript if f is not None]
    #     if not all(len(f) == 3 for f in faces):
    #         # print("Need to triangulate", obj.name)

    #         # Turn to mesh first
    #         ttm = rt.Turn_To_Mesh()
    #         rt.addmodifier(obj, ttm)

    #         # Triangulate
    #         ttp = rt.Turn_To_Poly()
    #         ttp.limitPolySize = True
    #         ttp.maxPolySize = 3
    #         rt.addmodifier(obj, ttp)
    #         rt.maxOps.collapseNodeTo(obj, 1, True)

    #     # Check that there are no dead elements
    #     if rt.polyop.GetHasDeadStructs(obj) != 0:
    #         # Remove dead structs
    #         # print("Need to collapse", obj.name)
    #         rt.polyop.CollapseDeadStructs(obj)

    # Prebake textures
    # b1k_pipeline.max.prebake_textures.process_open_file()

    # Delete meta links from non-zero instances
    for obj in rt.objects:
        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            continue
        if match.group("instance_id") == "0":
            continue
        if not match.group("meta_type"):
            continue
        rt.delete(obj)

    # Delete upper links from non-zero instances
    for obj in rt.objects:
        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            continue
        if match.group("instance_id") == "0":
            continue
        if match.group("joint_side") != "upper":
            continue
        rt.delete(obj)

    # Delete parts from non-zero instances
    for obj in rt.objects:
        if not obj.parent:
            continue
        tags = {"subpart", "extrapart", "connectedpart"}
        if not any("T" + tag in obj.name for tag in tags):
            continue
        match = b1k_pipeline.utils.parse_name(obj.parent.name)
        if not match:
            continue
        if match.group("instance_id") == "0":
            continue
        rt.delete(obj)

    # Update UV unwrap to use correct attribute
    for obj in rt.objects:
        # Get the attribute that's currently on there
        saved_hash = b1k_pipeline.max.prebake_textures.get_recorded_uv_unwrapping_hash(
            obj
        )
        if saved_hash is None:
            continue

        # Otherwise generate the new hash and save it
        hash_digest = b1k_pipeline.max.prebake_textures.hash_object(obj)

        # Add the new attr
        b1k_pipeline.max.prebake_textures.set_recorded_uv_unwrapping_hash(
            obj, hash_digest
        )

    # # Exit isolate mode
    # rt.IsolateSelection.ExitIsolateSelectionMode()

    # # Unhide everything
    # for obj in rt.objects:
    #     obj.isHidden = False

    # # Hide collision meshes
    # for obj in rt.objects:
    #     if "Mcollision" in obj.name:
    #         obj.isHidden = True

    # Save again.
    new_filename = processed_fn(filename)
    rt.saveMaxFile(str(new_filename))


def fix_common_issues_in_all_files():
    candidates = [
        pathlib.Path(x) for x in glob.glob(r"D:\ig_pipeline\cad\*\*\processed.max")
    ]
    # has_matching_processed = [processed_fn(x).exists() for x in candidates]
    for i, f in enumerate(tqdm.tqdm(candidates)):
        processFile(f)


if __name__ == "__main__":
    fix_common_issues_in_all_files()
