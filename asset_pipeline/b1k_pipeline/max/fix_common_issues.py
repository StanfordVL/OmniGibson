import csv
import json
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
import b1k_pipeline.max.import_fillable_meshes
import b1k_pipeline.max.extract_school_objects
import b1k_pipeline.max.prebake_textures
import b1k_pipeline.max.replace_bad_object
import b1k_pipeline.max.collision_vertex_reduction
import b1k_pipeline.max.collision_generation
import b1k_pipeline.max.match_links
from b1k_pipeline.max.merge_collision import merge_collision

rt = pymxs.runtime
RENDER_PRESET_FILENAME = str(
    (b1k_pipeline.utils.PIPELINE_ROOT / "render_presets" / "objrender.rps").absolute()
)


def get_renames():
    with open(
        b1k_pipeline.utils.PIPELINE_ROOT / "metadata/object_renames.csv", "r"
    ) as f:
        reader = csv.DictReader(f)
        renames = {}
        for row in reader:
            obj_id = row["ID (auto)"]
            old_cat = row["Original category (auto)"]
            new_cat = row["New Category"]
            in_name = f"{old_cat}-{obj_id}"
            out_name = f"{new_cat}-{obj_id}"
            renames[obj_id] = (in_name, out_name)

        return renames


RENAMES = get_renames()


def processed_fn(orig_fn: pathlib.Path):
    return orig_fn
    # return orig_fn.with_name(orig_fn.stem + '_autofix' + orig_fn.suffix)


def processFile(filename: pathlib.Path):
    # Load file, fixing the units
    print(f"\n\nProcessing {filename}")
    assert rt.loadMaxFile(str(filename), useFileUnits=False, quiet=True)

    made_any_changes = True  # for now. TODO: fix

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

    # Convert extracted school objects to BAD on these scenes
    ids_to_bad = b1k_pipeline.max.extract_school_objects.IDS_TO_MERGE
    for obj in rt.objects:
        m = b1k_pipeline.utils.parse_name(obj.name)
        if not m:
            continue
        if m.group("model_id") in ids_to_bad and not m.group("bad"):
            obj.name = "B-" + obj.name
            made_any_changes = True

    # # Delete meta links from non-zero or bad instances
    for obj in rt.objects:
        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            continue
        if not match.group("bad") and match.group("instance_id") == "0":
            continue
        if not match.group("meta_type"):
            continue
        rt.delete(obj)
        made_any_changes = True

    # # # Delete upper links from non-zero instances
    for obj in rt.objects:
        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            continue
        if not match.group("bad") and match.group("instance_id") == "0":
            continue
        if match.group("joint_side") != "upper":
            continue
        rt.delete(obj)
        made_any_changes = True

    # # Delete parts from non-zero instances
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
        made_any_changes = True

    # # Update UV unwrap to use correct attribute
    # for obj in rt.objects:
    #     # Get the attribute that's currently on there
    #     saved_hash = b1k_pipeline.max.prebake_textures.get_recorded_uv_unwrapping_hash(
    #         obj
    #     )
    #     if saved_hash is None:
    #         continue

    #     # Otherwise generate the new hash and save it
    #     hash_digest = b1k_pipeline.max.prebake_textures.hash_object(obj)

    #     # Add the new attr
    #     b1k_pipeline.max.prebake_textures.set_recorded_uv_unwrapping_hash(
    #         obj, hash_digest
    #     )

    # # Run the bad object replacement system for legacy scenes
    # # For this particular task we are only doing this to the legacy-containing scenes
    # target_name = filename.parts[-2]
    # if target_name.endswith("_int") or target_name.endswith("_garden"):
    #     comparison_data = (
    #         b1k_pipeline.max.replace_bad_object.replace_all_bad_legacy_objects_in_open_file()
    #     )
    #     print(f"Replaced {len(comparison_data)} bad objects in {filename}")
    #     with open(
    #         filename.parent / "artifacts" / "replaced_bad_objects.json", "w"
    #     ) as f:
    #         json.dump(comparison_data, f)

    # # Exit isolate mode
    rt.IsolateSelection.ExitIsolateSelectionMode()

    # # Unhide everything
    for obj in rt.objects:
        obj.isHidden = False

    # # Hide collision meshes
    for obj in rt.objects:
        if "Mcollision" in obj.name:
            obj.isHidden = True

    # Reduce collision mesh vertex counts
    # b1k_pipeline.max.collision_vertex_reduction.process_all_collision_objs()

    # Generate all missing collision meshes
    # b1k_pipeline.max.collision_generation.generate_all_missing_collision_meshes()

    # Match links
    # b1k_pipeline.max.match_links.process_all_objects()

    # Merge preexisting fillable meshes
    # processed_fillable = set()
    # while True:
    #     for obj in list(rt.objects):
    #         if not rt.classOf(obj) == rt.Editable_Poly:
    #             continue

    #         if obj in processed_fillable:
    #             continue

    #         fillable_meshes = []
    #         for child in obj.children:
    #             if "Mfillable" in child.name:
    #                 fillable_meshes.append(child)

    #         if len(fillable_meshes) < 1:
    #             continue

    #         print("Processing", obj.name, "with fillable meshes", fillable_meshes)
    #         new_fillable = merge_collision(fillable_meshes, obj)
    #         new_fillable.name = re.sub(
    #             r"(-M([a-z]+)(?:_([A-Za-z0-9]+))?(?:_([0-9]+))?)",
    #             "-Mfillable",
    #             new_fillable.name,
    #         )
    #         made_any_changes = True

    #         processed_fillable.add(obj)

    #         # Break out of the for loop so that the iteration list restarts.
    #         # Otherwise, we might get a RuntimeError due to the list changing size.
    #         break
    #     else:
    #         # Break out of the while loop
    #         break

    # Apply renames
    # for obj in rt.objects:
    #     match = b1k_pipeline.utils.parse_name(obj.name)
    #     if not match:
    #         continue

    #     category = match.group("category")
    #     model_id = match.group("model_id")
    #     rename_key = f"{category}-{model_id}"
    #     if model_id in RENAMES:
    #         old_key, new_key = RENAMES[model_id]
    #         if rename_key == old_key:
    #             obj.name = obj.name.replace(old_key, new_key)
    #             made_any_changes = True

    # Import fillable meshes
    # made_any_changes = (
    #     made_any_changes
    #     or b1k_pipeline.max.import_fillable_meshes.process_current_file()
    # )

    # Remove lights from bad objects and nonzero instances
    for light in list(rt.lights):
        match = b1k_pipeline.utils.parse_name(light.name)
        if not match:
            continue
        if match.group("bad") or match.group("instance_id") != "0":
            rt.delete(light)
            made_any_changes = True

    # Save again.
    if made_any_changes:
        new_filename = processed_fn(filename)
        rt.saveMaxFile(str(new_filename))


def fix_common_issues_in_all_files():
    candidates = [
        pathlib.Path(x)
        for x in glob.glob(r"D:\ig_pipeline\cad\scenes\school_*\processed.max")
    ]

    start_pattern = None  # specify a start pattern here to skip up to a file
    start_idx = 0
    if start_pattern:
        for i, x in enumerate(candidates):
            if start_idx in str(x):
                start_idx = i
                break

    for i, f in enumerate(tqdm.tqdm(candidates[start_idx:])):
        processFile(f)


if __name__ == "__main__":
    fix_common_issues_in_all_files()
